''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
from lib.models.Layers import EncoderLayer, DecoderLayer
import torch.nn.functional as F
from einops import rearrange, repeat
from copy import deepcopy
import torch_dct as dct
_h36m2mupots_3D = [10,8,11,12,13,14,15,16,4,5,6,1,2,3,0,7,9]
# _aist2mupots_3D = [10,9,8,11,12,13,7,6,5,2,3,4,1,0]
# _mupots2aist_3D = [10,9,8,11,12,13,7,6,5,2,3,4,1,0]
                #  0  1 2 3  4  5
_mupots2aist_3D = [13,12,11,8,9,10,7,6,5,2,3,4,1,0]
                #  0  1 2 3  4  5

_mupots2h36m_3D = deepcopy(_h36m2mupots_3D)
for i,d in enumerate(_h36m2mupots_3D):
    _mupots2h36m_3D[d] = i


_aist2mupots_3D = deepcopy(_mupots2aist_3D)
for i,d in enumerate(_mupots2aist_3D):
    _aist2mupots_3D[d] = i


mupots2aist_3D = []
for i in _mupots2aist_3D:
    for j in range(3):
        mupots2aist_3D.append(3*i+j)


aist2mupots_3D = []
for i in _aist2mupots_3D:
    for j in range(3):
        aist2mupots_3D.append(3*i+j)


h36m2mupots_3D = []
for i in _h36m2mupots_3D:
    for j in range(3):
        h36m2mupots_3D.append(3*i+j)


mupots2h36m_3D = deepcopy(h36m2mupots_3D)
for i,d in enumerate(h36m2mupots_3D):
    mupots2h36m_3D[d] = i


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s, *_ = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))
        self.register_buffer('pos_table2', self._get_sinusoid_encoding_table(n_position, d_hid))
        # self.register_buffer('pos_table3', self._get_sinusoid_encoding_table(n_position, d_hid))
    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self,x,n_person):
        p=self.pos_table[:,:x.size(1)].clone().detach()
        return x + p

    def forward2(self, x, n_person):
        # if x.shape[1]==135:
        #     p=self.pos_table3[:, :int(x.shape[1]/n_person)].clone().detach()
        #     p=p.repeat(1,n_person,1)
        # else:
        p=self.pos_table2[:, :int(x.shape[1]/n_person)].clone().detach()
        p=p.repeat(1,n_person,1)
        return x + p


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200, device='cuda'):

        super().__init__()
        self.position_embeddings = nn.Embedding(n_position, d_model)
        #self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.device=device
    def forward(self, src_seq,n_person, src_mask, return_attns=False, global_feature=False):
        
        enc_slf_attn_list = []
        # -- Forward
        #src_seq = self.layer_norm(src_seq)
        if global_feature:
            enc_output = self.dropout(self.position_enc.forward2(src_seq,n_person))
            #enc_output = self.dropout(src_seq)
        else:
            enc_output = self.dropout(self.position_enc(src_seq,n_person))
        #enc_output = self.layer_norm(enc_output)
        #enc_output=self.dropout(src_seq+position_embeddings)
        #enc_output = self.dropout(self.layer_norm(enc_output))
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list


        return enc_output,


class Decoder_two(nn.Module):

    def __init__(
            self, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1,device='cuda'):

        super().__init__()

        #self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.device=device

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        dec_output = (trg_seq)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []
        dec_output1, dec_slf_attn, dec_enc_attn = dec_layer(
            dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
        dec_output2, dec_slf_attn, dec_enc_attn = dec_layer(
            dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output1, dec_output2

class Decoder(nn.Module):

    def __init__(
            self, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1,device='cuda'):

        super().__init__()

        #self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.device=device

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        dec_output = (trg_seq)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output, dec_enc_attn_list

class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, src_pad_idx=1, trg_pad_idx=1,
            d_word_vec=64, d_model=64, d_inner=512,
            n_layers=3, n_head=8, d_k=32, d_v=32, dropout=0.2, n_position=100, coord=51,
            device='cuda',*args,**kwargs):

        super().__init__()
        
        self.device=device
        self.coord=coord
        
        self.d_model=d_model
        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx
        self.dropout = nn.Dropout(p=dropout)

        self.proj_i_p = torch.nn.Sequential(
            nn.Linear(coord,d_model), # coord: 15jointsx3
            nn.Linear(d_model,d_model)
        )

        self.proj_i_a = torch.nn.Sequential(
            nn.Linear(coord,d_model), # coord: 15jointsx3
            nn.Linear(d_model,d_model)
        )


        self.proj_o_p = torch.nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.Linear(d_model*2, coord)
        )

        self.proj_o_a = torch.nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.Linear(d_model*2, coord)
        )


        self.proj_i_2 = torch.nn.Sequential(
            nn.Linear(coord,d_model), # coord: 15jointsx3
            nn.Linear(d_model,d_model)
        )

        self.proj_o_2 = torch.nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.Linear(d_model*2, coord)
        )



        self.encoder_local = Encoder(
            n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout, device=self.device)


        self.decoder = Decoder_two(
            n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout, device=self.device)

        self.decoder_2 = Decoder(
            n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout, device=self.device)


        self.encoder_global = Encoder(
            n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout, device=self.device)



        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

    def dct(self, seq):
        if seq.dtype == torch.float16:
            seq = seq.to(torch.float32)
            out = rearrange(seq, 'b p f c -> b p c f')
            out = dct.dct(out)
            out = rearrange(out, 'b p c f -> b p f c').to(torch.float16)
        else:
            out = rearrange(seq, 'b p f c -> b p c f')
            out = dct.dct(out)
            out = rearrange(out, 'b p c f -> b p f c')
        return out


    # def loop_forward(self, src_seq):
    #     p,a,o = self.forward(src_seq)


    def forward(self, src_seq):
        '''
        src_seq: local global
        '''
        # return src_seq, src_seq, src_seq
        if len(src_seq.shape)==3: 
            multi=False
            src_seq = src_seq.to(torch.float32)[:,None]
        else:
            multi=True
        n_person=src_seq.shape[1] # 4
        
        dct_src = self.dct(src_seq)
        # dct_src = src_seq.clone()

        src_em=self.proj_i_a(dct_src) # source映射, [1024, 4, 8, 51] -> [1024, 4, 8, 128]

        src_em_local = rearrange(src_em, 'b p f c -> (b p) f c')  # 开始local   合并b和人 [b, 4, 8, 128] -> [b*4, 8, 128]
        enc_local_p, *_ = self.encoder_local(src_em_local, n_person, None) # [4096, 8, 128] -> [4096, 8, 128]
        # enc_local_p = src_em_local

        src_em_global = rearrange(src_em, 'b p f c -> b (p f) c')  # 开始global   合并人和帧 [b, 4, 8, 128] -> [b, 32, 128]
        mask_other=None
        mask_dec=None
        en_global,*_ = self.encoder_global(src_em_global,n_person, mask_other, global_feature=True) # [b, 32, 128]
        en_global = repeat(en_global, 'b f e -> (b p) f e', p=n_person) #[1024, 32, 128] -> [4096, 32, 128], 为和local保持一致batch\

        temp_a=src_seq.unsqueeze(1).repeat(1,src_seq.shape[1],1,1,1) # [1024, 4, 4, 8, 51]
        temp_b=src_seq.unsqueeze(2).repeat(1,1,src_seq.shape[1],1,1) # [1024, 4, 4, 8, 51]
        c=torch.mean((temp_a-temp_b)**2,dim=-1) # [1024, 4, 4, 8]
        c=c.reshape(c.shape[0]*c.shape[1],c.shape[2]*c.shape[3],1) # [4096, 32, 1]
        c = torch.exp(-c)

        en_world=torch.cat([enc_local_p,en_global+c],dim=1) # [4096, 40, 128]
        dec_out_p, dec_out_a = self.decoder(src_em_local, None, en_world, mask_dec) # [4096, 8, 128]
        dec_out_p += src_em_local
        dec_out_a += src_em_local

        out_p=self.proj_o_p(dec_out_p) # [4096, 8, 51]
        out_p=rearrange(out_p, '(b p) f c -> b p f c', p=n_person) # [1024, 4, 8, 51]

        out_a=self.proj_o_a(dec_out_a) # [4096, 8, 51]
        out_a=rearrange(out_a, '(b p) f c -> b p f c', p=n_person) # [1024, 4, 8, 51]



        out_avg = (out_p.clone().detach() + out_a.clone().detach()) / 2 # [1024, 4, 8, 51] # 
        out_dif = (out_p.clone().detach() - out_a.clone().detach()) / 2 # [1024, 4, 8, 51] # 

        # out_avg = self.dct(out_avg[None,...])[0]
        atten = torch.exp(-(out_dif.abs() + 1e-6) / (out_dif.abs().max(dim=2, keepdim=True)[0] + 1e-5)).sum(dim=-1)/self.coord# [1024, 4, 8]
        atten = atten.permute(0,2,1).diag_embed()
        atten = repeat(atten, 'b f p o -> (b p) (o f) i', i = 1) # [4096, 32, 1]



        temp_a=out_avg.unsqueeze(1).repeat(1,out_avg.shape[1],1,1,1) # [1024, 4, 4, 8, 51]
        temp_b=out_avg.unsqueeze(2).repeat(1,1,out_avg.shape[1],1,1) # [1024, 4, 4, 8, 51]
        c=torch.mean((temp_a-temp_b)**2,dim=-1) # [1024, 4, 4, 8]
        c=c.reshape(c.shape[0]*c.shape[1],c.shape[2]*c.shape[3],1) # [4096, 32, 1]
        c = torch.exp(-c)



        en_world=torch.cat([dec_out_a,en_global+c+atten],dim=1) # [4096, 40, 128]
        dec_out, dec_attention,*_ = self.decoder_2(src_em_local, None, en_world, mask_dec) # [4096, 8, 128]
        dec_out += src_em_local

        out=self.proj_o_2(dec_out) # [4096, 8, 51]
        out=rearrange(out, '(b p) f c -> b p f c', p=n_person) # [1024, 4, 8, 51]

        if multi:
            return out, out_p, out_a
        else:
            return out_p[:,0], out_a[:,0], out[:,0]
    



