import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils.geometry_utils import *
from einops import rearrange


class SmoothNetLoss(nn.Module):

    def __init__(self, w_accel, w_pos):
        super().__init__()
        self.w_accel = w_accel
        self.w_pos = w_pos

    def mask_lr1_loss(self, inputs, mask, targets):
        Bs, C, L = inputs.shape

        not_mask = 1 - mask.int()
        not_mask = not_mask.unsqueeze(1).repeat(1, C, 1).float()

        N = not_mask.sum(dtype=torch.float32)
        loss = F.l1_loss(
            inputs * not_mask, targets * not_mask, reduction='sum') / N
        return loss

    def smooth_l1_loss(self, pred, gt, beta= 0.1):
        l1_loss = torch.abs(pred - gt)
        cond = l1_loss < beta
        loss = torch.where(cond, 0.5*l1_loss**2/beta, l1_loss-0.5*beta) # 限制梯度最大为一
        loss = torch.mean(loss)
        return loss
    
    def ada_smooth_l1_loss(self, pred, gt, dt=0.01):
        l1_loss = torch.abs(pred - gt)
        beta = l1_loss.max().item()*(1-dt)
        for i in range(5):
            if (l1_loss > beta).sum() / l1_loss.numel() > dt:
                beta = beta * 1.2
            else:
                beta = beta * 0.9
        cond = l1_loss < beta
        loss = torch.where(cond, 0.5*l1_loss**2/beta, dt*(l1_loss-0.5*beta)) # 限制梯度最大为一
        loss = torch.mean(loss)
        return loss

    def get_loss(self, pred, gt, k): # [1024, 4, 8, 51]
        
        pos_loss = F.l1_loss(
            pred, gt, reduction='mean')
        # pos_loss = self.ada_smooth_l1_loss(pred, gt)



        # accel v1 norm
        accel_pred = pred[:,:-2] - 2 * pred[:,1:-1] + pred[:,2:]
        accel_gt = gt[:,:-2] - 2 * gt[:,1:-1] + gt[:,2:]

        accel_loss=F.l1_loss(
            accel_pred, accel_gt, reduction='mean')
        # accel_loss=self.ada_smooth_l1_loss(accel_pred, accel_gt)

        pos_loss = self.w_pos * pos_loss
        accel_loss = self.w_accel * accel_loss * k
        loss = pos_loss + accel_loss
        return loss

    def forward(self, denoise, gt: torch.Tensor):
        # [1024, 4, 8, 51] -> [4096, 8, 51]
        out_p, out_a, out = denoise
        loss_p = self.get_loss(out_p, gt, 0)
        loss_a = self.get_loss(out_a, gt, 1)
        loss = self.get_loss(out, gt, 0.5)
        loss = loss + loss_a + loss_p

        return loss, loss_p, loss_a
