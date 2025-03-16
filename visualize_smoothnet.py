import os
import torch
from lib.dataset import find_dataset_using_name
from lib.models.smoothnet import SmoothNet
from lib.core.visualize_config import parse_args
from lib.visualize.visualize import Visualize
from lib.models.Models_inf import Transformer


def main(cfg):

    dataset_class = find_dataset_using_name(cfg.DATASET_NAME)
    test_dataset = dataset_class(cfg,
                                 estimator=cfg.ESTIMATOR,
                                 return_type=cfg.BODY_REPRESENTATION,
                                 phase='test')

    # # ========= Initialize networks ========= #
    md = cfg.MODEL
    # # ========= Initialize networks ========= #
    dim = 3 
    if cfg.BODY_REPRESENTATION == "2D": dim = 2
    elif cfg.BODY_REPRESENTATION == "3D": dim = 3

    md = cfg.MODEL
    model = Transformer(d_word_vec=md.d_word_vec, d_model=md.d_model, d_inner=md.d_inner,
            n_layers=md.n_layers, n_head=md.n_head, d_k=md.d_k, d_v=md.d_v, 
            coord=51, device=cfg.DEVICE).to(cfg.DEVICE)

    visualizer = Visualize(test_dataset,cfg)

    if cfg.EVALUATE.PRETRAINED != '' and os.path.isfile(
            cfg.EVALUATE.PRETRAINED):
        checkpoint = torch.load(cfg.EVALUATE.PRETRAINED)
        model.load_state_dict(checkpoint['state_dict'])
        print(f'==> Loaded pretrained model from {cfg.EVALUATE.PRETRAINED}...')
    else:
        print(f'{cfg.EVALUATE.PRETRAINED} is not a pretrained model!!!!')
        exit()
    
    visualizer.visualize(model)


if __name__ == '__main__':
    cfg, cfg_file = parse_args()

    main(cfg)