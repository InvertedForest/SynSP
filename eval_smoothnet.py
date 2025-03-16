import torch
from lib.dataset import find_dataset_using_name
from lib.core.evaluate import Evaluator
from torch.utils.data import DataLoader
from lib.utils.utils import prepare_output_dir, worker_init_fn
from lib.core.evaluate_config import parse_args
import time
from lib.models.Models_inf import Transformer

def main(cfg):
    test_datasets=[]

    all_estimator=cfg.ESTIMATOR.split(",")
    all_body_representation=cfg.BODY_REPRESENTATION.split(",")
    all_dataset=cfg.DATASET_NAME.split(",")

    for dataset_index in range(len(all_dataset)):
        estimator=all_estimator[dataset_index]
        body_representation=all_body_representation[dataset_index]
        dataset=all_dataset[dataset_index]

        dataset_class = find_dataset_using_name(dataset)

        print("Loading dataset ("+str(dataset_index)+")......")

        test_datasets.append(dataset_class(cfg,
                                    estimator=estimator,
                                    return_type=body_representation,
                                    phase='test'))
    test_loader=[]

    for test_dataset in test_datasets:
        test_loader.append(DataLoader(dataset=test_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=cfg.TRAIN.WORKERS_NUM,
                                pin_memory=True,
                                worker_init_fn=worker_init_fn))


    # # ========= Initialize networks ========= #
    dim = 4 
    if cfg.BODY_REPRESENTATION == "2D": dim = 2
    elif cfg.BODY_REPRESENTATION == "3D": dim = 3

    md = cfg.MODEL
    model = Transformer(d_word_vec=md.d_word_vec, d_model=md.d_model, d_inner=md.d_inner,
            n_layers=md.n_layers, n_head=md.n_head, d_k=md.d_k, d_v=md.d_v, dim=dim,
            coord=test_dataset.input_dimension, persons=cfg.MODEL.persons, device=cfg.DEVICE).to(cfg.DEVICE)
    checkpoint = torch.load(cfg.EVALUATE.PRETRAINED)
    model.load_state_dict(checkpoint['state_dict'], strict=True)


    evaluator = Evaluator(model=model, test_loader=test_loader, cfg=cfg)
    for i in [2]:
        evaluator.flg = i
        evaluator.run()


if __name__ == '__main__':
    cfg, cfg_file = parse_args()
    cfg = prepare_output_dir(cfg, cfg_file)
    time1 = time.time()
    main(cfg)
    print(time.time()-time1)