 # SynSP: Synergy of Smoothness and Precision in Pose Sequences Refinement (CVPR 2024)

This repo is the official implementation of "**SynSP: Synergy of Smoothness and Precision in Pose Sequences Refinement**". The code is based on the [SmoothNet](https://github.com/cure-lab/SmoothNet). 

**Under Construction...**

[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_SynSP_Synergy_of_Smoothness_and_Precision_in_Pose_Sequences_Refinement_CVPR_2024_paper.pdf)

### Major Features

- Model training and evaluation for **2D pose, 3D pose, and SMPL body representation**.
- Outstanding network architecture.
- Supporting multi-view inputs.
- Provides 100x faster Sliding Window Average Algorithm.
- More details can be found in the paper.

## Description
Predicting human pose sequences via existing pose estimators often encounters various estimation errors. Motion refinement methods aim to optimize the predicted human pose sequences from pose estimators while ensuring minimal computational overhead and latency. Prior investigations have primarily concentrated on  striking a balance between the two objectives, i.e., smoothness and precision, while optimizing the predicted pose sequences. However, it has come to our attention that the tension between these two objectives can provide additional quality cues about the predicted pose sequences. These cues, in turn, are able to aid the network in optimizing lower-quality poses. To leverage this quality information, we propose a motion refinement network, termed SynSP, to achieve a Synergy of Smoothness and Precision in the sequence refinement tasks. Moreover, SynSP can also address multi-view poses of one person simultaneously, fixing inaccuracies in predicted poses through heightened attention to similar poses from other views, thereby amplifying the resultant quality cues and overall performance.

## Getting Started

### Environment Requirement

Clone the repo:
```bash
git clone https://github.com/InvertedForest/SynSP.git
```
Create  environment:
```bash
python=3.10.8
torch=1.13.1 
torchvision=0.14.1
[Instructions](https://pytorch.org/get-started/previous-versions/)
```
Install the required packages:
```bash
pip install -r requirements.txt
```

### Prepare Data
Please refer to the [SmoothNet](https://github.com/cure-lab/SmoothNet?tab=readme-ov-file#prepare-data) for the data preparation.

### Training

Run the commands below to start training:

```shell script
python train_smoothnet.py --cfg [config file] --dataset_name [dataset name] --estimator [backbone estimator you use] --body_representation [smpl/3D/2D] --slide_window_size [slide window size]
```

For example, you can train on 3D representation of Human3.6M using backbone estimator FCN with silde window size 8 by:

```shell script
python train_smoothnet.py --cfg configs/h36m_fcn_3D.yaml --dataset_name h36m --estimator fcn --body_representation 3D --slide_window_size 8
```

You can easily train on multiple datasets using "," to split multiple datasets / estimator / body representation. For example, you can train on `AIST++` - `VIBE` - `3D` and `3DPW` - `SPIN` - `3D` with silde window size 8 by:

```shell script
python train_smoothnet.py --cfg configs/h36m_fcn_3D.yaml --dataset_name aist,pw3d --estimator vibe,spin --body_representation 3D,3D  --slide_window_size 8
```

Note that the training and testing datasets should be downloaded and prepared before training.

### Evaluation

Run the commands below to start evaluation:

```shell script
python eval_smoothnet.py --cfg [config file] --checkpoint [pretrained checkpoint] --dataset_name [dataset name] --estimator [backbone estimator you use] --body_representation [smpl/3D/2D] --slide_window_size [slide window size] --tradition [savgol/oneeuro/gaus1d]
```

For example, you can evaluate `MPI-INF-3DHP` - `TCMR` - `3D` and `MPI-INF-3DHP` - `VIBE` - `3D` using SmoothNet trained on `3DPW` - `SPIN` - `3D` with silde window size 8, and compare the results with traditional filters `oneeuro` by:

```shell script
python eval_smoothnet.py --cfg configs/pw3d_spin_3D.yaml --checkpoint data/checkpoints/pw3d_spin_3D/checkpoints_8.pth.tar --dataset_name mpiinf3dhp,mpiinf3dhp --estimator tcmr,vibe --body_representation 3D,3D --slide_window_size 8 --tradition oneeuro
```

Note that the pretrained checkpoints and testing datasets should be downloaded and prepared before evaluation.

The data and checkpoints used in our experiment can be downloaded here. 

[Google Drive](https://drive.google.com/drive/folders/1eZAXlF1cSbMuswyPCmWe-3oWS_3g6uk0?usp=sharing)

### Visualization

Here, we only provide demo visualization based on offline processed detected poses of specific datasets(e.g. AIST++, Human3.6M, and 3DPW). To visualize on arbitrary given video, please refer to the [inference/demo](https://github.com/open-mmlab/mmhuman3d/blob/main/docs/getting_started.md) of [MMHuman3D](https://github.com/open-mmlab/mmhuman3d).

un the commands below to start evaluation:

```shell script
python visualize_smoothnet.py --cfg [config file] --checkpoint [pretrained checkpoint] --dataset_name [dataset name] --estimator [backbone estimator you use] --body_representation [smpl/3D/2D] --slide_window_size [slide window size] --visualize_video_id [visualize sequence id] --output_video_path [visualization output video path]
```

For example, you can visualize the `second` sequence of `3DPW` - `SPIN` - `3D` using SmoothNet trained on `3DPW` - `SPIN` - `3D` with silde window size 32, and output the video to `./visualize` by:

```shell script
python visualize_smoothnet.py --cfg configs/pw3d_spin_3D.yaml --checkpoint data/checkpoints/pw3d_spin_3D/checkpoints_8.pth.tar --dataset_name pw3d --estimator spin --body_representation 3D --slide_window_size 32 --visualize_video_id 2 --output_video_path ./visualize
```

## License
This code is available for **non-commercial scientific research purposes** as defined in the [LICENSE file](./LICENSE). By downloading and using this code you agree to the terms in the [LICENSE](./LICENSE). Third-party datasets and software are subject to their respective licenses.
