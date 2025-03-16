# Date：2023-5-17 
# Description: For smoothnet train, eval, debug, time
# h36m_hrnet_2D h36m_cpn_2D h36m_hourglass_2D h36m_fcn_3D pw3d_pare_3D pw3d_tcmr_3D aist_spin_smpl aist_vibe_smpl aist_vibe_3D h36m_videoposet27_3D h36m_ppt_3D mpiinf3dhp_spin_3D mupots_tposenet_3D jhmdb_simplepose_2D mpiinf3dhp_tcmr_3D mocap_noise_3D
#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
IFS='_'
windows=8
# datadir=mocap_noise_3D
# datadir=aist_vibe_3D
# datadir=nh36m_fcns_3D
datadir=mh36m_fcn_3D
# datadir=aist_spin_smpl
# datadir=h36m_hourglass_2D

# datadir=aist_vibe_3D
experiment=cvpr_$datadir\_$windows
exp=($experiment)
method=${exp[2]}
dataset=${exp[1]}
represent=${exp[3]}

# experiment=h36m_aist_mocap_3D_8
# dataset=h36m,aist,mocap
# method=fcn,vibe,noise
# represent=3D,3D,3D
# experiment=h36m_mocap_3D_8
# dataset=h36m,mocap
# method=fcn,noise
# represent=3D,3D

# method=fcn,ppt
# dataset=h36m,h36m
# represent=3D,3D
# tradition=oneeuro
# # tradition=gaus1d
# tradition=savgol
# # windows=${exp[4]}
IFS=' '

gopy="python"
status0="working..."

cont="train_smoothnet.py \
    --cfg /root/mnt/SynSP/configs/cvpr/mh36m_fcn_3D.yaml \
    --dataset_name $dataset \
    --estimator $method \
    --body_representation $represent \
    --slide_window_size $windows \
    --exp_name $experiment"
# -e eval -None train -t time -d debug -Num gpu
for arg in $*
do
    case $arg in 
        "d")
            gopy="python -m debugpy --listen 0.0.0.0:5678  --wait-for-client"
            status0="waiting for vscode..."
            ;;
        "t")
            gopy="kernprof -l " # python -m line_profiler train_smoothnet.py.lprof
            # cp -n /dev/hm/bak/* /dev/shm/
            # tar -zcvf result.tar.gz result.json
            # cp result.tar.gz ~/bosfs/wangtao
            # gopy="viztracer " # vizviewer --use_external_processor result.json
            status0="timing..."
            ;;
        "e")
            cont="eval_smoothnet.py \
                --cfg configs/cvpr/aist_spin_smpl.yaml \
                --checkpoint /root/mnt/results/pw3d_aist_spin_smpl_8_26-07-2023_13-44-00!/39_checkpoint.pth.tar \
                --dataset_name $dataset \
                --estimator $method \
                --body_representation $represent \
                --slide_window_size $windows \
                --tradition $tradition "
            ;;
        "v")
            cont="visualize_smoothnet.py \
                --cfg configs/h36m_fcn_3D.yaml \
                --checkpoint /root/mnt/SynSP/results/cvpr_h36m_fcn_3D_8_09-11-2023_18-32-46/54_checkpoint.pth.tar \
                --dataset_name $dataset \
                --estimator $method \
                --body_representation $represent \
                --slide_window_size $windows
                --visualize_video_id 2 \
                --output_video_path ./"
            ;;
        [0-7])
            CUDA_VISIBLE_DEVICES=$arg
            ;;
        *)
            echo "error: wrong arg: $arg"
            exit
            ;;
    esac
done


launch="$gopy $cont"
# export CUDA_VISIBLE_DEVICES=0,1,2
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
echo $launch
printf '\033[1;32;40m %b\033[0m\n' "$status0";
$launch
echo $experiment
echo $tradition

#/root/mnt/SynSP/results/sg_h36m_hourglass_2D_8_11-07-2023_16-40-48/69_checkpoint.pth.tar
#/root/mnt/SynSP/results/normal_aist_spin_smpl_8_15-08-2023_20-37-46/69_checkpoint.pth.tar
#/root/mnt/SynSP/results/sg_h36m_fcn_3D_8_11-07-2023_16-40-49/69_checkpoint.pth.tar
# aist_spin_smpl --cfg configs/cvpr/aist_spin_smpl.yaml  --checkpoint /root/mnt/results/pw3d_aist_spin_smpl_8_26-07-2023_13-44-00!/39_checkpoint.pth.tar
# h36m_hourglass_2D --cfg configs/cvpr/h36m_hourglass_2D.yaml  --checkpoint /root/mnt/SynSP/results/cvpr_h36m_hourglass_2D_8_03-03-2025_09-51-55/69_checkpoint.pth.tar
# h36m_fcn_3D --cfg configs/cvpr/h36m_fcn_3D.yaml  --checkpoint /root/mnt/SynSP/results/cvpr_h36m_fcn_3D_8_03-03-2025_09-51-55/69_checkpoint.pth.tar
