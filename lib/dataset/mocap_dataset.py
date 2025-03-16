from lib.dataset import BaseDataset
import numpy as np
import os
import bisect
from lib.utils.geometry_utils import *


# MOCAP_TO_AIST =  [9,7,6,1,2,5,26,25,2,17,18,19,13,16]
MOCAP_TO_H36M =  [0,1,2,4,6,7,9,12,14,15,16,24,25,27,17,18,20]
class MOCAPDataset(BaseDataset):

    def __init__(self, cfg, estimator='spin', return_type='3D', phase='train'):

        BaseDataset.__init__(self, cfg)

        self.dataset_name = "mocap"
        self.std = cfg.TRAIN.noise_std
        self.noise_type = cfg.TRAIN.noise_type

        if phase == 'train':
            self.phase = phase  # 'train' | 'test'
        elif phase == 'test':
            self.phase = phase
        else:
            raise NotImplementedError(
                "Unknown phase type! Valid phase: [\'train\',\'test\']. You can edit the code for additional implements"
            )

        if return_type in ['3D']:  # no 2D
            self.return_type = return_type  # '3D' | '2D' | 'smpl'
        else:
            raise NotImplementedError(
                "Unknown return type! Valid phase: [\'3D\','smpl']. You can edit the code for additional implement"
            )

        if estimator in ['noise']:
            self.estimator = estimator  # 'spin' | 'eft' | 'pare'
        else:
            raise NotImplementedError(
                "Unknown estimator! Valid phase: [\'spin\',\'eft\','pare']. You can edit the code for additional implement"
            )

        print('#############################################################')
        print('You are loading the [' + self.phase + 'ing set] of dataset [' +
              self.dataset_name + ']')
        print('You are using pose esimator [' + self.estimator + ']')
        print('The type of the data is [' + self.return_type + ']')

        self.slide_window_size = cfg.MODEL.SLIDE_WINDOW_SIZE
        self.evaluate_slide_window_step=cfg.EVALUATE.SLIDE_WINDOW_STEP_SIZE

        self.base_data_path = cfg.DATASET.BASE_DIR

        try:
            ground_truth_data = np.load(f'/root/mnt/SynSP/.vscode/cmumocap/mocap/one_{self.phase}_4seconds_30.npy',
                                        allow_pickle=True)[:,0]/100 # one person
        except:
            raise ImportError("Ground-truth data do not exist!")

        self.data_len = [len(seq)-self.slide_window_size+1 if (len(seq)-self.slide_window_size)>=0 else 0 for seq in ground_truth_data]
        self.data_start_num = [
                sum(self.data_len[0:i]) for i in range(len(self.data_len))
            ]
        # for i in range(len(self.data_start_num)-2,1):
        #     if self.data_start_num[i]==self.data_start_num[i-1]:
        #         self.data_start_num[i]=self.data_start_num[i+1]

        self.frame_num = sum(self.data_len)
        print('The frame number is [' + str(self.frame_num) + ']')

        self.sequence_num = len(ground_truth_data)
        print('The sequence number is [' + str(self.sequence_num) + ']')

        
        print('#############################################################')

        self.proj_std_ratio = 115
        self.proj_data_ratio = 4/2000 * self.proj_std_ratio
        self.std = self.std / self.proj_std_ratio
        da=[-i[:,MOCAP_TO_H36M] for i in ground_truth_data] # (19832, 120, 31, 3) -> (19832, 120, 17, 3)
        self.ground_truth_data_joints_3d = [i.reshape(i.shape[0],-1) for i in da] # (19832, 120, 17, 3) -> (19832, 120, 42)

        '''
        import matplotlib.pyplot as plt  
        data = ground_truth_data[0][0]
        fig = plt.figure()  
        ax = fig.add_subplot(111)
        
        # 画出数据点  
        ax.scatter(data[:, 0], data[:, 1])  
        
        # 在每个点的右上方标出索引  
        for i in range(len(data)):  
            ax.text(data[i, 0] + 0.01, data[i, 1] + 0.01, str(i), fontsize=10)  
        
        # 设置x轴和y轴的标签  
        ax.set_aspect('equal', adjustable='box')
        plt.savefig("test.png")
        plt.close()
        '''
        self.input_dimension = self.ground_truth_data_joints_3d[0].shape[-1]

    def __len__(self):
        if self.phase == "train":
            return self.frame_num
        elif self.phase == "test":
            return self.sequence_num

    def __getitem__(self, index):

        if self.phase == "train":
            return self.get_data(index)

        elif self.phase == "test":
            return self.get_test_data(index)
        
    # def noise(self, data, scale=0.01, mask_p=1): # (b, 8, 42)
    #     data = data.reshape(data.shape[0], data.shape[1], -1, 3) # (b, 8, 14, 3)
    #     dt = (data.max(axis=2) - data.min(axis=2))[:,:,None] # (b, 8, 1, 3)
    #     noise_dx = np.random.normal(0, (scale*dt), data.shape) # (b, 8, 14, 3)
    #     mask = np.random.uniform(0,1,data.shape) < mask_p  # (b, 8, 14, 3)
    #     noise_dx *= mask
    #     data = data + noise_dx
    #     data = data.reshape(data.shape[0], data.shape[1], -1) # (b, 8, 42)
    #     return data
    def noise(self, data): # (b, 8, 42)
        if self.noise_type == "gaussian":
            noises = self.std * np.random.randn(*data.shape).astype(np.float32)
        elif self.noise_type == "uniform":
            noises = self.std * (np.random.rand(*data.shape).astype(np.float32) - 0.5)
        return data + noises

    def get_data(self, index):
        position = bisect.bisect(self.data_start_num, index)-1

        gt_data = self.ground_truth_data_joints_3d[position] # [120, 42]

        start_idx = index - self.data_start_num[position]
        end_idx = start_idx + self.slide_window_size

        gt_data = gt_data[start_idx:end_idx, :] # [8,42]
        pred_data = self.noise(gt_data[None])[0]

        return {"gt": gt_data, "pred": pred_data}


    def get_test_data(self, index):

        gt_data = self.ground_truth_data_joints_3d[index]
        ground_truth_data_len = len(gt_data)


        start_idx=np.arange(0,ground_truth_data_len-self.slide_window_size+1,self.evaluate_slide_window_step)
        gt_data_=[]
        pred_data_=[]
        for idx in start_idx:
            gt_data_.append(gt_data[idx:idx+self.slide_window_size,:])

        gt_data=np.array(gt_data_)
        np.random.seed(index)
        pred_data=self.noise(gt_data)


        return {"gt": gt_data, "pred": pred_data}