from lib.dataset import BaseDataset
import numpy as np
import os
import bisect
from lib.models.Models_inf import _aist2mupots_3D, mupots2aist_3D

class MuPoTSDataset(BaseDataset):

    def __init__(self, cfg, estimator='tposenet', return_type='3D', phase='test'):

        BaseDataset.__init__(self, cfg)

        self.dataset_name = "mupots"

        if phase == 'test':
            self.phase = phase  # 'test'
        else:
            raise NotImplementedError(
                "Unknown phase type! Valid phase: [\'test\']. You can edit the code for additional implements"
            )

        if return_type in ['3D']:  # no 2D, 'smpl'
            self.return_type = return_type  # '3D'
        else:
            raise NotImplementedError(
                "Unknown return type! Valid phase: [\'3D\']. You can edit the code for additional implement"
            )

        if estimator in ['tposenetrefinenet','tposenet']:
            self.estimator = estimator  # 'tposenet','tposenetrefinenet'
        else:
            raise NotImplementedError(
                "Unknown estimator! Valid phase: [\'tposenetrefinenet\',\'tposenet\']. You can edit the code for additional implement"
            )

        print('#############################################################')
        print('You are loading the [' + self.phase + 'ing set] of dataset [' +
              self.dataset_name + ']')
        print('You are using pose esimator [' + str(self.estimator) + ']')
        print('The type of the data is [' + str(self.return_type) + ']')

        self.slide_window_size = cfg.MODEL.SLIDE_WINDOW_SIZE
        self.evaluate_slide_window_step = cfg.EVALUATE.SLIDE_WINDOW_STEP_SIZE

        self.base_data_path = cfg.DATASET.BASE_DIR

        try:
            ground_truth_data = np.load(os.path.join(
                self.base_data_path,
                self.dataset_name+"_"+self.estimator+"_"+self.return_type,
                "groundtruth",
                self.dataset_name + "_" + "gt"+"_"+self.return_type + "_" + self.phase + ".npz"),
                                        allow_pickle=True)
        except:
            raise ImportError("Ground-truth data do not exist!")

        try:
            detected_data = np.load(os.path.join(
                self.base_data_path, 
                self.dataset_name+"_"+self.estimator+"_"+self.return_type,
                "detected",
                self.dataset_name + "_" + self.estimator+"_"+self.return_type + "_" + self.phase + ".npz"),
                                        allow_pickle=True)
        except:
            raise ImportError("Detected data do not exist!")
            
        ground_truth_data_len = sum(
            len(seq) for seq in ground_truth_data["imgname"])
        detected_data_len = sum(len(seq) for seq in detected_data["imgname"])

        if ground_truth_data_len != detected_data_len:
            raise ImportError(
                "Detected data is not the same size with ground_truth data!")

        self.data_len = [len(seq)-self.slide_window_size if (len(seq)-self.slide_window_size)>0 else 0 for seq in ground_truth_data["imgname"]]
        self.data_start_num = [
                sum(self.data_len[0:i]) for i in range(len(self.data_len))
            ]
        for i in range(len(self.data_start_num)-2,1):
            if self.data_start_num[i]==self.data_start_num[i-1]:
                self.data_start_num[i]=self.data_start_num[i+1]


        self.frame_num = ground_truth_data_len
        print('The frame number is [' + str(self.frame_num) + ']')

        self.sequence_num = len(ground_truth_data["imgname"])
        print('The sequence number is [' + str(self.sequence_num) + ']')
        
        print('#############################################################')

        if self.return_type == '3D':
            self.ground_truth_data_imgname = ground_truth_data["imgname"]
            self.ground_truth_data_joints_3d = [da[:,mupots2aist_3D] for da in ground_truth_data["joints_3d"]]

            self.detected_data_imgname = detected_data["imgname"]
            self.detected_data_joints_3d = [da[:,mupots2aist_3D] for da in detected_data["joints_3d"]]
            self.input_dimension = self.detected_data_joints_3d[0].shape[1]

    '''
    import matplotlib.pyplot as plt  
    data = -self.ground_truth_data_joints_3d[0][0].reshape(-1,3)
    fig = plt.figure()  
    ax = fig.add_subplot(111)
    
    # 画出数据点  
    ax.scatter(data[:, 0], data[:, 1])  
    
    # 在每个点的右上方标出索引  
    for i in range(len(data)):  
        ax.text(data[i, 0] + 0.01, data[i, 1] + 0.01, str(i), fontsize=10)  
    
    # 设置x轴和y轴的比例为1:1
    ax.set_aspect('equal', adjustable='box')
    plt.savefig("test.png")
    plt.close()
    '''
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

    def get_data(self, index):
        position = bisect.bisect(self.data_start_num, index)-1

        ground_truth_data_len = len(self.ground_truth_data_imgname[position])
        detected_data_len = len(self.detected_data_imgname[position])

        if ground_truth_data_len != detected_data_len:
            raise ImportError(
                "Detected data is not the same size with ground_truth data!")

        if self.return_type == '3D':
            gt_data = self.ground_truth_data_joints_3d[position].reshape(
                ground_truth_data_len, -1, 3)
            pred_data = self.detected_data_joints_3d[position].reshape(
                ground_truth_data_len, -1, 3)

            gt_data = gt_data.reshape(ground_truth_data_len, -1)
            pred_data = pred_data.reshape(ground_truth_data_len, -1)

        if self.slide_window_size <= ground_truth_data_len:
            gt_data = np.concatenate(
                (gt_data, np.zeros(tuple((1, )) + tuple(gt_data.shape[1:]))),
                axis=0)
            pred_data = np.concatenate(
                (pred_data,
                 np.zeros(tuple((1, )) + tuple(pred_data.shape[1:]))),
                axis=0)

            start_idx = (index - self.data_start_num[position]) % (
                ground_truth_data_len - self.slide_window_size + 1)
            end_idx = start_idx + self.slide_window_size

            gt_data = gt_data[start_idx:end_idx, :]
            pred_data = pred_data[start_idx:end_idx, :]
        else:
            gt_data = np.concatenate((
                gt_data,
                np.zeros(
                    tuple((self.slide_window_size - ground_truth_data_len, )) +
                    tuple(gt_data.shape[1:]))),
                                     axis=0)
            pred_data = np.concatenate((
                pred_data,
                np.zeros(
                    tuple((self.slide_window_size - ground_truth_data_len, )) +
                    tuple(pred_data.shape[1:]))),
                                       axis=0)

        return {"gt": gt_data, "pred": pred_data}

    def get_test_data(self, index):
        ground_truth_data_len = len(self.ground_truth_data_imgname[index])
        detected_data_len = len(self.detected_data_imgname[index])

        if ground_truth_data_len != detected_data_len:
            raise ImportError(
                "Detected data is not the same size with ground_truth data!")

        if self.return_type == '3D':
            gt_data = self.ground_truth_data_joints_3d[index].reshape(
                ground_truth_data_len, -1, 3)
            pred_data = self.detected_data_joints_3d[index].reshape(
                ground_truth_data_len, -1, 3)

            gt_data = gt_data.reshape(ground_truth_data_len, -1)
            pred_data = pred_data.reshape(ground_truth_data_len, -1)

        if self.slide_window_size <= ground_truth_data_len:
            start_idx=np.arange(0,ground_truth_data_len-self.slide_window_size+1,self.evaluate_slide_window_step)
            gt_data_=[]
            pred_data_=[]
            for idx in start_idx:
                gt_data_.append(gt_data[idx:idx+self.slide_window_size,:])
                pred_data_.append(pred_data[idx:idx+self.slide_window_size,:])

            gt_data=np.array(gt_data_)
            pred_data=np.array(pred_data_)
        else:
            gt_data = np.concatenate((
                gt_data,
                np.zeros(
                    tuple((self.slide_window_size - ground_truth_data_len, )) +
                    tuple(gt_data.shape[1:]))),
                                     axis=0)[np.newaxis, :]
            pred_data = np.concatenate((
                pred_data,
                np.zeros(
                    tuple((self.slide_window_size - ground_truth_data_len, )) +
                    tuple(pred_data.shape[1:]))),
                                       axis=0)[np.newaxis, :]

        return {"gt": gt_data, "pred": pred_data}
