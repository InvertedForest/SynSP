from lib.dataset import BaseDataset
import numpy as np
import os
import bisect
from lib.utils.geometry_utils import *


H36M_IMG_SHAPE=1000


class MH36MDataset(BaseDataset):

    def __init__(self, cfg, estimator='fcn', return_type='3D', phase='train'):

        BaseDataset.__init__(self, cfg)

        self.dataset_name = "h36m" # multi-view h36m

        if phase == 'train':
            self.phase = phase  # 'train' | 'test'
        elif phase == 'test':
            self.phase = phase
        else:
            raise NotImplementedError(
                "Unknown phase type! Valid phase: [\'train\',\'test\']. You can edit the code for additional implements"
            )

        if return_type in ['3D','smpl','2D']:  
            self.return_type = return_type 
        else:
            raise NotImplementedError(
                "Unknown return type! Valid phase: [\'2D\',\'3D\',\'smpl\']. You can edit the code for additional implement"
            )

        if estimator in ['fcn','vibe','tcmr','hourglass','cpn','hrnet','rle','videoposet27','videoposet81','videoposet243']:
            self.estimator = estimator  # 'fcn'
        else:
            raise NotImplementedError(
                "Unknown estimator! Valid phase: [\'fcn\',\'vibe\',\'tcmr\',\'hourglass\',\'cpn\',\'hrnet\',\'rle\',\'videoposet27\',\'videoposet81\',\'videoposet243\']. You can edit the code for additional implement"
            )

        print('#############################################################')
        print('You are loading the [' + self.phase + 'ing set] of dataset [' +
              self.dataset_name + ']')
        print('You are using pose esimator [' + str(self.estimator) + ']')
        print('The type of the data is [' + str(self.return_type) + ']')

        self.slide_window_size = cfg.MODEL.SLIDE_WINDOW_SIZE # 8
        self.evaluate_slide_window_step = cfg.EVALUATE.SLIDE_WINDOW_STEP_SIZE # 1

        self.base_data_path = cfg.DATASET.BASE_DIR

        try:
            ground_truth_data = np.load(os.path.join( # gt数据
                self.base_data_path,
                self.dataset_name+"_"+self.estimator+"_"+self.return_type,
                "groundtruth",
                self.dataset_name + "_" + "gt"+"_"+self.return_type + "_" + self.phase + ".npz"),
                                        allow_pickle=True)
        except:
            raise ImportError("Ground-truth data do not exist!")

        try:
            detected_data = np.load(os.path.join( # 抖动数据
                self.base_data_path, 
                self.dataset_name+"_"+self.estimator+"_"+self.return_type,
                "detected",
                self.dataset_name + "_" + self.estimator+"_"+self.return_type + "_" + self.phase + ".npz"),
                                        allow_pickle=True)
        except:
            raise ImportError("Detected data do not exist!")
        
        # 对于3D的ground_truth_data和detected_data, 有以下key:
        #   name       shape          dtype
        #   imgname    [600, ]      str
        #   joints_3d  [600, **, 51]  float
        #   (600代表一个人, **代表不同长度的时间轴, 第二维度代表某个时刻的pose)
        cor_len = dict()
        cor_person = dict()
        cor_index = dict()
        ## check
        for i, name in enumerate(detected_data['imgname']): 
            camera_name = name[0].split('camera')[0]
            if camera_name not in cor_len.keys():
                cor_person[camera_name] = 1
                cor_len[camera_name] = len(name)
                cor_index[camera_name] = [i]
            else:
                cor_person[camera_name] += 1
                assert cor_len[camera_name] == len(name), 'frame numbers are not the same'
                cor_index[camera_name].append(i)
        
        
        ## transfer index
        def transfer(data):
            if self.return_type == '3D':
                coor = "joints_3d"
            if self.return_type == '2D':
                coor = "joints_2d"
            tdata = {}
            tdata["imgname"] = []
            tdata[coor] = []
            # 读取到内存，加速处理
            ndata = np.array(data["imgname"])
            ddata = np.array(data[coor])
            
            for k,v in cor_index.items():
                tdata["imgname"].append(np.stack(ndata[v], axis=1))
                min_dim = min([i.shape[0]for i in ddata[v]])
                crop_data = [i[:min_dim] for i in ddata[v]]
                tdata[coor].append(np.stack(crop_data, axis=1))
            return tdata
            

        ground_truth_data = transfer(ground_truth_data)
        detected_data = transfer(detected_data)
        # tdata = np.array(ground_truth_data["imgname"])
        # ddata = np.array(detected_data["imgname"])
        # tdata1 = np.array(ground_truth_data["joints_2d"])
        # ddata1 = np.array(detected_data["joints_2d"])
        # 对于处理后的3D的t_gt_data和t_de_data, 有以下key:
        #   name       shape          dtype
        #   imgname    [150, ]      str
        #   joints_3d  [150, 4, **, 51]  float
        #   (150代表一个场景, **代表不同长度的时间轴, 第三维度代表某个时刻的pose)


        ground_truth_data_len = sum(len(seq) for seq in ground_truth_data["imgname"]) # 代表1559752个pose
        detected_data_len     = sum(len(seq) for seq in detected_data["imgname"]) # 同上

        if ground_truth_data_len != detected_data_len:
            raise ImportError(
                "Detected data is not the same size with ground_truth data!")

        self.data_len = [len(seq)-self.slide_window_size+1 if (len(seq)-self.slide_window_size)>0 else 0 for seq in ground_truth_data["imgname"]] # 150个每个能被windows覆盖的长度
        self.data_start_num = [
                sum(self.data_len[0:i]) if i != 0 else 0 for i in range(len(self.data_len)) # 长度转start index
            ] # 累加了一下，代表这150个每个的起始位置, 整个dataset的调用相当于把这减去windows长度的150个序列拼接起来成一个序列用

        # self.frame_num = ground_truth_data_len # dataset的长度在于有几个样本,很显然要减去8之后的才是
        self.frame_num = sum(self.data_len)
        # print('The frame number is [' + str(self.frame_num) + ']')
        print('The frame number is [' + str(ground_truth_data_len) + ']')

        self.sequence_num = len(ground_truth_data["imgname"])
        print('The sequence number is [' + str(self.sequence_num) + ']')
        
        print('#############################################################')

        if self.return_type == '3D':
            self.ground_truth_data_imgname = ground_truth_data["imgname"]
            self.ground_truth_data_joints_3d = ground_truth_data["joints_3d"]
            self.detected_data_imgname = detected_data["imgname"]
            self.detected_data_joints_3d = detected_data["joints_3d"]

            self.input_dimension = ground_truth_data["joints_3d"][0].shape[-1]

        elif self.return_type == 'smpl':
            self.ground_truth_data_imgname = ground_truth_data["imgname"]
            self.ground_truth_data_pose = ground_truth_data["pose"]
            self.ground_truth_data_shape = ground_truth_data["shape"]
            self.detected_data_imgname = detected_data["imgname"]
            self.detected_data_pose = detected_data["pose"]
            self.detected_data_shape = detected_data["shape"]

            if cfg.TRAIN.USE_6D_SMPL:
                self.input_dimension = 6 * 24
                for i in range(len(self.ground_truth_data_pose)):
                    self.ground_truth_data_pose[i] = numpy_axis_to_rot6D(
                        self.ground_truth_data_pose[i].reshape(-1, 3)).reshape(
                            -1, self.input_dimension)

                for i in range(len(self.detected_data_pose)):
                    self.detected_data_pose[i] = numpy_axis_to_rot6D(
                        self.detected_data_pose[i].reshape(-1, 3)).reshape(
                            -1, self.input_dimension)
            else:
                self.input_dimension = 3 * 24

        elif self.return_type == '2D':
            self.ground_truth_data_imgname = ground_truth_data["imgname"]
            da = ground_truth_data["joints_2d"]
            self.ground_truth_data_joints_2d = [(i-H36M_IMG_SHAPE/2)/(H36M_IMG_SHAPE/2) for i in da]
            


            self.detected_data_imgname = detected_data["imgname"]
            da = detected_data["joints_2d"]
            self.detected_data_joints_2d = [(i-H36M_IMG_SHAPE/2)/(H36M_IMG_SHAPE/2) for i in da]

            self.input_dimension = ground_truth_data["joints_2d"][0].shape[-1]


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
        position = bisect.bisect(self.data_start_num, index)-1 # 找出index在序列里的位置, -1代表找的是start位置
        # 这个场景有多少帧
        ground_truth_data_len = len(self.ground_truth_data_imgname[position])
        detected_data_len = len(self.detected_data_imgname[position])

        if ground_truth_data_len != detected_data_len:
            raise ImportError(
                "Detected data is not the same size with ground_truth data!")

        if self.return_type == '3D':
            gt_data = self.ground_truth_data_joints_3d[position]
            pred_data = self.detected_data_joints_3d[position]
        elif self.return_type == '2D':
            gt_data = self.ground_truth_data_joints_2d[position]
            pred_data = self.detected_data_joints_2d[position]

        elif self.return_type == 'smpl':
            gt_data = self.ground_truth_data_pose[position].reshape(
                ground_truth_data_len, -1)
            pred_data = self.detected_data_pose[position].reshape(
                ground_truth_data_len, -1)

        if self.slide_window_size <= ground_truth_data_len:
            start_idx = index - self.data_start_num[position]
            end_idx = start_idx + self.slide_window_size

            gt_data = gt_data[start_idx:end_idx, :]
            pred_data = pred_data[start_idx:end_idx, :]
        else: #能否去掉？
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
        # assert gt_data.shape[0] == 8 or pred_data.shape[0] == 8, 'short for window'
        gt_data = gt_data.transpose(1,0,2)
        pred_data = pred_data.transpose(1,0,2)
        return {"gt": gt_data, "pred": pred_data}

    def get_test_data(self, index):
        ground_truth_data_len = len(self.ground_truth_data_imgname[index])
        detected_data_len = len(self.detected_data_imgname[index])

        if ground_truth_data_len != detected_data_len:
            raise ImportError(
                "Detected data is not the same size with ground_truth data!")

        if self.return_type == '3D':
            gt_data = self.ground_truth_data_joints_3d[index]
            pred_data = self.detected_data_joints_3d[index]
        elif self.return_type == '2D':
            gt_data = self.ground_truth_data_joints_2d[index]
            pred_data = self.detected_data_joints_2d[index]
        
        elif self.return_type == 'smpl':
            gt_data = self.ground_truth_data_pose[index].reshape(
                ground_truth_data_len, -1)
            pred_data = self.detected_data_pose[index].reshape(
                ground_truth_data_len, -1)

            gt_shape = self.ground_truth_data_shape[index].reshape(
                ground_truth_data_len, -1)
            pred_shape = self.detected_data_shape[index].reshape(
                ground_truth_data_len, -1)
            gt_data = np.concatenate((gt_data, gt_shape), axis=-1)
            pred_data = np.concatenate((pred_data, pred_shape), axis=-1)

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
        #[b,8,4,51] -> [b,4,8,51]
        gt_data = gt_data.transpose(0,2,1,3)
        pred_data = pred_data.transpose(0,2,1,3)
        return {"gt": gt_data, "pred": pred_data}
        
