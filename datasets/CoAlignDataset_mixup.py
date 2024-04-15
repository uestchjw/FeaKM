import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
sys.path.append("./")




#! stage1_v4/5数据集格式:
# feature (2, 128, 100, 250)
# cls_preds (2, 2, 100, 250)
# reg_preds (2, 14, 100, 250)
# transformation_matrix_clean (4, 4)
#! 最大值归一化
class CoAlignDataset_mixup(Dataset):    #! 对应notion中的dataset_v5版本
    def __init__(self, mode:str) -> None:
        super().__init__()
        self.mode = mode
        self.items = []
        self.train_path = "/data/block0/hjw/Code/Where2comm+LightGlue_hjw/datasets/stage1_v5/train.txt"
        self.val_path = "/data/block0/hjw/Code/Where2comm+LightGlue_hjw/datasets/stage1_v5/val.txt"
        if mode == "train":
            self.data_path = np.loadtxt(self.train_path, dtype=str)
        elif mode == "val":
            self.data_path = np.loadtxt(self.val_path, dtype=str)
        else:
            raise Exception("Only support train or val mode for CoAlign dataset.")


    def __getitem__(self, idx):
        data_dict = np.load(self.data_path[idx], allow_pickle=True)["arr_0"].item()
        for k,v in data_dict.items():
            data_dict[k] = torch.tensor(v, dtype=torch.float32)
        
        feature = data_dict["feature"]
        vmax, _ = torch.max(feature, dim = 3, keepdim=True)
        vmax, _ = torch.max(vmax, dim = 2, keepdim=True)
        vmax, _ = torch.max(vmax, dim = 0, keepdim=True) # [1, 128, 1, 1]
        #! 最大值归一化:  -> [0, 1]
        feature /= (1e-9 + vmax) 
        data_dict["feature"] = feature
        return data_dict

    def __len__(self):
        return len(self.data_path)

if __name__ == "__main__":
    pass
