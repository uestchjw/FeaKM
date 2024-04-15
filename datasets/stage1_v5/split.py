import os
import sys
sys.path.append("./")
from pathlib import Path
import numpy as np


#! stage1_v5数据集是在v4的基础上, train和val各取出70%作为训练集, 剩下的作为验证集
from settings import Train_data_path_v4, Val_data_path_v4
def stage1_v5_mixup():
    split_save_path = "/data/block0/hjw/Code/Where2comm+LightGlue_hjw/datasets/stage1_v5/"
    train_list = list(Path(Train_data_path_v4).iterdir())
    val_list = list(Path(Val_data_path_v4).iterdir())
    np.random.shuffle(train_list)
    np.random.shuffle(val_list)
    train_list1 = train_list[:int(0.7*len(train_list))]
    train_list2 = train_list[int(0.7*len(train_list)):]
    val_list1 = val_list[:int(0.7*len(val_list))]
    val_list2 = val_list[int(0.7*len(val_list)):]

    train_list1.extend(val_list1)  # len: 4619
    train_list2.extend(val_list2)  # len: 1981
    np.savetxt(split_save_path + "train.txt", train_list1, fmt="%s")
    np.savetxt(split_save_path + "val.txt", train_list2, fmt="%s")

if __name__ == "__main__":
    stage1_v5_mixup()