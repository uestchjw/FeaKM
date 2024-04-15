
import torch
import importlib
from pathlib import Path
import numpy as np
import time
import os
import cv2
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from tqdm import tqdm
import torch



from skimage.transform import EuclideanTransform
from skimage.measure import ransac

def to_homogeneous(points):
    """Convert N-dimensional points to homogeneous coordinates.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N).
    Returns:
        A torch.Tensor or numpy.ndarray with size (..., N+1).
    """
    if isinstance(points, torch.Tensor):
        pad = points.new_ones(points.shape[:-1] + (1,))
        return torch.cat([points, pad], dim=-1)
    elif isinstance(points, np.ndarray):
        pad = np.ones((points.shape[:-1] + (1,)), dtype=points.dtype)
        return np.concatenate([points, pad], axis=-1)
    else:
        raise ValueError
def from_homogeneous(points, eps=0.0):
    """Remove the homogeneous dimension of N-dimensional points.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N+1).
        eps: Epsilon value to prevent zero division.
    Returns:
        A torch.Tensor or numpy ndarray with size (..., N).
    """
    return points[..., :-1] / (points[..., -1:] + eps)

def warp_points_torch_hjw_v2(points, H, inverse=True):
    """
    #! points: x轴向右, y轴向下的像素坐标系
    #! H: map中心为原点, x轴向右, y轴向下的pair_T_Matrix(物理)
    Warp a list of points with the INVERSE of the given homography.
    The inverse is used to be coherent with tf.contrib.image.transform
    Arguments:
        points: batched list of N points, shape (B, N, 2).
        H: batched or not (shapes (B, 3, 3) and (3, 3) respectively).
        inverse: Whether to multiply the points by H or the inverse of H
    Returns: a Tensor of shape (B, N, 2) containing the new coordinates of the warps.
    """

    # Get the points to the homogeneous format
    points = to_homogeneous(points)

    # Apply the homography
    K = torch.tensor([[1/0.8, 0, 250/2],
                    [0, 1/0.8, 100/2],
                    [0, 0, 1]], dtype=torch.float32).cpu()
    if len(H.shape) == 2:
        M = K @ H @ torch.linalg.inv(K)
        M = M.T
    elif len(H.shape) == 3:
        M = torch.zeros_like(H)
        for i in range(H.shape[0]):
            M[i,...] = (K @ H[i, ...] @ torch.inverse(K)).T
    #* [u1, v1, 1] = H [u0, v0, 1] : u表示向右的距离, v表示向下的距离


    M = (torch.inverse(M) if inverse else M)
    warped_points = torch.einsum("...nj,...ji->...ni", points, M)

    warped_points = from_homogeneous(warped_points, eps=1e-5)
    return warped_points





Train_data_path_v4 = "/data/block0/hjw/Datasets/CoAlign+LightGlue/train_part/stage1_v4"
Val_data_path_v4 = "/data/block0/hjw/Datasets/CoAlign+LightGlue/val_part/stage1_v4"


#! stage1_v4/5数据集格式:
# feature (2, 128, 100, 250)
# cls_preds (2, 2, 100, 250)
# reg_preds (2, 14, 100, 250)
# transformation_matrix_clean (4, 4)
def generate_matrix():
    w_detector = False
    # train_data_path = "/data/block0/hjw/Code/Where2comm+LightGlue_hjw/datasets/stage1_v5/train.txt"
    val_data_path = "/data/block0/hjw/Code/Where2comm+LightGlue_hjw/datasets/stage1_v5/val.txt"

    #! 参数设置
    conf_path = "./configs/wo_detector.yaml"
    ckpt = "./outputs/training/feakm_wo/checkpoint_best.tar"
    pairs_need = 8
    visualize = False
    pred_H_save_path = f"./outputs/pred_H/least_pairs_{pairs_need}/"
    img_save_path = f"./visualization/"
    os.makedirs(pred_H_save_path, exist_ok=True)
    os.makedirs(img_save_path, exist_ok=True)

    if w_detector:
        h, w = 96, 248 # crop to 8的倍数, 原本是100, 250
    else:
        h, w = 100, 250

    data_path = np.loadtxt(val_data_path, dtype=str)
    data_path = [i for i in data_path if "val_part" in i]  


    conf = OmegaConf.load(conf_path) 
    mod = importlib.import_module("models." + conf.model.name)
    model = getattr(mod, conf.model.name)(conf)

    ckpt = torch.load(str(ckpt), map_location="cpu")
    state_dict = ckpt["model"]
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    for f_name in data_path:
        idx = str(f_name).split("/")[-1].split(".")[0] # e.g. 4276
        data_dict = np.load(f_name, allow_pickle=True)["arr_0"].item() 
        for k,v in data_dict.items():
            data_dict[k] = torch.tensor(v, dtype=torch.float32)
        feature = data_dict["feature"]
        #! 针对有detector的model, 需要调整feature的大小
        if w_detector:
            # 定义裁剪的起始点和大小
            start_x, start_y = 2, 1  # 起始点
            target_height, target_width = 96, 248  # 目标大小
            # 使用裁剪的方法调整特征图大小
            feature = feature[:, :, start_x:start_x+target_height, start_y:start_y+target_width]
        vmax, _ = torch.max(feature, dim = 3, keepdim=True)
        vmax, _ = torch.max(vmax, dim = 2, keepdim=True)
        vmax, _ = torch.max(vmax, dim = 0, keepdim=True) # [1, 128, 1, 1]
        feature /= (1e-9 + vmax) 
        data_dict["feature"] = feature
        # batch_size = 1
        data_dict["feature"] = data_dict["feature"].unsqueeze(0)
        data_dict["cls_preds"] = data_dict["cls_preds"].unsqueeze(0)
        data_dict["reg_preds"] = data_dict["reg_preds"].unsqueeze(0)
        data_dict["transformation_matrix_clean"] = data_dict["transformation_matrix_clean"].unsqueeze(0)

        pred = model(data_dict)
        kp0 = pred["keypoints0"][0,...]
        kp1 = pred["keypoints1"][0,...]
        kp0 = [mm.tolist() for mm in kp0 if torch.sum(mm)!=0] #! kp是x轴向右, y轴向下的像素坐标系
        kp1 = [mm.tolist() for mm in kp1 if torch.sum(mm)!=0] # [[181.0, 15.0], [214.0, 45.0], [206.0, 61.0]]

        src = []
        dst = []
        if not (pred["matches0"]==-1).all():
            matches0 = pred["matches0"].squeeze(0)
            matches1 = pred["matches1"].squeeze(0)
            for i in range(len(matches0)):
                if matches0[i] != -1:
                    src.append(pred["keypoints0"][0, i, :].tolist())
                    dst.append(pred["keypoints1"][0, matches0[i], :].tolist())
            assert len(src) == len(dst)
            print(f"There is {len(src)} pairs of kps.")

            if len(src) < pairs_need:
                continue
            #! GT
            transformation_matrix_clean = data_dict["transformation_matrix_clean"].clone() # (1, 4, 4)
            H_GT = transformation_matrix_clean[0, [0, 1],:][:,[0, 1, 3]] # [2, 3]
            H_GT = H_GT.type(torch.float32)
            tmp = torch.tensor([[0, 0, 1]], dtype=torch.float32)
            H_GT = torch.cat([H_GT, tmp], dim=0) # [3, 3]
            assert abs(H_GT[0, 0]) <= 1 and abs(H_GT[0, 1]) <= 1 and abs(H_GT[1, 0]) <= 1 and abs(H_GT[1, 1]) <= 1 
            print(f"H_GT=\n{H_GT}")

            K = torch.tensor([[1/0.8, 0, w/2],
                            [0, 1/0.8, h/2],
                            [0, 0, 1]], dtype=torch.float32).cpu()
            # M_GT = K @ H_GT @ torch.inverse(K)
            # # print(f"M_GT=\n{M_GT}")

            src, dst = np.array(src), np.array(dst)
            #! skimage计算H
            model_robust, inliers = ransac((src, dst), EuclideanTransform, min_samples=3, residual_threshold=1, max_trials=200)

            if not isinstance(inliers, np.ndarray) or inliers.sum() < pairs_need:
                continue
            M_pred = torch.tensor(model_robust.params, dtype=torch.float32)
            H_pred = torch.inverse(K) @ M_pred @ K
            assert not torch.isnan(H_pred).any()
            print(f"H_pred=\n{H_pred}")
            print(f"inliers/total: {inliers.sum()}/{len(src)}")

            src, dst = src[inliers, :], dst[inliers, :]
            # raise Exception

            # 可视化: 绿色的点是src, 红色的点是kp_1to0
            if visualize:
                fig = plt.figure()
                #! # CoAlign中所有的关键点
                ax1 = fig.add_subplot(211) 
                kp0, kp1 = torch.tensor(kp0), torch.tensor(kp1)
                kp_1to0 = warp_points_torch_hjw_v2(kp1, H_GT, inverse=True)

                ax1.scatter(kp0[:,0], h - kp0[:,1], c="green", label = "kp0", s=5)
                ax1.scatter(kp_1to0[:,0], h - kp_1to0[:, 1], c="red", s=3, label="kp_1to0")
                ax1.set_xlabel("x")
                ax1.set_ylabel("y")
                ax1.set_xbound(0, 250)
                ax1.set_ybound(0, 100)
                # ax1.legend()
                ax1.set_title(f"Keypoints by CoAlign of {idx}", fontsize = 10)

                #! LightGlue得到的匹配点
                ax2 = fig.add_subplot(212)
                src, dst = torch.tensor(src, dtype=torch.float32), torch.tensor(dst, dtype=torch.float32) # [n, 2]
                kp_1to0 = warp_points_torch_hjw_v2(dst, H_GT, inverse=True) # [n, 2]
                ax2.scatter(src[:,0], h - src[:,1], c="green", label = "kp0", s = 5)
                ax2.scatter(kp_1to0[:,0], h - kp_1to0[:, 1], c="red", s=3, label="kp_1to0")
                for kk in range(src.shape[0]):
                    ax2.plot( [src[kk, 0], kp_1to0[kk, 0]] , [h-src[kk, 1], h-kp_1to0[kk, 1]], c="black")
                ax2.set_xlabel("x")
                ax2.set_ylabel("y")
                ax2.set_xbound(0, 250)
                ax2.set_ybound(0, 100)
                # ax2.legend()
                ax2.set_title(f"LightGlue resuls of {idx}", fontsize = 10)
                plt.savefig( img_save_path + f"{idx}.png")
                plt.close()
            
            np.save(pred_H_save_path + f"{idx}.npy", H_pred.numpy())

if __name__ == "__main__":
    generate_matrix()


