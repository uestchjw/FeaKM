
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import Module
import torch.nn as nn
import importlib


from models.utils.losses import NLLLoss

IGNORE_FEATURE = -2
UNMATCHED_FEATURE = -1

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


#! 这里存在deepcopy的问题, 要小心
def warp_points_torch_hjw_v2(points, H, h, w, inverse=True):
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
    K = torch.tensor([[1/0.8, 0, w/2],
                    [0, 1/0.8, h/2],
                    [0, 0, 1]], dtype=torch.float32).cuda()
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




@torch.no_grad()
def gt_matches_from_homography(kp0, kp1, H, h, w, pos_th=3, neg_th=6):
    if kp0.shape[1] == 0 or kp1.shape[1] == 0:
        b_size, n_kp0 = kp0.shape[:2]
        n_kp1 = kp1.shape[1]
        assignment = torch.zeros(
            b_size, n_kp0, n_kp1, dtype=torch.bool, device=kp0.device
        )
        m0 = -torch.ones_like(kp0[:, :, 0]).long()
        m1 = -torch.ones_like(kp1[:, :, 0]).long()
        return assignment, m0, m1


    #! 14开始采用: x向右, y向下的像素坐标系
    kp0_1 = warp_points_torch_hjw_v2(kp0, H, h, w, inverse=False)
    kp1_0 = warp_points_torch_hjw_v2(kp1, H, h, w, inverse=True)

    #! ============================ 验证几何变换 =============================================
    #! 这里可以对应上了
    # h, w = 100, 252
    # B = kp0.shape[0]
    # num = 0

    # while num < B:
    #     fig, ax = plt.subplots()
    #     #* 这里的kp都是x轴向右, y轴向下的像素坐标系
    #     ax.scatter(kp0[num, :, 0].cpu(), h - kp0[num, :, 1].cpu(), c = 'red', label='kp0')
    #     ax.scatter(kp1_0[num, :, 0].cpu(), h - kp1_0[num, :, 1].cpu(), c='green', s = 8, label = 'kp1_0')
    #     plt.legend()
    #     plt.savefig(f"/data/block0/hjw/Code/Where2comm+LightGlue_hjw/val_gt/Myown/PipeLine_hjw/{num+1}.jpg")
    #     plt.close()
    #     num += 1
    # raise Exception
    #! ============================ 验证几何变换 =============================================


    # build a distance matrix of size [... x M x N]
    dist0 = torch.sum((kp0_1.unsqueeze(-2) - kp1.unsqueeze(-3)) ** 2, -1)
    dist1 = torch.sum((kp0.unsqueeze(-2) - kp1_0.unsqueeze(-3)) ** 2, -1)
    dist = torch.max(dist0, dist1)

    reward = (dist < pos_th**2).float() - (dist > neg_th**2).float()

    min0 = dist.min(-1).indices
    min1 = dist.min(-2).indices

    ismin0 = torch.zeros(dist.shape, dtype=torch.bool, device=dist.device)
    ismin1 = ismin0.clone()
    ismin0.scatter_(-1, min0.unsqueeze(-1), value=1)
    ismin1.scatter_(-2, min1.unsqueeze(-2), value=1)
    positive = ismin0 & ismin1 & (dist < pos_th**2)

    negative0 = dist0.min(-1).values > neg_th**2
    negative1 = dist1.min(-2).values > neg_th**2

    # pack the indices of positive matches
    # if -1: unmatched point
    # if -2: ignore point
    unmatched = min0.new_tensor(UNMATCHED_FEATURE)
    ignore = min0.new_tensor(IGNORE_FEATURE)
    m0 = torch.where(positive.any(-1), min0, ignore)
    m1 = torch.where(positive.any(-2), min1, ignore)
    m0 = torch.where(negative0, unmatched, m0)
    m1 = torch.where(negative1, unmatched, m1)

    return {
        "assignment": positive,
        "reward": reward,
        "matches0": m0,
        "matches1": m1,
        "matching_scores0": (m0 > -1).float(),
        "matching_scores1": (m1 > -1).float(),
        "proj_0to1": kp0_1,
        "proj_1to0": kp1_0,
    }



#! stage1_v4数据集格式:
# feature (2, 128, 100, 250)
# cls_preds (2, 2, 100, 250)
# reg_preds (2, 14, 100, 250)
# transformation_matrix_clean (4, 4)

class Pipeline(Module):
    def __init__(self, conf) -> None:
        super().__init__()
        self.conf = conf
        mod1 = importlib.import_module("models.extractor." + conf.model.extractor.name)
        self.extractor = getattr(mod1, conf.model.extractor.name)(conf.model.extractor)
        mod2 = importlib.import_module("models.matcher." + conf.model.matcher.name)
        self.matcher = getattr(mod2, conf.model.matcher.name)(conf.model.matcher)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.has_detector = conf.model.extractor.has_detector

    def forward(self, data_dict):
        feature0 = data_dict["feature"][:, 0, ...]
        feature1 = data_dict["feature"][:, 1, ...]
        cls_mask0 = data_dict["cls_preds"][:, 0, ...]
        cls_mask1 = data_dict["cls_preds"][:, 1, ...]
        
        #! extractor
        pred0 = self.extractor((feature0, cls_mask0))
        pred1 = self.extractor((feature1, cls_mask1))
        pred = {
            **{k + "0": v for k, v in pred0.items()},
            **{k + "1": v for k, v in pred1.items()},
        } 
        # keypoints:       [B, 100, 2]  绝对位置 [194.5, 4.5]
        # keypoint_scores: [B, 100]     
        # descriptors:     [B, 100, 256]

        #! matcher
        pred = {**pred, **self.matcher({**data_dict, **pred})}
        # for k,v in pred.items():
        #     print(k, v.shape)
        # matches0:  [8, 100]
        # matches1:  [8, 100]
        # matching_scores0:   [8, 100]
        # matching_scores1:   [8, 100]
        # ref_descriptors0:   [8, 9, 100, 256]
        # ref_descriptors1:   [8, 9, 100, 256]
        # log_assigment:      [8, 101, 101]
        # prune0:             [8, 100]
        # prune1:             [8, 100]
        # todo: GT怎么生成呢?
        # todo: gluefactory\models\matchers\homography_matcher.py

        return pred

    def loss(self, pred, data_dict):
        if self.has_detector: 
            h, w = 96, 248 # need crop
        else:
            h, w = 100, 250 # origin feature 
        #! 在计算loss时计算GT
        if not self.conf.model.run_gt_in_forward:
            T = data_dict["transformation_matrix_clean"] # [b, 4, 4]
            T = T[:, [0, 1],:][:,:,[0, 1, 3]] # [b, 2, 3]

            # 从[B,2,3]变为[B,3,3], 在下方添加[0,0,1]
            tmp = [torch.tensor([[0., 0., 1.]]) for _ in range(T.shape[0])]
            tmp = torch.stack(tmp, dim=0).to(self.device)
            T = torch.cat([T, tmp], dim=1) # [B, 3, 3]
            data_dict["H_0to1"] = T #* [B, 3, 3] 这是原本的pair_T_matrix,原点在map中心, x轴向右, y轴向下
            gt_pred = gt_matches_from_homography(
                                                pred["keypoints0"], 
                                                pred["keypoints1"], 
                                                data_dict["H_0to1"],
                                                h,
                                                w,
                                                pos_th = self.conf.model.th_positive,
                                                neg_th = self.conf.model.th_negative
                                                )
            pred.update({f"gt_{k}": v for k, v in gt_pred.items()})
            losses, metrics = self.matcher.loss(pred, {**pred,**data_dict})
            total = losses["total"]
            return {**losses, "total":total}, metrics

