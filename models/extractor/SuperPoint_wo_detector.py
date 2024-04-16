import torch
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F

from models.utils.misc import pad_and_stack

def simple_nms(scores, radius):
    """Perform non maximum suppression on the heatmap using max-pooling.
    This method does not suppress contiguous points that have the same score.
    Args:
        scores: the score heatmap of size `(B, H, W)`.
        radius: an integer scalar, the radius of the NMS window.
    """

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=radius * 2 + 1, stride=1, padding=radius
        )

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def top_k_keypoints(keypoints, scores, k):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0, sorted=True)
    return keypoints[indices], scores


def sample_k_keypoints(keypoints, scores, k):
    if k >= len(keypoints):
        return keypoints, scores
    indices = torch.multinomial(scores, k, replacement=False)
    return keypoints[indices], scores[indices]


# Legacy (broken) sampling of the descriptors
def sample_descriptors(keypoints, descriptors, s):
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor(
        [(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)],
    ).to(
        keypoints
    )[None]
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    args = {"align_corners": True} if torch.__version__ >= "1.3" else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", **args
    )
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1
    )
    return descriptors


# The original keypoint sampling is incorrect. We patch it here but
# keep the original one above for legacy.
def sample_descriptors_fix_sampling(keypoints, descriptors, s: int = 8):
    """Interpolate descriptors at keypoint locations"""
    b, c, h, w = descriptors.shape
    keypoints = keypoints / (keypoints.new_tensor([w, h]) * s)
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", align_corners=False
    )
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1
    )
    return descriptors


def soft_argmax_refinement(keypoints, scores, radius: int):
    width = 2 * radius + 1
    sum_ = torch.nn.functional.avg_pool2d(
        scores[:, None], width, 1, radius, divisor_override=1
    )
    ar = torch.arange(-radius, radius + 1).to(scores)
    kernel_x = ar[None].expand(width, -1)[None, None]
    dx = torch.nn.functional.conv2d(scores[:, None], kernel_x, padding=radius)
    dy = torch.nn.functional.conv2d(
        scores[:, None], kernel_x.transpose(2, 3), padding=radius
    )
    dydx = torch.stack([dy[:, 0], dx[:, 0]], -1) / sum_[:, 0, :, :, None]
    refined_keypoints = []
    for i, kpts in enumerate(keypoints):
        delta = dydx[i][tuple(kpts.t())]
        refined_keypoints.append(kpts.float() + delta)
    return refined_keypoints



class SuperPoint_wo_detector(Module):
    def __init__(self, conf) -> None:
        super().__init__()
        self.conf = conf #! conf: conf.model.extractor
        
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=0.5) #! CoALign_02开始采用

        c1, c2, c3, c4, c5 = 128, 128, 256, 256, 256
        # todo: 是否需要1x1卷积: 从256维降到32维
        # self.conv0 = nn.Conv2d(256, 32, kernel_size=1, stride=1, padding=0)

        self.conv1a = nn.Conv2d(128, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        
        self.descriptor = nn.Sequential(
                                        nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(inplace = True),
                                        nn.BatchNorm2d(c5, eps=0.001),
                                        nn.Conv2d(c5, self.conf.descriptor_dim, kernel_size=1, stride=1, padding=0),
                                        nn.BatchNorm2d(self.conf.descriptor_dim, eps=0.001)
        )


    def forward(self, data):
        feature, cls_mask = data    #* [b, 128, 100, 250], [N, 2, 100, 250]

        b, _, h, w = feature.shape 
        pred = {}

        #! detector
        cls_mask = F.sigmoid(cls_mask) # cls_mask要先做sigmoid
        scores, _ = torch.max(cls_mask, dim=1) #! 两个anchor的最大值 [N, 100, 252]
        dense_scores = scores # dense_scores: NMS之前的scores

        #! descriptor 
        if self.conf.has_descriptor:
            # compute the dense descriptors
            # x = self.relu(self.conv0(feature)) # 1×1卷积, 降维

            x = self.relu(self.conv1a(feature))
            x = self.relu(self.conv1b(x))
            x = self.pool(x)
            # x = self.dropout(x)
            x = self.relu(self.conv2a(x))
            x = self.relu(self.conv2b(x))
            x = self.pool(x)
            # x = self.dropout(x)
            x = self.relu(self.conv3a(x))
            x = self.relu(self.conv3b(x))
            x = self.pool(x)
            # x = self.dropout(x)
            x = self.relu(self.conv4a(x))
            x = self.relu(self.conv4b(x)) #* [b, 256, 12, 31]
            

            # cDa = self.relu(self.convDa(x))
            # dense_desc = self.convDb(cDa)
            dense_desc = self.descriptor(x)
            dense_desc = torch.nn.functional.normalize(dense_desc, p=2, dim=1)
            pred["descriptors"] = dense_desc  #* [b, 256, 12, 31] 

        if self.conf.sparse_outputs:
            # assert self.conf.has_detector and self.conf.has_descriptor

            # 判断是否需要NMS: 分辨率0.4m/pixel
            if self.conf.nms_radius > 0: # 4
                scores = simple_nms(scores, self.conf.nms_radius)  #* [B, 100, 250]
            # Discard keypoints near the image borders
            if self.conf.remove_borders:
                scores[:, : self.conf.remove_borders] = -1
                scores[:, :, : self.conf.remove_borders] = -1
                if "image_size" in data:
                    for i in range(scores.shape[0]):
                        w, h = data["image_size"][i]
                        scores[i, int(h.item()) - self.conf.remove_borders :] = -1
                        scores[i, :, int(w.item()) - self.conf.remove_borders :] = -1
                else:
                    scores[:, -self.conf.remove_borders :] = -1
                    scores[:, :, -self.conf.remove_borders :] = -1
            
            # Extract keypoints
            best_kp = torch.where(scores > self.conf.detection_threshold) #* ([],[],[]), 符合存储坐标系
            scores = scores[best_kp]   #* torch.size([124])
            # Separate into batches
            keypoints = [
                torch.stack(best_kp[1:3], dim=-1)[best_kp[0] == i] for i in range(b) #* len=16
            ]
            scores = [scores[best_kp[0] == i] for i in range(b)]

            # Keep the k keypoints with highest score
            max_kps = self.conf.max_num_keypoints

            # for val we allow different 
            if not self.training and self.conf.max_num_keypoints_val is not None:
                max_kps = self.conf.max_num_keypoints_val
        

            # Keep the k keypoints with highest score
            if max_kps > 0:
                if self.conf.randomize_keypoints_training and self.training:
                    # instead of selecting top-k, sample k by score weights
                    keypoints, scores = list(
                        zip(
                            *[
                                sample_k_keypoints(k, s, max_kps)
                                for k, s in zip(keypoints, scores)
                            ]
                        )
                    )
                else:
                    keypoints, scores = list(
                        zip(
                            *[
                                top_k_keypoints(k, s, max_kps)
                                for k, s in zip(keypoints, scores)
                            ]
                        )
                    )
                keypoints, scores = list(keypoints), list(scores)
            # 到这里keypoints仍然符合存储坐标系

            if self.conf["refinement_radius"] > 0:  #* =0, 不进行
                keypoints = soft_argmax_refinement(
                    keypoints, dense_scores, self.conf["refinement_radius"]
                )
        
        #! *********************** 这里注意: 从存储坐标系变到了x轴向右, y轴向下的像素坐标系 **************************
        # Convert (h, w) to (x, y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]
        # [[ 77.,  28.],
        # [149.,  62.],
        # [146.,  64.],
        # [ 74.,  66.],
        # [168.,  75.],
        # [218.,  88.]]

        if self.conf.force_num_keypoints: 
            keypoints = pad_and_stack(     #* [16, 100, 2]
                keypoints,
                max_kps,
                -2,
                mode="zeros",   
                # mode="random_c", # origin
                bounds=(
                    0,
                    torch.tensor(feature.shape[-2:])
                    .min()
                    .item(),
                ),
            )
            scores = pad_and_stack(scores, max_kps, -1, mode="zeros")
        else:
            keypoints = torch.stack(keypoints, 0)
            scores = torch.stack(scores, 0)

        # Extract descriptors
        if (len(keypoints) == 1):
            # Batch sampling of the descriptors
            if self.conf.legacy_sampling:
                desc = sample_descriptors(keypoints, dense_desc, 8)
            else:
                desc = sample_descriptors_fix_sampling(keypoints, dense_desc, 8)
        else:
            if self.conf.legacy_sampling: #! True
                desc = [
                    sample_descriptors(k[None], d[None], 8)[0]
                    for k, d in zip(keypoints, dense_desc)
                ]
            else:
                desc = [
                    sample_descriptors_fix_sampling(k[None], d[None], 8)[0]
                    for k, d in zip(keypoints, dense_desc)
                ]


        if isinstance(desc, list):
            desc = torch.stack(desc, 0) # [B, 256, 100]

        pred = {
            "keypoints": keypoints, #* [b, num, 2]
            # "keypoints": keypoints + 0.5,  #! 这里我觉得不用加0.5
            "keypoint_scores": scores,
            "descriptors": desc.transpose(-1, -2), #* [b, num, 256]
        }
        return pred
