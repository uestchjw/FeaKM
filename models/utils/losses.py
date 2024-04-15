import torch
import torch.nn as nn
from omegaconf import OmegaConf


#! log_assignment: [b, 129, 129], 网络输出结果
#! weights: [b, 129, 129], weights其实就是扩展的gt_assignment
def weight_loss(log_assignment, weights, gamma=0.0):
    b, m, n = log_assignment.shape
    m -= 1
    n -= 1

    loss_sc = log_assignment * weights #! 1越多, 越准确

    num_neg0 = weights[:, :m, -1].sum(-1).clamp(min=1.0)
    num_neg1 = weights[:, -1, :n].sum(-1).clamp(min=1.0)
    num_pos = weights[:, :m, :n].sum((-1, -2)).clamp(min=1.0)  #! [16]
    # [1., 5., 1., 7., 1., 1., 2., 2., 3., 1., 8., 1., 2., 5., 3., 1.]
    # [7., 7., 1., 9., 3., 1., 2., 6., 6., 4., 8., 1., 5., 6., 3., 4.]
    # print(f"num_pos={num_pos}")
    # raise Exception
    nll_pos = -loss_sc[:, :m, :n].sum((-1, -2)) #! pred和gt重合的越多, loss就越小
    nll_pos /= num_pos.clamp(min=1.0)

    nll_neg0 = -loss_sc[:, :m, -1].sum(-1)
    nll_neg1 = -loss_sc[:, -1, :n].sum(-1)

    nll_neg = (nll_neg0 + nll_neg1) / (num_neg0 + num_neg1)

    return nll_pos, nll_neg, num_pos, (num_neg0 + num_neg1) / 2.0


class NLLLoss(nn.Module):
    default_conf = {
        "nll_balancing": 0.5,
        "gamma_f": 0.0,  # focal loss
    }

    def __init__(self, conf):
        super().__init__()
        self.conf = OmegaConf.merge(self.default_conf, conf)
        self.loss_fn = self.nll_loss

    def forward(self, pred, data, weights=None):
        log_assignment = pred["log_assignment"]  #! [b, 129, 129],  129 = num_keypoints + 1
        if weights is None:
            weights = self.loss_fn(log_assignment, data) #! weights其实就是扩展的gt_assignment: [b, 129, 129], 129 = num_keypoints + 1
        nll_pos, nll_neg, num_pos, num_neg = weight_loss(
            log_assignment, weights, gamma=self.conf.gamma_f
        )
        
        # print(f"num_pos={num_pos}")
        # [7., 7., 1., 9., 3., 1., 2., 6., 6., 4., 8., 1., 5., 6., 3., 4.]

        nll = (
            self.conf.nll_balancing * nll_pos + (1 - self.conf.nll_balancing) * nll_neg
        )

        return (
            nll,
            weights,
            {
                "assignment_nll": nll,
                "nll_pos": nll_pos,
                "nll_neg": nll_neg,
                "num_matchable": num_pos,
                "num_unmatchable": num_neg,
            },
        )

    def nll_loss(self, log_assignment, data):
        m, n = data["gt_matches0"].size(-1), data["gt_matches1"].size(-1)
        positive = data["gt_assignment"].float() #! [b, 128, 128]

        # data["gt_matches0"] #! [b, 128]
        neg0 = (data["gt_matches0"] == -1).float() #! [b, 128]
        neg1 = (data["gt_matches1"] == -1).float() #! [b, 128]

        weights = torch.zeros_like(log_assignment)
        weights[:, :m, :n] = positive

        weights[:, :m, -1] = neg0
        weights[:, -1, :m] = neg1
        return weights #! weights其实就是扩展的gt_assignment: [b, 129, 129], 129 = num_keypoints + 1
