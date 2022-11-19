import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.point import point_sample


class ISRLoss(nn.Module):
    def __init__(self):
        super(ISRLoss, self).__init__()

    def forward(self, out, target):
        pred = out['pred']
        coarse_loss = F.l1_loss(pred, target)

        points = out['points']  # (B, num_points, 2)
        rend = out['rend']  # (B, C, num_points)
        gt_points = point_sample(
            target.float(),
            points,
            mode='nearest',
            align_corners=False
        )
        points_loss = F.l1_loss(rend, gt_points)
        loss = coarse_loss + points_loss
        return loss, coarse_loss, points_loss
