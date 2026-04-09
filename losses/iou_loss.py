import torch
import torch.nn as nn


class IoULoss(nn.Module):

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        self.eps = eps

        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"Invalid reduction '{reduction}'. Choose from 'none', 'mean', 'sum'.")
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        pred_x1  = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
        pred_y1  = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
        pred_x2  = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
        pred_y2  = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

        tgt_x1   = target_boxes[:, 0] - target_boxes[:, 2] / 2
        tgt_y1   = target_boxes[:, 1] - target_boxes[:, 3] / 2
        tgt_x2   = target_boxes[:, 0] + target_boxes[:, 2] / 2
        tgt_y2   = target_boxes[:, 1] + target_boxes[:, 3] / 2

        inter_x1 = torch.max(pred_x1, tgt_x1)
        inter_y1 = torch.max(pred_y1, tgt_y1)
        inter_x2 = torch.min(pred_x2, tgt_x2)
        inter_y2 = torch.min(pred_y2, tgt_y2)

        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

        pred_area  = (pred_x2 - pred_x1).clamp(0) * (pred_y2 - pred_y1).clamp(0)
        tgt_area   = (tgt_x2  - tgt_x1).clamp(0)  * (tgt_y2  - tgt_y1).clamp(0)
        union_area = pred_area + tgt_area - inter_area + self.eps

        iou  = inter_area / union_area         
        loss = 1.0 - iou                        

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else: 
            return loss