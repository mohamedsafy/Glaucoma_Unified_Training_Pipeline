import torchmetrics
import torch
from typing import Any

def get_metrics(num_classes: int, device: torch.device) -> dict[str, Any]:
    '''
    Factory function to create fresh metric instances for each validation run.
    The function return a torchmetrics Precision and Recall instances.
    '''

    
    metrics = {
        "precision": torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='none').to(device),
          "recall": torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='none').to(device)
          }
    return metrics



def calculate_metrics(pred_mask, true_mask, batch_size=None):
    """
    Calculates metrics for:
    1. Cup (Class 1)
    2. Whole Disc (Class 1 + Class 2) -> The Standard Benchmark Metric

    Calculates the following metrics:
    1. Dice
    2. IoU
    """
    iou_disc, iou_cup = 0.0, 0.0
    dice_disc, dice_cup = 0.0, 0.0
    batch_size = true_mask.size(0) if batch_size is None else batch_size
    smooth = 1e-6

    for i in range(batch_size):
        p = pred_mask[i]
        t = true_mask[i]

        # --- CUP (Class 1) ---
        p_c, t_c = (p == 1).float(), (t == 1).float()
        inter_c = (p_c * t_c).sum()
        union_c = p_c.sum() + t_c.sum() - inter_c
        iou_cup += (inter_c + smooth) / (union_c + smooth)
        dice_cup += (2. * inter_c + smooth) / (p_c.sum() + t_c.sum() + smooth)

        # --- WHOLE DISC (Class 1 OR Class 2) ---
        # We combine Rim (1) and Cup (2) to get the full circle
        p_d = ((p == 1) | (p == 2)).float()
        t_d = ((t == 1) | (t == 2)).float()

        inter_d = (p_d * t_d).sum()
        union_d = p_d.sum() + t_d.sum() - inter_d
        iou_disc += (inter_d + smooth) / (union_d + smooth)
        dice_disc += (2. * inter_d + smooth) / (p_d.sum() + t_d.sum() + smooth)

    return (iou_disc/batch_size, iou_cup/batch_size,
            dice_disc/batch_size, dice_cup/batch_size)