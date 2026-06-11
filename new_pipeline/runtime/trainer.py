from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
import os
from typing import Any, Tuple

from new_pipeline.reporting.report_generator import ReportGenerator
from new_pipeline.runtime.combined_loss import CombinedLoss
from new_pipeline.runtime.data_module import DataModule


from new_pipeline.utils.metrics_utils import calculate_metrics, calculate_metrics_batched, get_metrics
import torch.cuda.amp as amp
from tqdm import tqdm
import torch
from torch import nn

import gc

def defaultdict_factory():
    return dict()

@dataclass
class Trainer:
    model: Any
    device: torch.device
    criterion: CombinedLoss
    optimizer: Any
    scheduler: Any
    scaler: Any
    data_module: DataModule
    epochs: int
    accumulation_steps: int = 1
    report_generator: ReportGenerator = None
    visualization_samples: list[str] | str | None = None  # List of sample names to visualize, 'ALL' for all samples, or None for no visualization
    @property
    def train_ds(self) -> Any:
        return self.data_module.train_ds

    @property
    def val_ds(self) -> Any:
        return self.data_module.val_ds

    @property
    def test_ds(self) -> Any:
        return self.data_module.test_ds

    @property
    def train_loader(self) -> Any:
        return self.data_module.train_loader()

    @property
    def val_loader(self) -> Any:
        return self.data_module.val_loader()

    @property
    def test_loader(self) -> Any:
        return self.data_module.test_loader()

    def train(self) -> None:
        print("🚀 Starting Training...")
        best_dice = 0.0
        for epoch in range(1, self.epochs + 1):
            # Use your train_one_epoch logic here (adapted to self.model, self.train_loader, etc.)
            train_loss,t_dice_c,t_dice_d,t_iou_c,t_iou_d,lr = self.train_one_epoch(epoch)
            val_loss, val_dice, dice_d, dice_c, iou_d, iou_c, recall, precision = self.validate(epoch)

            print(f"Epoch {epoch} Metrics:")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Val Dice CUP | DISC : {dice_c}|{dice_d}")
            print(f"Val IoU CUP | DISC : {iou_c}|{iou_d}")

            # Scheduler Step

            self.scheduler.step(val_dice)

            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(self.model.state_dict(), os.path.join(self.exp_dir, 'best_model.pth'))
                self.report_generator.log_best_epoch(
                {'epoch':epoch, 'val/dice_cup':dice_c, 'val/dice_disc': dice_d, 'val/iou_cup': iou_c, 'val/iou_disc': iou_d, 'val/precision': precision, 'val/recall': recall,
                                'train/dice_cup':t_dice_c, 'train/dice_disc': t_dice_d, 'train/iou_cup': t_iou_c, 'train/iou_disc': t_iou_d, 'val/lr':lr,
                                'val/loss': val_loss, 'train/loss': train_loss})
                print(f"⭐ New Best Dice: {best_dice:.4f} at Epoch {epoch}")

    def train_one_epoch(self, epoch: int) -> tuple[Any, ...]:
        print(f"🔄 Epoch {epoch}/{self.epochs} - Training...")
        self.model.train()
        running_loss = 0.0
        acc_metrics = {"iou_d": 0.0, "iou_c": 0.0, "dice_d": 0.0, "dice_c": 0.0}

        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch}")

        for i, (images, masks, _) in pbar:
            images = images.to(self.device).float()
            masks = masks.to(self.device)

            # Gradient Accumulation & AMP
            with amp.autocast(enabled=self.scaler is not None):
                outputs = self.model(images)
                loss =  self.criterion(outputs, masks)
                # Normalize loss if using accumulation
                loss = loss / self.accumulation_steps
            
            if self.scaler:
                self.scaler.scale(loss).backward()
                if (i + 1) % self.accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                loss.backward()
                if (i + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # Metric Tracking
            running_loss += loss.item() * self.accumulation_steps
            preds = torch.argmax(outputs, dim=1)

            # Using your existing calculate_metrics helper
            iou_d, iou_c, dice_d, dice_c = calculate_metrics_batched(outputs, masks)
            acc_metrics["iou_d"] += iou_d
            acc_metrics["iou_c"] += iou_c
            acc_metrics["dice_d"] += dice_d
            acc_metrics["dice_c"] += dice_c

            pbar.set_postfix(loss=running_loss/(i+1))

        # Epoch-end logging
        n = len(self.train_loader)
        avg_loss = running_loss / n
        self.report_generator.log_train_step(avg_loss, acc_metrics, epoch)
        return avg_loss, acc_metrics["dice_c"]/n, acc_metrics["dice_d"]/n, acc_metrics["iou_c"]/n, acc_metrics["iou_d"]/n, self.optimizer.param_groups[0]['lr']


    def validate(self, epoch: int) -> Tuple[Any, ...]:
        print(f"🔍 Epoch {epoch}/{self.epochs} - Validating...")
        self.model.eval()
        
        val_loss = 0.0
        num_classes = 3
        val_metrics = get_metrics(num_classes, self.device)
        
        running_metrics = {"iou_d": 0.0, "iou_c": 0.0, "dice_d": 0.0, "dice_c": 0.0}
        samples_to_visualize = []
        

        with torch.no_grad():
            # Enable Automatic Mixed Precision for faster inference math
            with torch.cuda.amp.autocast(enabled=(self.device == 'cuda')):
                for images, masks, names in tqdm(self.val_loader, desc='Val'):
                    images = images.to(self.device, non_blocking=True).float()
                    masks = masks.to(self.device, non_blocking=True)

                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                    val_loss += loss.item()

                    preds = torch.argmax(outputs, dim=1) 

                    # Fast GPU Vectorized calculation
                    iou_d, iou_c, dice_d, dice_c = calculate_metrics_batched(preds, masks)
                    
                    running_metrics["iou_d"] += iou_d
                    running_metrics["iou_c"] += iou_c
                    running_metrics["dice_d"] += dice_d
                    running_metrics["dice_c"] += dice_c

                    # Torchmetrics batch update
                    val_metrics['precision'].update(preds, masks)
                    val_metrics['recall'].update(preds, masks)

                    # --- OPTIMIZED LOGGING BLOCK ---
                    # Move entire batches to CPU in ONE unified operation instead of 16 individual ones
                    images_cpu = images.cpu()
                    masks_cpu = masks.cpu()
                    preds_cpu = preds.cpu()

                    # Loop entirely in host memory (RAM) without stalling the GPU
                    for name, img, msk, prd in zip(names, images_cpu, masks_cpu, preds_cpu):
                        samples_to_visualize.append({
                            'name': name,
                            'image': img,
                            'mask': msk,
                            'pred': prd
                        })
                        iou_d, iou_c, dice_d, dice_c = calculate_metrics_batched(outputs, masks)
                        self.report_generator.log_sample_progress(name, epoch, dice_c.item(), dice_d.item(), iou_c.item(), iou_d.item())

        # Average metrics over total number of batches
        num_batches = len(self.val_loader)
        avg_loss = val_loss / num_batches
        
        for key in running_metrics:
            running_metrics[key] /= num_batches

        # Final Torchmetrics computation
        results = {name: metric.compute() for name, metric in val_metrics.items()}
        avg_precision = results['precision'].mean().item()
        avg_recall = results['recall'].mean().item()

        # Log step package
        running_metrics.update({'precision': avg_precision, 'recall': avg_recall})
        self.report_generator.log_val_step(avg_loss, running_metrics,self.optimizer.param_groups[0]['lr'], epoch, samples=samples_to_visualize)

        mean_dice = (running_metrics["dice_d"] + running_metrics["dice_c"]) / 2

        return (
            avg_loss, 
            mean_dice, 
            running_metrics["dice_d"], 
            running_metrics["dice_c"], 
            running_metrics["iou_d"], 
            running_metrics["iou_c"], 
            avg_recall, 
            avg_precision
        )

    def test(self) -> tuple[Any, ...]:
        pass

    def clean(self):
        #self.data_module.clean()
        self._clean()

    def _clean(self):
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.criterion = None
        
        gc.collect()

        # 4. Clear CUDA Cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize() # Wait for all kernels to finish

        print("✨ GPU Memory Cleared.")
        
