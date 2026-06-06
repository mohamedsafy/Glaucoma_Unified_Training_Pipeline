from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from new_pipeline.reporting.report_generator import ReportGenerator
from new_pipeline.runtime.combined_loss import CombinedLoss
from new_pipeline.runtime.data_module import DataModule


from new_pipeline.utils.metrics_utils import calculate_metrics, get_metrics
import torch.cuda.amp as amp
from tqdm import tqdm
import torch
from torch import nn

import gc

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
                #torch.save(self.model.state_dict(), os.path.join(self.exp_dir, 'best_model.pth'))
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
            iou_d, iou_c, dice_d, dice_c = calculate_metrics(preds, masks)
            acc_metrics["iou_d"] += iou_d.item()
            acc_metrics["iou_c"] += iou_c.item()
            acc_metrics["dice_d"] += dice_d.item()
            acc_metrics["dice_c"] += dice_c.item()

            pbar.set_postfix(loss=running_loss/(i+1))

        # Epoch-end logging
        n = len(self.train_loader)
        avg_loss = running_loss / n
        self.report_generator.log_train_step(avg_loss, acc_metrics, epoch)
        return avg_loss, acc_metrics["dice_c"]/n, acc_metrics["dice_d"]/n, acc_metrics["iou_c"]/n, acc_metrics["iou_d"]/n, self.optimizer.param_groups[0]['lr']


    def validate(self, epoch: int) -> tuple[Any, ...]:
        print(f"🔍 Epoch {epoch}/{self.epochs} - Validating...")
        self.model.eval()
        val_loss = 0.0
        num_classes = 3
        # Use factory to get fresh metrics each validation run
        val_metrics = get_metrics(num_classes, self.device)

        metrics = {"iou_d": 0.0, "iou_c": 0.0, "dice_d": 0.0, "dice_c": 0.0, 'precision': val_metrics['precision'], 'recall': val_metrics['recall']}

        samples_to_visualize = []

        with torch.no_grad():
            for images, masks, names in tqdm(self.val_loader, desc='Val'):
                images = images.to(self.device).float()
                masks = masks.to(self.device)

                outputs = self.model(images)
                
                loss = self.criterion(outputs, masks)
                val_loss += loss.item()

                preds_prob = torch.softmax(outputs, dim=1)
                preds = torch.argmax(preds_prob, dim=1)
                iou_d, iou_c, dice_d, dice_c = calculate_metrics(preds, masks)

                metrics['precision'].update(preds, masks)
                metrics['recall'].update(preds, masks)
                metrics["iou_d"] += iou_d.item()
                metrics["iou_c"] += iou_c.item()
                metrics["dice_d"] += dice_d.item()
                metrics["dice_c"] += dice_c.item()

                #--- NEW: Targeted Visualization Logic ---
                if self.visualization_samples is not None:
                    print(f"Checking visualization for batch with names: {names}")
                    for idx, name in enumerate(names):
                        if True:
                        #if name in self.visualization_samples or self.visualization_samples == 'ALL':
                            print(f"Adding sample '{name}' to visualization for epoch {epoch}")
                            # Store the data as a dictionary for easy plotting later
                            samples_to_visualize.append({
                                'name': name,
                                'image': images[idx].cpu(),
                                'mask': masks[idx].cpu(),
                                'pred': preds[idx].cpu()
                            })
                else:
                    print("No visualization samples specified, skipping visualization for this epoch.") 

        # Scheduler Step
        '''if self.scheduler:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(avg_loss)
            else:
                self.scheduler.step()'''

        # Compute and Log
        results = {name: metric.compute() for name, metric in val_metrics.items()}
        avg_loss = val_loss / len(self.val_loader)

        self.report_generator.log_val_step(avg_loss, metrics, self.optimizer.param_groups[0]['lr'], epoch, samples=samples_to_visualize)


        n = len(self.val_loader)
        return avg_loss, (metrics["dice_d"]/n + metrics["dice_c"]/n)/2,metrics["dice_d"]/n, metrics["dice_c"]/n, metrics["iou_d"]/n, metrics["iou_c"]/n, results['recall'].mean().item(), results['precision'].mean().item()


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
        
