from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Union
from torch.utils.tensorboard import SummaryWriter
from new_pipeline.config.run_config import RunConfig
from new_pipeline.utils.mask_utils import visualize_result
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
from PIL import Image
import os
import json
import traceback
import numpy as np
import io
from new_pipeline.utils.mask_utils import get_colored_mask

def graphs_factory():
    return [
                                                                        {
                                                                        'tags': ['val/dice_cup', 'val/dice_disc'],
                                                                        'title': 'Validation Dice Scores',
                                                                        'filename':'dice_comparison',
                                                                        'ylabel': 'Score'
                                                                        },
                                                                        {
                                                                        'tags': ['val/iou_cup', 'val/iou_disc'],
                                                                        'title': 'Validation IoU Scores',
                                                                        'filename':'iou_comparison',
                                                                        'ylabel': 'Score'
                                                                        },
                                                                        {
                                                                        'tags': ['val/lr'],
                                                                        'title': 'Learning Rate Schedule',
                                                                        'filename':'lr_schedule',
                                                                        'ylabel': 'LR'
                                                                        },
                                                                        {
                                                                        'tags': ['train/loss', 'val/loss'],
                                                                        'title': 'Training & Validation Loss',
                                                                        'filename':'loss_curve',
                                                                        'ylabel': 'Loss'
                                                                        },
                                                                        {
                                                                        'tags': ['val/recall'],
                                                                        'title': 'Mean Recall',
                                                                        'filename':'recall_curve',
                                                                        'ylabel': 'Score'
                                                                        },
                                                                        {
                                                                        'tags': ['val/precision'],
                                                                        'title': 'Mean Precision',
                                                                        'filename':'precision_curve',
                                                                        'ylabel': 'Score'
                                                                        }
                                                                    ]

def default_best_epoch_factory():
    return {'epoch':1, 'val/dice_cup':0, 'val/dice_disc': 0, 'val/iou_cup': 0, 'val/iou_disc': 0, 'val/precision': 0, 'val/recall': 0,
                                'train/dice_cup':0, 'train/dice_disc': 0, 'train/iou_cup': 0, 'train/iou_disc': 0, 'val/lr':0,
                                'val/loss': 0, 'train/loss': 0}

@dataclass
class ReportGenerator:
    val_dataset: Any
    visualization_samples: Optional[Union[list[str], str]]
    visualization_epochs: list[int] = field(default_factory=list)
    config: RunConfig = None
    exp_dir: str = None
    exp_title: str = None
    writer: SummaryWriter = None
    best_epoch: dict[str, Any] = field(default_factory=default_best_epoch_factory)
    graphs: list[dict[str, Any]] = field(default_factory=graphs_factory)
    ea: event_accumulator = field(default_factory= lambda : None)

    def log_train_step(self, loss: Any, metrics: dict[str, Any], epoch: int) -> None:
        print(f"Report Generator:Logging training metrics for epoch {epoch}...")
        self.writer.add_scalar("train/loss", loss, epoch)
        self.writer.add_scalar("train/dice_cup", metrics["dice_c"], epoch)
        self.writer.add_scalar("train/dice_disc", metrics["dice_d"], epoch)
        self.writer.add_scalar("train/iou_cup", metrics["iou_c"], epoch)
        self.writer.add_scalar("train/iou_disc", metrics["iou_d"], epoch)

    def log_val_step(
        self,
        loss: Any,
        metrics: dict[str, Any],
        lr: float,
        epoch: int,
        samples: list[Any],
    ) -> None:
        #print(f"Report Generator:Logging validation metrics for epoch {epoch}...")
        self.writer.add_scalar("val/loss", loss, epoch)
        self.writer.add_scalar("val/dice_cup", metrics["dice_c"], epoch)
        self.writer.add_scalar("val/dice_disc", metrics["dice_d"], epoch)
        self.writer.add_scalar("val/iou_cup", metrics["iou_c"], epoch)
        self.writer.add_scalar("val/iou_disc", metrics["iou_d"], epoch)
        self.writer.add_scalar("val/recall", metrics["recall"], epoch)
        self.writer.add_scalar("val/precision", metrics["precision"], epoch)
        self.writer.add_scalar("val/lr", lr, epoch)

        # Visualization logging (if any samples were collected)
        if samples:
            for sample in samples:
                #print(f"Logging visualization for sample: {sample['name']} at epoch {epoch}")
                self.writer.add_image(f'val/samples/{sample["name"]}/visualization', visualize_result(sample), epoch)
        #else:
            #print("No visualization samples to log for this epoch.")

    def log_sample_progress(self, name: str, epoch: int, dice_c: float, dice_d: float, iou_c: float, iou_d: float) -> None:
        self.writer.add_scalar(f'val/samples/{name}/dice_cup', dice_c, epoch)
        self.writer.add_scalar(f'val/samples/{name}/dice_disc', dice_d, epoch)
        self.writer.add_scalar(f'val/samples/{name}/iou_cup', iou_c, epoch)
        self.writer.add_scalar(f'val/samples/{name}/iou_disc', iou_d, epoch)

    def log_best_epoch(self, epoch):
        self.best_epoch = epoch

    def generate(self, log_dir: str = None) -> None:
        size_guidance = {
        event_accumulator.IMAGES: 0, # 0 = Load every single image logged
        event_accumulator.SCALARS: 0,
        }

        if log_dir is None:
            log_dir = os.path.join(self.exp_dir,
                        sorted([f for f in os.listdir(self.exp_dir) if f.startswith('events.')])[-1])

        print(f"Generating graphs from log file:{log_dir}")
        # 1. Load the event file
        self.ea = event_accumulator.EventAccumulator(log_dir, size_guidance=size_guidance) # log_dir is where TB files are
        self.ea.Reload()
        scalars = self.ea.scalars.Keys()
        #scal = self.ea.Scalars('train/loss')
        graphs = self.generate_graphs()
        for graph in graphs:
            graph['graph'].savefig(f"{self.exp_dir}/{graph['filename']}.png", dpi=300, bbox_inches='tight')


        print("Starting to generate progression matrix")
        self.generate_progression_matrix()
        self.generate_heatmaps()


    def generate_graphs(self):
        graphs= []

        try:
            for graph in self.graphs:
                fig = self._plot_joint(graph['tags'], graph['title'], graph['ylabel'])
                graphs.append({'filename':graph['filename'], 'graph':fig})

        except Exception as e:
            print(f"⚠️ Error generating graphs: {e}")

        return graphs

    def _get_scalar(self, tag):
      return [x.value for x in self.ea.Scalars(tag)], [x.step for x in self.ea.Scalars(tag)]

    def _plot_joint(self, tags, title, ylabel='Score'):
        fig = plt.figure(figsize=(10, 6))
        # 1. Plot all lines first
        for tag in tags:
            try:
                values, steps = self._get_scalar(tag)
                label = tag.split('/')[-1].replace('_', ' ').capitalize()
                plt.plot(steps, values, label=label, linewidth=2)
                print(len(steps))
            except Exception as ex:
                print(f"Error while finding tag {tag}, : {ex}") # Skip tags not found in TensorBoard

        for tag in tags:
            plt.scatter(self.best_epoch['epoch'], self.best_epoch[tag], color='red', s=40, zorder=5)

                # Professional Annotation: "Ep X: Value"
            plt.annotate(
                f"Ep {self.best_epoch['epoch']}: {self.best_epoch[tag]:.4f}",
                xy=(self.best_epoch['epoch'], self.best_epoch[tag]),
                xytext=(10, -10), # Offset text slightly
                textcoords='offset points',
                fontsize=9,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3), # Light highlight
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2') # Small curved arrow
            )

        plt.title(self.exp_title+"\n"+title)
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        if ylabel != 'LR':
            plt.ylim(bottom=0)
        else:
            plt.autoscale(enable=True, axis='both', tight=None)
        plt.legend(frameon=True, loc='lower right')
        plt.grid(True, linestyle=':', alpha=0.6)
        fig.canvas.draw()
        ret = Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.buffer_rgba())
        plt.close()
        return fig
    
    def generate_progression_matrix(self):
        """
        Args:
            ea: The EventAccumulator.
            dataset: The validation dataset object (to pull Input/GT).
            best_epoch: int.
            image_names: list of strings (filenames).
            target_epochs: list of ints (training steps).
            output_path: str.
        """
        output_path = os.path.join(self.exp_dir, 'progression_matrix.png')
        image_names = self.visualization_samples
        target_epochs = self.visualization_epochs
        if self.best_epoch['epoch'] not in target_epochs:
            target_epochs = [self.best_epoch['epoch']] + target_epochs

        dataset = self.val_dataset
        # Rows: Input + GT + Target Epochs
        reference_rows = ['Input', 'GT']
        all_row_types = reference_rows + target_epochs
        num_rows = len(all_row_types)
        num_cols = len(image_names)

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))

        # Standardize axes to 2D array
        if num_rows == 1: axes = [axes]
        if num_cols == 1: axes = [[a] for a in axes]

        # Pre-denormalization constants for the 'Input' row
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        for c, name in enumerate(image_names):
            # --- 1. Fetch Static Data from Dataset ---
            # Assuming dataset.get_by_name or similar logic exists
            # If not, we find the index of the filename
            try:
                idx = dataset.ids.index(name)
                img_tensor, mask_tensor, _ = dataset[idx]

                # Process Input Image
                img_np = img_tensor.permute(1, 2, 0).numpy()
                img_np = np.clip((img_np * std + mean), 0, 1)

                # Process GT Mask (using your existing coloring function)
                # Assuming get_colored_mask is available globally
                gt_colored = get_colored_mask(mask_tensor.squeeze().numpy()) / 255.0

                axes[0][c].imshow(img_np)
                axes[1][c].imshow(gt_colored)
            except Exception as e:
                axes[0][c].text(0.5, 0.5, f'Error\n{name}', ha='center')
                axes[1][c].text(0.5, 0.5, 'GT Missing', ha='center')
                print(f"⚠️ Error loading image {name}: {e}")
            print("Continuing after failure")

            # --- 2. Fetch Epoch Predictions from TensorBoard ---
            for r_idx, epoch in enumerate(target_epochs):
                # Row index is r_idx + 2 (because of Input and GT rows)
                r = r_idx + 2
                tag = f'val/samples/{name}/visualization'

                try:
                    image_events = self.ea.Images(tag)
                    img_event = next((e for e in image_events if e.step == epoch), None)

                    if img_event:
                        pred_img = Image.open(io.BytesIO(img_event.encoded_image_string))
                        axes[r][c].imshow(pred_img)
                    else:
                        axes[r][c].text(0.5, 0.5, f'Ep {epoch}\nMissing', ha='center')
                except Exception as e:
                    print(f"⚠️ Error fetching prediction for {name} at epoch {epoch}: {e}")
                    axes[r][c].text(0.5, 0.5, 'Not Found', ha='center')

        # --- 3. Formatting & Labels ---
        for r, row_type in enumerate(all_row_types):
            # Row Labels
            label = str(row_type)
            if row_type == self.best_epoch['epoch']: label = f"Epoch {row_type}\n⭐ BEST"
            elif isinstance(row_type, int): label = f"Epoch {row_type}"

            axes[r][0].set_ylabel(label, fontsize=14, fontweight='bold', labelpad=15)

            for c, name in enumerate(image_names):
                axes[r][c].set_xticks([])
                axes[r][c].set_yticks([])
                if r == 0:
                    axes[r][c].set_title(f"Sample:\n{name}", fontsize=10)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Full Matrix Report saved to: {output_path}")

    def generate_heatmaps(self, log_dir: str = None, image_names: list[str] = None, target_epochs: list[int] = None, exp_dir: str = None) -> None:
        '''
        Generate heatmaps for each validation sample metrics recorded in each validate loop.
        A heatmap is generated for each metric - X axis is epoch, Y axis is sample, color is metric value.
        The heatmaps are saved as images in the experiment directory.

        Optimized implementation: scan scalar keys once and build compact lookups to minimize EA calls.
        '''
        if log_dir is None:
            log_dir = os.path.join(self.exp_dir, sorted([f for f in os.listdir(self.exp_dir) if f.startswith('events.')])[-1])
        if image_names is None:
            image_names = self.val_dataset.ids
        if target_epochs is None:
            target_epochs = list(range(1, self.config.epochs + 1))
        if exp_dir is None:
            exp_dir = self.exp_dir

        image_list = list(image_names)
        epoch_list = list(target_epochs)
        if not image_list or not epoch_list:
            return

        self.ea = event_accumulator.EventAccumulator(log_dir, size_guidance={event_accumulator.SCALARS: 0})
        self.ea.Reload()

        metrics = ['dice_cup', 'dice_disc', 'iou_cup', 'iou_disc']
        metric_set = set(metrics)
        image_set = set(image_list)
        epoch_set = set(epoch_list)

        # Build metric->sample->{epoch:value} mapping by scanning keys once
        print(f"Building data mapping for heatmaps from log: {log_dir}")
        data: dict[str, dict[str, dict[int, float]]] = {m: {} for m in metrics}
        for key in self.ea.scalars.Keys():
            try:
                prefix_name, sample_name, metric = key.rsplit('/', 2)
            except Exception:
                continue
            if prefix_name != 'val/samples' or metric not in metric_set or sample_name not in image_set:
                continue
            try:
                events = self.ea.Scalars(key)
            except Exception:
                continue
            samp = data[metric].setdefault(sample_name, {})
            for e in events:
                if e.step in epoch_set:
                    samp[e.step] = e.value

        print(f"Data mapping completed. Generating heatmaps for metrics: {metrics}")
        os.makedirs(exp_dir, exist_ok=True)
        for metric in metrics:
            heat = np.full((len(image_list), len(epoch_list)), np.nan, dtype=float)
            mdata = data.get(metric, {})
            for i, name in enumerate(image_list):
                vals = mdata.get(name, {})
                for j, ep in enumerate(epoch_list):
                    heat[i, j] = vals.get(ep, np.nan)

            print(f"Heatmap for {metric} generated with shape {heat.shape}. Saving to {exp_dir}")
            fig, ax = plt.subplots(figsize=(max(6, len(epoch_list) * 0.6), max(3, len(image_list) * 0.25)))

            print(f"Plotting heatmap for {metric}...")
            im = ax.imshow(heat, aspect='auto', cmap='viridis')

            print(f"Configuring axes for {metric} heatmap...")
            ax.set_xticks(list(range(len(epoch_list))))
            ax.set_xticklabels(epoch_list, rotation=45, fontsize=6)
            ax.set_yticks(list(range(len(image_list))))
            ax.set_yticklabels(image_list, fontsize=6)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Sample')
            ax.set_title(f'{metric.replace("_", " ").capitalize()} Heatmap')
            fig.colorbar(im, ax=ax, label=metric.replace('_', ' ').capitalize())
            fig.tight_layout()

            print(f"Saving heatmap for {metric}...")
            fig.savefig(os.path.join(exp_dir, f'{metric}_heatmap.png'), dpi=100, bbox_inches='tight')
            plt.close(fig)

            print(f"✅ Heatmap for {metric} saved to: {os.path.join(exp_dir, f'{metric}_heatmap.png')}")


    def save_metadata(self, ex: Exception | None = None) -> None:
        '''
        Saves metadata about the run, caches all information about the run from config
        '''
        if self.exp_dir is None:
            raise ValueError("Cannot save metadata without an experiment directory.")

        os.makedirs(self.exp_dir, exist_ok=True)
        if self.writer is not None:
            self.writer.flush()

        metadata = {
            "experiment": {
                "title": self.exp_title,
                "directory": self.exp_dir,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "status": "failed" if ex else "completed",
            },
            "config": self._to_jsonable(self.config),
            "best_epoch_stats": self._to_jsonable(self.best_epoch),
            "visualization_samples": self._to_jsonable(self.visualization_samples),
            "exception": self._exception_metadata(ex),
        }

        save_path = os.path.join(self.exp_dir, "experiment_metadata.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        self._save_text_summary(metadata)

    def _exception_metadata(self, ex: Exception | None) -> dict[str, Any] | None:
        if ex is None:
            return None

        return {
            "type": type(ex).__name__,
            "message": str(ex),
            "traceback": traceback.format_exception(type(ex), ex, ex.__traceback__),
        }

    def _save_text_summary(self, metadata: dict[str, Any]) -> None:
        summary_path = os.path.join(self.exp_dir, "summary.txt")
        best_epoch = metadata["best_epoch_stats"]
        config = metadata["config"]
        experiment = metadata["experiment"]

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"Experiment Summary: {experiment['title']}\n")
            f.write("=" * 40 + "\n")
            f.write(f"Status: {experiment['status']}\n")
            f.write(f"Saved At: {experiment['saved_at']}\n")
            f.write(f"Experiment Directory: {experiment['directory']}\n\n")

            f.write(f"Best Epoch: {best_epoch.get('epoch')}\n")
            for key, value in best_epoch.items():
                if key == "epoch":
                    continue
                if isinstance(value, (int, float)):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")

            f.write("\nConfig Used:\n")
            for key, value in config.items():
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, indent=2)
                f.write(f"{key}: {value}\n")

            if metadata["exception"]:
                f.write("\nException:\n")
                f.write(f"{metadata['exception']['type']}: {metadata['exception']['message']}\n")

    def _to_jsonable(self, value: Any) -> Any:
        if dataclass_isinstance(value):
            return {
                key: self._to_jsonable(val)
                for key, val in vars(value).items()
            }
        if isinstance(value, dict):
            return {
                str(key): self._to_jsonable(val)
                for key, val in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [self._to_jsonable(item) for item in value]
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def clean(self):
        if hasattr(self, 'writer'):
            self.writer.close()

def dataclass_isinstance(value: Any) -> bool:
    return hasattr(value, "__dataclass_fields__") and not isinstance(value, type)
