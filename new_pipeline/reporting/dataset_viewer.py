import matplotlib.pyplot as plt
from new_pipeline.utils.mask_utils import get_colored_mask
import numpy as np


def view_dataset(
    dataset,
    num_samples: int = 5,
    samples_indices: list = None,
    cell_size: int = 4,
    max_rows_per_figure: int = 4,
) -> None:
    """Utility to visualize samples from a dataset."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    if samples_indices is None:
        indices = list(range(num_samples))
    else:
        indices = samples_indices

    print(f"Visualizing {len(indices)} samples from the dataset...")

    for start in range(0, len(indices), max_rows_per_figure):
        batch_indices = indices[start:start + max_rows_per_figure]
        num_rows = len(batch_indices)

        fig, axes = plt.subplots(
            num_rows,
            2,
            figsize=(cell_size * 2, cell_size * num_rows),
            squeeze=False,
        )

        for row_idx, dataset_idx in enumerate(batch_indices):
            print(f"Visualizing sample {dataset_idx + 1}")
            image, mask, name = dataset[dataset_idx]

            image_np = image.permute(1, 2, 0).numpy()
            image_np = np.clip((image_np * std + mean), 0, 1)
            mask_np = get_colored_mask(mask.numpy())

            axes[row_idx, 0].imshow(image_np)
            axes[row_idx, 0].set_title("Image - " + name)
            axes[row_idx, 0].axis("off")

            axes[row_idx, 1].imshow(mask_np)
            axes[row_idx, 1].set_title("Mask - " + name)
            axes[row_idx, 1].axis("off")

        plt.tight_layout()
        plt.show()
