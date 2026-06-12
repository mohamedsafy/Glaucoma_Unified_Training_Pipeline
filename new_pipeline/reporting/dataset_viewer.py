import matplotlib.pyplot as plt
from new_pipeline.utils.mask_utils import get_colored_mask
import numpy as np

def view_dataset(dataset, num_samples: int = 5) -> None:
    """Utility to visualize random samples from a dataset."""
    print(f"🔍 Visualizing {num_samples} random samples from the dataset...")
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    indices = np.random.choice(len(dataset), size=num_samples, replace=False)

    fig, axes = plt.subplots(num_samples, 2, figsize=(5, 5))

    for idx in range(num_samples):
        print(f"Visualizing sample {idx + 1}/{num_samples}")
        image, mask, name = dataset[idx]
        image_np = image.permute(1, 2, 0).numpy()
        image_np = np.clip((image_np * std + mean), 0, 1)
        mask_np = get_colored_mask(mask.numpy())

        axes[idx, 0].imshow(image_np)
        axes[idx, 0].set_title("Image - " + name)
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(mask_np, cmap="gray")
        axes[idx, 1].set_title("Mask - " + name)
        axes[idx, 1].axis("off")

    plt.show()