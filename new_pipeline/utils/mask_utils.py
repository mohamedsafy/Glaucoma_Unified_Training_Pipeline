from PIL import Image
import numpy as np

def get_colored_mask(mask_labels):
    # mask_labels is a HxW numpy array with integer class labels (0, 1, 2)
    color_map = np.array([
        [255, 255, 255],     # Class 0 (background) -> white
        [0, 0, 0],    # Class 1 (cup) -> Black
        [128,128,128]    # Class 2 (disc) -> Gray
    ], dtype=np.uint8)
    colored_mask = color_map[mask_labels]
    return colored_mask

def read_mask(mask_path):
    """Reads a mask image and converts it to a numpy array of class indices."""
    mask_image = Image.open(mask_path).convert("L")
    mask_array = np.array(mask_image)
    mask_classes = np.zeros_like(mask_array, dtype=np.int64)
    mask_classes[mask_array == 0] = 1
    mask_classes[mask_array == 128] = 2

    return mask_classes

def visualize_result(sample):

    if not sample:
        return

    # Standard ImageNet denormalization constants
    mean = np.array([0.485, 0.456, 0.406])[:, None, None]
    std = np.array([0.229, 0.224, 0.225])[:, None, None]

    try:
        # 2. Prepare Masks (Colored)
        # sample['pred'] is already argmaxed from the val loop
        colored_pred = get_colored_mask(sample['pred'].numpy()) / 255.0

        # 5. Convert to TensorBoard format (CHW)
        montage_hwc = (colored_pred * 255).astype(np.uint8)
        montage_chw = np.transpose(montage_hwc, (2, 0, 1))

        return montage_chw

    except Exception as e:
        print(f"⚠️ Visualization error: {e}")
