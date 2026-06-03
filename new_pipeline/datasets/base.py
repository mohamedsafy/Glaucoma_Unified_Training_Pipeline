import os
import shutil

from new_pipeline.config.run_config import DatasetConfig
from new_pipeline.utils.mask_utils import read_mask
from new_pipeline.datasets.generate_roi_dataset import generate_roi_dataset
from torch.utils.data import Dataset
from torch import Tensor, from_numpy
from PIL import Image
import numpy as np

class DatasetSourcer:
    def source(self, config: DatasetConfig) -> str:
        if config.roi:
            local_root = config.local_root + "_ROI"
            drive_root = config.drive_root + "_ROI"
            try:
                root = self._source(local_root, drive_root)
                return root
            except Exception as e:
                print(f"Failed to source dataset with ROI: {e}")
                print(f"Generating ROI dataset from original dataset at {config.local_root}...")
                root = self._source(config.local_root, config.drive_root)  # First source the original dataset
                root = generate_roi_dataset(root, local_root)  # Then generate the ROI dataset from it
                return root
            
        return self._source(config.local_root, config.drive_root)

    def _source(self, local_root: str, drive_root: str) -> str:
        if  os.path.exists(local_root):
            print(f"Dataset already exists at {local_root}. Skipping download.")
            return local_root
        
        print(f"Dataset not found at {local_root}. Attempting to pull from google drive...")
        if os.path.exists(drive_root):
            print(f"Found dataset at {drive_root}. Copying to {local_root}...")
            return shutil.copytree(drive_root, local_root)

        print(f"Dataset not found at {drive_root} either. Attempting to download from source...")
        raise NotImplementedError("Dataset download logic not implemented yet.")

class DatasetSpliter:
    pass

class StandardDataset(Dataset):

    def __init__(self, root: str,
                    ids: list[str] = None,
                    transforms=None, dataset_multiplier=1, n=None):
            self.images_dir = os.path.join(root, 'images')
            self.masks_dir = os.path.join(root, 'masks')

            if ids is None:
                ids = [f for f in os.listdir(self.images_dir) if f.endswith(('.png', '.jpg', '.bmp'))]

            self.ids = ids * dataset_multiplier
            self.transforms = transforms
            self.ids.sort()
            if n is not None:
                self.ids = self.ids[:n]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img_name = self.ids[i]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, os.path.splitext(img_name)[0] + '.bmp')

        # Open Images
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = read_mask(mask_path)

        # --- AUGMENT ---
        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        # Ensure mask_classes is a LongTensor before returning
        if isinstance(mask, Tensor):
            mask = mask.long()
        else: # if it's still a numpy array after augmentation (e.g., if ToTensorV2 was not applied for some reason)
            mask = from_numpy(mask).long()

        return image, mask, img_name




