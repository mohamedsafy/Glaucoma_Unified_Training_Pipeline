import os
import shutil

import torch

from new_pipeline.config.run_config import DatasetConfig
from new_pipeline.utils.mask_utils import read_mask
from new_pipeline.datasets.generate_roi_dataset import generate_roi_dataset
from torch.utils.data import Dataset
from torch import Tensor, from_numpy
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor

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
                 transforms=None, dataset_multiplier=1, n=None, in_memory=True):
        self.images_dir = os.path.join(root, 'images')
        self.masks_dir = os.path.join(root, 'masks')
        self.in_memory = in_memory
        self.transforms = transforms

        if ids is None:
            ids = [f for f in os.listdir(self.images_dir) if f.endswith(('.png', '.jpg', '.bmp'))]

        ids.sort()
        if n is not None:
            ids = ids[:n]

        # Save the finalized, multiplied list of names for index retrieval
        self.ids = ids * dataset_multiplier

        if self.in_memory:
            # OPTIMIZATION: Only load UNIQUE images from disk to save hours of duplicate work
            unique_ids = list(set(ids))
            unique_ids.sort()
            
            unique_storage = {}
            print(f"🚀 Parallel preloading {len(unique_ids)} unique samples into memory...")

            # Helper function for worker threads
            def load_single_item(img_name):
                try:
                    img_path = os.path.join(self.images_dir, img_name)
                    mask_path = os.path.join(self.masks_dir, os.path.splitext(img_name)[0] + '.bmp')
                    image, mask = self.get_image(img_path, mask_path)
                except Exception as e:
                    print(f"Error loading {img_name}: {e}")
                    image, mask = None, None
                return img_name, image, mask

            # Use maximum available CPU workers to read data in parallel
            max_workers = min(32, os.cpu_count() + 4)
            print(f"Using {max_workers} worker threads for parallel loading.")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = executor.map(load_single_item, unique_ids)
                for img_name, image, mask in results:
                    unique_storage[img_name] = (image, mask)

            # Map the unique preloaded items to your final multiplied list structure
            self.images = torch.stack([unique_storage[name][0] for name in self.ids])
            self.masks = torch.stack([unique_storage[name][1] for name in self.ids])
            print("✅ Finished preloading dataset into memory.")

            

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        if self.in_memory:
            return self.images[i], self.masks[i], self.ids[i]
        img_name = self.ids[i]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, os.path.splitext(img_name)[0] + '.bmp')

        image, mask = self.get_image(img_path, mask_path)
        
        return image, mask, img_name

        

    def get_image(self, img_path: str, mask_path: str) -> Tensor:
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

        return image, mask




