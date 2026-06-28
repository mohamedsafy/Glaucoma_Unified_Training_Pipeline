import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def generate_roi_dataset(input_dir, output_dir, crop_size=512, max_workers=os.cpu_count() * 4):

    def find_bbox(mask_np):
        # Universal bbox: detect objects that aren't the majority background
        bg_color = mask_np[0, 0]
        rows, cols = np.where(mask_np < 250) if bg_color > 127 else np.where(mask_np > 10)
        if len(rows) == 0: return None
        return int(rows.min()), int(cols.min()), int(rows.max()+1), int(cols.max()+1)

    def process_single_image(args):
        mask_file, mask_dir, img_dir, out_img_dir, out_mask_dir = args
        try:
            mask_path = os.path.join(mask_dir, mask_file)
            mask_pil = Image.open(mask_path).convert("L")
            mask = np.array(mask_pil)

            bbox = find_bbox(mask)
            if bbox is None: return False

            # ROI Math
            t, l, b, r = bbox
            cy, cx = (t + b) // 2, (l + r) // 2
            half = crop_size // 2
            start_y, start_x = cy - half, cx - half
            end_y, end_x = start_y + crop_size, start_x + crop_size

            # Image Matching
            base_name = os.path.splitext(mask_file)[0]
            if base_name.endswith("_mask"): base_name = base_name[:-5]

            found_img_path = None
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif']:
                p = os.path.join(img_dir, base_name + ext)
                if os.path.exists(p):
                    found_img_path = p
                    break

            if not found_img_path: return False
            image = np.array(Image.open(found_img_path).convert("RGB"))

            # Pad & Crop Logic
            h, w = image.shape[:2]
            pad_t, pad_l = max(0, -start_y), max(0, -start_x)
            pad_b, pad_r = max(0, end_y - h), max(0, end_x - w)

            if any([pad_t, pad_l, pad_b, pad_r]):
                image = np.pad(image, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode='constant')
                mask = np.pad(mask, ((pad_t, pad_b), (pad_l, pad_r)), mode='constant', constant_values=255 if mask[0,0]>127 else 0)
                start_y, start_x = start_y + pad_t, start_x + pad_l
                end_y, end_x = end_y + pad_t, end_x + pad_l

            img_crop = image[start_y:start_y+crop_size, start_x:start_x+crop_size]
            mask_crop = mask[start_y:start_y+crop_size, start_x:start_x+crop_size]

            Image.fromarray(img_crop).save(os.path.join(out_img_dir, f"{base_name}.jpg"))
            Image.fromarray(mask_crop).save(os.path.join(out_mask_dir, f"{base_name}.bmp"))
            return True
        except Exception:
            return False

    def get_subfolder(base_path, options):
        for opt in options:
            if os.path.exists(os.path.join(base_path, opt)): return opt
            if os.path.exists(os.path.join(base_path, opt.capitalize())): return opt.capitalize()
        return None

    # --- MAIN LOGIC ---
    # Detect if it's split-based (REFUGE) or flat-based (ORIGA)
    subfolders = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    has_splits = any(s in subfolders for s in ['train', 'val', 'test'])

    splits_to_process = subfolders if has_splits else ['.'] # Use root if no splits found
    #print(f"Found the following splits {splits_to_process}")
    #splits_to_process = list(set(subfolders) & set(['train', 'val', 'test'])) #support for drishti

    #print(f"Found the following splits {splits_to_process}")
    #print(f"has splits: {has_splits}")
    #print(f"subfolders: {subfolders}")
    for split in splits_to_process:
        #print("help")
        curr_input_path = os.path.join(input_dir, split)

        # Flexibly find images and masks folders
        m_folder = get_subfolder(curr_input_path, ['mask', 'masks', 'gts', 'GroundTruth'])
        i_folder = get_subfolder(curr_input_path, ['images', 'Images', 'img', 'IMG', 'image'])

        #print(f"m_folder: {m_folder}")
        #print(f"i_folder: {i_folder}")

        if not m_folder or not i_folder:
            continue # Skip folder if image/mask pair directory not found

        out_split_path = os.path.join(output_dir, split if split != '.' else '')
        out_img_dir = os.path.join(out_split_path, 'images')
        out_mask_dir = os.path.join(out_split_path, 'masks')

        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_mask_dir, exist_ok=True)

        mask_dir = os.path.join(curr_input_path, m_folder)
        img_dir = os.path.join(curr_input_path, i_folder)

        mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.bmp', '.jpg', '.tif'))]
        tasks = [(f, mask_dir, img_dir, out_img_dir, out_mask_dir) for f in mask_files]

        print(f"🚀 Processing {split}: Found {len(mask_files)} masks...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(executor.map(process_single_image, tasks), total=len(tasks)))

    print(f"\n✅ ROI Generation Complete! Saved to {output_dir}")
    return output_dir
