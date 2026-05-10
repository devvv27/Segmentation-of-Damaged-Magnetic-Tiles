import os
import random
from pathlib import Path
from PIL import Image
from collections import defaultdict

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
BASE_PATH   = Path(r"C:\\DewangData\\ProjectsTHISSEM\\CV\\Project")
OUTPUT_PATH = Path(r"C:\\DewangData\\ProjectsTHISSEM\\CV\\MagneticTilesDataset_Augmented")

FOLDERS = ['MT_Blowhole', 'MT_Break', 'MT_Crack', 'MT_Fray', 'MT_Free', 'MT_Uneven']

TARGET_SIZE = (256, 256)
SKIP_AUGMENTATION_FOLDERS = ['MT_Free']

TRAIN_RATIO = 0.6
VAL_RATIO   = 0.2
# TEST_RATIO  = 0.2 (remainder)

RANDOM_SEED = 42

# ─────────────────────────────────────────────────────────────
# AUGMENTATION DEFINITIONS (identical to original)
# ─────────────────────────────────────────────────────────────

def apply_hflip_rot10(img):
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    fillcolor = 0 if img.mode == 'L' else (0, 0, 0)
    return img.rotate(10, expand=False, fillcolor=fillcolor)

def apply_hflip_rot_neg10(img):
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    fillcolor = 0 if img.mode == 'L' else (0, 0, 0)
    return img.rotate(-10, expand=False, fillcolor=fillcolor)

def apply_vflip_rot10(img):
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    fillcolor = 0 if img.mode == 'L' else (0, 0, 0)
    return img.rotate(10, expand=False, fillcolor=fillcolor)

def apply_vflip_rot_neg10(img):
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    fillcolor = 0 if img.mode == 'L' else (0, 0, 0)
    return img.rotate(-10, expand=False, fillcolor=fillcolor)

# 4 augmentations for Blowhole, Break, Free(skipped), Uneven
AUGMENTATIONS = {
    'hflip':   lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
    'vflip':   lambda img: img.transpose(Image.FLIP_TOP_BOTTOM),
    'rot10':   lambda img: img.rotate(10,  expand=False, fillcolor=0 if img.mode == 'L' else (0, 0, 0)),
    'rot15':   lambda img: img.rotate(15,  expand=False, fillcolor=0 if img.mode == 'L' else (0, 0, 0)),
}

# 10 augmentations for MT_Fray and MT_Crack
AUGMENTATIONS_10 = {
    'hflip':          lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
    'vflip':          lambda img: img.transpose(Image.FLIP_TOP_BOTTOM),
    'rot10':          lambda img: img.rotate(10,  expand=False, fillcolor=0 if img.mode == 'L' else (0, 0, 0)),
    'rot15':          lambda img: img.rotate(15,  expand=False, fillcolor=0 if img.mode == 'L' else (0, 0, 0)),
    'rot_neg10':      lambda img: img.rotate(-10, expand=False, fillcolor=0 if img.mode == 'L' else (0, 0, 0)),
    'rot_neg15':      lambda img: img.rotate(-15, expand=False, fillcolor=0 if img.mode == 'L' else (0, 0, 0)),
    'hflip_rot10':    apply_hflip_rot10,
    'hflip_rot_neg10':apply_hflip_rot_neg10,
    'vflip_rot10':    apply_vflip_rot10,
    'vflip_rot_neg10':apply_vflip_rot_neg10,
}


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def detect_image_gt_pairs(img_dir):
    """Return dict {base_name: {'image': Path, 'gt': Path}} for all jpg/png pairs."""
    pairs = {}
    for jpg_file in sorted(Path(img_dir).glob('*.jpg')):
        base_name = jpg_file.stem
        png_file  = Path(img_dir) / f"{base_name}.png"
        if png_file.exists():
            pairs[base_name] = {'image': jpg_file, 'gt': png_file}
    return pairs


def resize_pair(img_path, gt_path, target_size):
    image   = Image.open(img_path).convert('RGB')
    gt_mask = Image.open(gt_path).convert('L')
    return (
        image.resize(target_size, Image.LANCZOS),
        gt_mask.resize(target_size, Image.NEAREST)
    )


def apply_augmentation(image, gt_mask, aug_func):
    return aug_func(image), aug_func(gt_mask)


def save_pair(img, gt, imgs_dir, gts_dir, base_name, suffix=''):
    """Save image as jpg and GT as png into their respective folders."""
    name = f"{base_name}{suffix}"
    img.save(imgs_dir / f"{name}.jpg", quality=95)
    gt.save(gts_dir  / f"{name}.png")


def make_split_dirs(output_path, split, folder):
    """Create and return (imgs_dir, gts_dir) for a given split and class folder."""
    imgs_dir = output_path / split / folder / 'Imgs'
    gts_dir  = output_path / split / folder / 'GTs'
    imgs_dir.mkdir(parents=True, exist_ok=True)
    gts_dir.mkdir(parents=True, exist_ok=True)
    return imgs_dir, gts_dir


def split_pairs(pairs_list, train_ratio, val_ratio, seed):
    """Shuffle and split a list of base_names into train/val/test."""
    random.seed(seed)
    names = list(pairs_list)
    random.shuffle(names)
    n          = len(names)
    train_end  = int(n * train_ratio)
    val_end    = int(n * (train_ratio + val_ratio))
    return names[:train_end], names[train_end:val_end], names[val_end:]


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def process_dataset():
    print("=" * 80)
    print("DATASET AUGMENTATION SUMMARY")
    print("=" * 80)
    print(f"{'Folder':<20} {'Original':<12} {'Train':<10} {'Val':<10} {'Test':<10} {'Note'}")
    print("-" * 80)

    grand_orig = 0
    grand_train = grand_val = grand_test = 0

    for folder in FOLDERS:
        img_dir = BASE_PATH / folder / 'Imgs'

        if not img_dir.exists():
            print(f"{folder:<20} folder not found, skipping.")
            continue

        pairs        = detect_image_gt_pairs(img_dir)
        original_count = len(pairs)

        # Pick augmentation strategy
        if folder == 'MT_Fray':
            aug_dict   = AUGMENTATIONS_10
            num_repeats = 2
            note       = "Original + 10 augs × 2"
        elif folder == 'MT_Crack':
            aug_dict   = AUGMENTATIONS_10
            num_repeats = 1
            note       = "Original + 10 augs"
        elif folder in SKIP_AUGMENTATION_FOLDERS:
            aug_dict   = {}
            num_repeats = 0
            note       = "Resized only"
        else:
            aug_dict   = AUGMENTATIONS
            num_repeats = 1
            note       = "Original + 4 augs"

        # Split ORIGINAL pairs 60/20/20 (augmentation applied after split,
        # so val/test only contain resized originals — no data leakage)
        train_names, val_names, test_names = split_pairs(
            pairs.keys(), TRAIN_RATIO, VAL_RATIO, RANDOM_SEED
        )

        splits = {
            'train': train_names,
            'val':   val_names,
            'test':  test_names,
        }

        split_counts = {'train': 0, 'val': 0, 'test': 0}

        for split_name, names in splits.items():
            imgs_dir, gts_dir = make_split_dirs(OUTPUT_PATH, split_name, folder)

            for base_name in names:
                pair     = pairs[base_name]
                img_res, gt_res = resize_pair(pair['image'], pair['gt'], TARGET_SIZE)

                # Always save the resized original
                save_pair(img_res, gt_res, imgs_dir, gts_dir, base_name)
                split_counts[split_name] += 1

                # Augmentation only on train split
                if split_name == 'train' and aug_dict:
                    for repeat_idx in range(num_repeats):
                        for aug_name, aug_func in aug_dict.items():
                            img_aug, gt_aug = apply_augmentation(img_res, gt_res, aug_func)
                            suffix = f"_{aug_name}_r{repeat_idx+1}" if num_repeats > 1 else f"_{aug_name}"
                            save_pair(img_aug, gt_aug, imgs_dir, gts_dir, base_name, suffix)
                            split_counts['train'] += 1

        grand_orig  += original_count
        grand_train += split_counts['train']
        grand_val   += split_counts['val']
        grand_test  += split_counts['test']

        print(f"{folder:<20} {original_count:<12} {split_counts['train']:<10} {split_counts['val']:<10} {split_counts['test']:<10} {note}")
        print(f"  Train pairs: {len(train_names)} orig → {split_counts['train']} total | "
              f"Val: {split_counts['val']} | Test: {split_counts['test']}")

    print("-" * 80)
    print(f"{'TOTAL':<20} {grand_orig:<12} {grand_train:<10} {grand_val:<10} {grand_test:<10}")
    print("=" * 80)
    print(f"\nOutput saved to: {OUTPUT_PATH}")
    print(f"Structure: train/val/test → MT_*/Imgs/ and MT_*/GTs/")


if __name__ == "__main__":
    process_dataset()
