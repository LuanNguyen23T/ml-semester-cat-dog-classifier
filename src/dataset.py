"""
Oxford-IIIT Pet Dataset & Advanced Data Augmentation Pipeline.
Includes:
- Reused and upgraded CLAHE and Bilateral filtering from CLIP-Animal-Classifier.
- Teacher's requirement: Extreme Part-Level Cropping & Random Erasing
  (enables recognition from isolated parts like paws, tail, ears, fur patches).
- Support for both 37 breeds classification and binary Cat vs Dog classification.
"""

import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class CLAHETransform:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
    on the L channel in CIE-LAB color space to enhance fur texture and edge details.
    Reused and modernized from CLIP-Animal-Classifier.
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8), p=0.5):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.rand() > self.p:
            return img
        img_np = np.array(img)
        # If RGBA or Grayscale, convert to RGB first
        if img_np.ndim == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)

        bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        l_clahe = clahe.apply(l)
        enhanced_bgr = cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)
        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(enhanced_rgb)


class BilateralFilterTransform:
    """
    Applies Bilateral Filter to smooth image background while strictly preserving
    fine pet fur borders, whiskers, and contours.
    Reused from CLIP-Animal-Classifier.
    """
    def __init__(self, d=7, sigma_color=50, sigma_space=50, p=0.3):
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.rand() > self.p:
            return img
        img_np = np.array(img)
        if img_np.ndim == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)

        bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        filtered_bgr = cv2.bilateralFilter(bgr, d=self.d, sigmaColor=self.sigma_color, sigmaSpace=self.sigma_space)
        filtered_rgb = cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(filtered_rgb)


class PartCropTransform:
    """
    Teacher's Requirement: Part-Level Augmentation.
    Forces model to recognize pet breeds / species when ONLY isolated parts
    (paw, tail, ear, muzzle, fur patch) are visible.
    - 50% probability: Standard RandomResizedCrop (scale: 0.5 - 1.0)
    - 25% probability: Upper part crop (Head / Ears / Face)
    - 25% probability: Lower part crop (Paws / Legs / Torso / Tail)
    """
    def __init__(self, size=(224, 224), part_prob=0.5, part_scale=(0.10, 0.40), global_scale=(0.5, 1.0)):
        self.size = size
        self.part_prob = part_prob
        self.part_scale = part_scale
        self.global_crop = T.RandomResizedCrop(size=size, scale=global_scale, ratio=(0.75, 1.33))

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.rand() >= self.part_prob:
            return self.global_crop(img)

        w, h = img.size
        # Phân chia vùng giải phẫu: nửa trên (tai/mặt) và nửa dưới (chân/thân/đuôi)
        if np.random.rand() < 0.5:
            # Vùng trên: Tai, mắt, mặt
            upper_sub = img.crop((0, 0, w, int(h * 0.65)))
            return T.RandomResizedCrop(size=self.size, scale=self.part_scale, ratio=(0.75, 1.33))(upper_sub)
        else:
            # Vùng dưới: Bàn chân, móng, thân và đuôi
            lower_sub = img.crop((0, int(h * 0.38), w, h))
            return T.RandomResizedCrop(size=self.size, scale=self.part_scale, ratio=(0.75, 1.33))(lower_sub)


def get_transforms(img_size=224, is_train=True, use_part_aug=True, use_clahe=True):
    """
    Builds data transform pipelines for training and evaluation.
    """
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])

    if is_train:
        transform_list = []
        if use_clahe:
            transform_list.append(CLAHETransform(p=0.4))
            transform_list.append(BilateralFilterTransform(p=0.25))

        if use_part_aug:
            # Multi-scale part crop (ear, tail, leg, torso)
            transform_list.append(PartCropTransform(size=(img_size, img_size), part_prob=0.5))
        else:
            transform_list.append(T.RandomResizedCrop(img_size, scale=(0.6, 1.0)))

        transform_list.extend([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            normalize,
            # Random Erasing: occlude large regions so the model relies on other parts
            T.RandomErasing(p=0.35, scale=(0.08, 0.35), value="random")
        ])
        return T.Compose(transform_list)
    else:
        # Standard validation / test transform
        return T.Compose([
            T.Resize((int(img_size * 1.14), int(img_size * 1.14))),
            T.CenterCrop(img_size),
            T.ToTensor(),
            normalize
        ])


class OxfordPetDataset(Dataset):
    """
    PyTorch Dataset for Oxford-IIIT Pet.
    Supports:
      - task='breed': 37 classes (0..36)
      - task='binary': 2 classes (0: Cat, 1: Dog)
    """
    def __init__(self, root_dir, split="trainval", task="breed", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.task = task
        self.transform = transform

        self.images_dir = os.path.join(root_dir, "images")
        if not os.path.exists(self.images_dir):
            # 1. Kiểm tra file nén zip nội bộ nếu có
            zip_candidates = [
                root_dir + ".zip",
                os.path.join(os.path.dirname(root_dir), "oxford-iiit-pet.zip"),
                os.path.join(root_dir, "oxford-iiit-pet.zip")
            ]
            for z_path in zip_candidates:
                if os.path.exists(z_path):
                    print(f">> [DATASET] Đang giải nén {z_path}...")
                    import zipfile
                    with zipfile.ZipFile(z_path, 'r') as zf:
                        zf.extractall(path=os.path.dirname(root_dir))
                    break

            # 2. Tự động tải từ torchvision nếu chưa có dữ liệu (hỗ trợ clone mới trên máy khác/Colab)
            if not os.path.exists(self.images_dir):
                print(f">> [DATASET] Chưa có ảnh tại {root_dir}. Đang tự động tải Oxford-IIIT Pet từ torchvision...")
                from torchvision.datasets import OxfordIIITPet as TVOxfordPet
                download_target = os.path.dirname(root_dir) if os.path.basename(root_dir) == "oxford-iiit-pet" else root_dir
                os.makedirs(download_target, exist_ok=True)
                TVOxfordPet(root=download_target, download=True)

        self.split_file = os.path.join(root_dir, "annotations", f"{split}.txt")
        self.list_file = os.path.join(root_dir, "annotations", "list.txt")

        # Load metadata mappings: class_id -> (breed_name, species)
        self.classes, self.class_names, self.species_map = self._load_classes()

        # Load split items
        self.samples = self._load_split_samples()

    def _load_classes(self):
        classes = {}
        class_names = []
        species_map = {}
        with open(self.list_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                name, class_id, spec_id = parts[0], int(parts[1]), int(parts[2])
                breed_name = "_".join(name.split("_")[:-1])
                species = "Cat" if spec_id == 1 else "Dog"
                classes[class_id] = (breed_name, species)

        sorted_ids = sorted(classes.keys())
        class_names = [classes[cid][0] for cid in sorted_ids]
        for cid in sorted_ids:
            species_map[cid - 1] = 0 if classes[cid][1] == "Cat" else 1

        return classes, class_names, species_map

    def _load_split_samples(self):
        samples = []
        with open(self.split_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                img_name = parts[0]
                class_id = int(parts[1]) - 1  # 0-indexed (0..36)
                spec_id = int(parts[2]) - 1   # 0: Cat, 1: Dog
                
                img_path = os.path.join(self.images_dir, f"{img_name}.jpg")
                if os.path.exists(img_path):
                    target = class_id if self.task == "breed" else spec_id
                    samples.append((img_path, target, class_id, spec_id, img_name))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, target, class_id, spec_id, img_name = self.samples[idx]
        with Image.open(img_path) as img:
            img = img.convert("RGB")

        if self.transform is not None:
            tensor = self.transform(img)
        else:
            tensor = TF.to_tensor(img)

        return tensor, target, {
            "image_name": img_name,
            "breed_name": self.class_names[class_id],
            "species": "Cat" if spec_id == 0 else "Dog",
            "class_id": class_id,
            "species_id": spec_id
        }



class OxfordPetCLIPDataset(OxfordPetDataset):
    """
    Oxford-IIIT Pet Dataset paired with natural language text prompts for CLIP training.
    Yields (image_tensor, text_tokens, target_class_id).
    """
    PROMPT_TEMPLATES = [
        "a photo of a {}, a type of pet.",
        "a photo of the cute {}.",
        "a close-up photo of a {}.",
        "a portrait of a beautiful {}.",
        "a photo of a pet {}."
    ]

    def __init__(self, tokenizer, *args, is_train=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.is_train = is_train

    def __getitem__(self, idx):
        tensor, target, meta = super().__getitem__(idx)
        breed_clean = meta["breed_name"].replace("_", " ")
        if self.is_train:
            import random
            template = random.choice(self.PROMPT_TEMPLATES)
        else:
            template = self.PROMPT_TEMPLATES[0]
        caption = template.format(breed_clean)
        text_tokens = self.tokenizer.encode(caption)
        return tensor, text_tokens, target
