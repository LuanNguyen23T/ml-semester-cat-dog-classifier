"""Tải và chuẩn bị bộ ảnh thực tế từ internet (khác với tập huấn luyện) để demo web.

Bộ ảnh gồm:
1. 37 Giống Thú Cưng trong danh mục (12 mèo, 25 chó) - 10 ảnh / giống = 370 ảnh.
   - 21 giống chó: Lấy trực tiếp từ Stanford Dogs (nguồn ImageNet độc lập với Oxford).
   - 4 giống chó & 12 giống mèo: Tìm nạp từ Wikimedia Commons API (ảnh chụp đời thực có bản quyền mở).
2. Nhóm kiểm thử Nhận Diện Giống Chưa Tồn Tại & Ảnh Không Phải Chó Mèo:
   - 10 ảnh động vật hoang dã & đồ vật/phương tiện (Hổ, Sư tử, Gấu, Ô tô, Máy bay,...).
   - 10 ảnh các giống chó chưa từng học (Siberian Husky, Golden Retriever, German Shepherd,...).
"""

from __future__ import annotations

import io
import json
import random
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = REPO_ROOT / "demo_samples"

# 37 breeds catalog
BREED_LIST = [
    # 12 Cats
    ("Abyssinian", "Cat", "Abyssinian cat"),
    ("Bengal", "Cat", "Bengal cat"),
    ("Birman", "Cat", "Birman cat"),
    ("Bombay", "Cat", "Bombay cat"),
    ("British_Shorthair", "Cat", "British Shorthair cat"),
    ("Egyptian_Mau", "Cat", "Egyptian Mau cat"),
    ("Maine_Coon", "Cat", "Maine Coon cat"),
    ("Persian", "Cat", "Persian cat"),
    ("Ragdoll", "Cat", "Ragdoll cat"),
    ("Russian_Blue", "Cat", "Russian Blue cat"),
    ("Siamese", "Cat", "Siamese cat"),
    ("Sphynx", "Cat", "Sphynx cat"),
    # 25 Dogs
    ("american_bulldog", "Dog", "American Bulldog dog"),
    ("american_pit_bull_terrier", "Dog", "American Pit Bull Terrier dog"),
    ("basset_hound", "Dog", "Basset Hound dog"),
    ("beagle", "Dog", "Beagle dog"),
    ("boxer", "Dog", "Boxer dog"),
    ("chihuahua", "Dog", "Chihuahua dog"),
    ("english_cocker_spaniel", "Dog", "English Cocker Spaniel dog"),
    ("english_setter", "Dog", "English Setter dog"),
    ("german_shorthaired", "Dog", "German Shorthaired Pointer dog"),
    ("great_pyrenees", "Dog", "Great Pyrenees dog"),
    ("havanese", "Dog", "Havanese dog"),
    ("japanese_chin", "Dog", "Japanese Chin dog"),
    ("keeshond", "Dog", "Keeshond dog"),
    ("leonberger", "Dog", "Leonberger dog"),
    ("miniature_pinscher", "Dog", "Miniature Pinscher dog"),
    ("newfoundland", "Dog", "Newfoundland dog"),
    ("pomeranian", "Dog", "Pomeranian dog"),
    ("pug", "Dog", "Pug dog"),
    ("saint_bernard", "Dog", "Saint Bernard dog"),
    ("samoyed", "Dog", "Samoyed dog"),
    ("scottish_terrier", "Dog", "Scottish Terrier dog"),
    ("shiba_inu", "Dog", "Shiba Inu dog"),
    ("staffordshire_bull_terrier", "Dog", "Staffordshire Bull Terrier dog"),
    ("wheaten_terrier", "Dog", "Soft-coated Wheaten Terrier dog"),
    ("yorkshire_terrier", "Dog", "Yorkshire Terrier dog"),
]

# Mapping Stanford Dogs overlapping labels
OVERLAPPING_STANFORD = {
    "basset_hound": 10,
    "beagle": 11,
    "boxer": 91,
    "chihuahua": 0,
    "english_cocker_spaniel": 68,
    "english_setter": 61,
    "german_shorthaired": 59,
    "great_pyrenees": 105,
    "japanese_chin": 1,
    "keeshond": 109,
    "leonberger": 103,
    "miniature_pinscher": 86,
    "newfoundland": 104,
    "pomeranian": 107,
    "pug": 102,
    "saint_bernard": 96,
    "samoyed": 106,
    "scottish_terrier": 48,
    "staffordshire_bull_terrier": 28,
    "wheaten_terrier": 51,
    "yorkshire_terrier": 36,
}

# Stanford Dog labels for novel/unseen dog breeds
UNSEEN_STANFORD_DOGS = {
    "golden_retriever": 56,
    "rottweiler": 83,
    "german_shepherd": 84,
    "doberman": 85,
    "french_bulldog": 94,
    "siberian_husky": 99,
}

NON_PET_SEARCH_QUERIES = [
    ("tiger", "Bengal tiger in wild"),
    ("lion", "African lion male"),
    ("elephant", "African elephant wildlife"),
    ("bear", "Brown grizzly bear"),
    ("eagle", "Bald eagle flying bird"),
    ("sports_car", "Sports car automobile modern"),
    ("airplane", "Commercial airplane aircraft"),
    ("bicycle", "Mountain bicycle road"),
    ("guitar", "Acoustic guitar instrument"),
    ("landscape", "Mountain landscape lake scenery"),
]


def fetch_wikimedia_urls(query: str, limit: int = 15) -> List[str]:
    """Tìm nạp danh sách URL ảnh từ Wikimedia Commons API."""
    params = {
        "action": "query",
        "generator": "search",
        "gsrnamespace": "6",
        "gsrsearch": query,
        "gsrlimit": str(limit * 2),
        "prop": "imageinfo",
        "iiprop": "url|mime|size",
        "format": "json",
    }
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "PetClassifierResearch/1.0 (contact: student@demo.edu.vn)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode())
    except Exception as exc:
        print(f"  [CẢNH BÁO] Lỗi tìm kiếm Wikimedia query '{query}': {exc}")
        return []

    pages = data.get("query", {}).get("pages", {})
    image_urls = []
    for pid, p in pages.items():
        ii = p.get("imageinfo", [{}])[0]
        mime = ii.get("mime", "")
        u = ii.get("url", "")
        size = ii.get("size", 0)
        # Bỏ qua icon hoặc ảnh quá nhỏ / không phải jpeg, png
        if ("jpeg" in mime or "jpg" in mime or "png" in mime) and size > 25000:
            image_urls.append(u)
        if len(image_urls) >= limit:
            break
    return image_urls


def download_and_save_image(url: str, save_path: Path, max_size: int = 1024) -> bool:
    """Tải ảnh từ URL, chuẩn hóa sang RGB JPEG và lưu cục bộ."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "PetClassifierResearch/1.0 (contact: student@demo.edu.vn)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw_bytes = resp.read()
        img = Image.open(io.BytesIO(raw_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        # Resize nhẹ nếu kích thước quá lớn để web load nhanh
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(save_path, "JPEG", quality=90)
        return True
    except Exception as exc:
        return False


def main() -> None:
    print("=" * 80)
    print("CHUẨN BỊ BỘ ẢNH THỰC TẾ ĐỂ DEMO WEB & KIỂM THỬ MODULE GIỐNG LẠ")
    print("=" * 80)

    # 1. Tải Stanford Dogs dataset cục bộ cho 21 giống chó và các giống lạ
    print("\n[Bước 1] Nạp Stanford Dogs từ cache nội bộ...")
    ds = load_dataset("maurice-fp/stanford-dogs")["test"]
    st_buckets: Dict[int, List[object]] = defaultdict(list)
    for row in ds:
        st_buckets[row["label"]].append(row["image"])

    rng = random.Random(42)

    # 2. Xử lý 37 giống thú cưng (10 ảnh / giống)
    known_dir = DEMO_DIR / "known_breeds"
    print(f"\n[Bước 2] Lưu 37 giống thú cưng (10 ảnh/giống) vào: {known_dir}")

    total_known_saved = 0
    for breed_name, species, search_query in tqdm(BREED_LIST, desc="Chuẩn bị 37 giống"):
        breed_folder = known_dir / breed_name
        breed_folder.mkdir(parents=True, exist_ok=True)

        existing = list(breed_folder.glob("*.jpg"))
        if len(existing) >= 10:
            total_known_saved += len(existing)
            continue

        saved_count = len(existing)

        # Ưu tiên lấy từ Stanford Dogs nếu có
        if breed_name in OVERLAPPING_STANFORD:
            st_label = OVERLAPPING_STANFORD[breed_name]
            candidates = st_buckets[st_label]
            chosen = rng.sample(candidates, min(10, len(candidates)))
            for idx, pil_img in enumerate(chosen, start=1):
                out_path = breed_folder / f"sample_{idx:02d}.jpg"
                if not out_path.exists():
                    img = pil_img.convert("RGB")
                    if max(img.size) > 1024:
                        img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                    img.save(out_path, "JPEG", quality=90)
                saved_count += 1
        else:
            # Tìm nạp từ Wikimedia Commons cho 12 mèo và 4 giống chó còn lại
            urls = fetch_wikimedia_urls(search_query, limit=15)
            img_idx = 1
            for u in urls:
                if saved_count >= 10:
                    break
                out_path = breed_folder / f"sample_{img_idx:02d}.jpg"
                if not out_path.exists():
                    success = download_and_save_image(u, out_path)
                    if success:
                        saved_count += 1
                        img_idx += 1
                        time.sleep(0.3)  # Giảm tải cho Wikimedia API

        total_known_saved += saved_count

    print(f"\n[OK] Đã hoàn thành nạp {total_known_saved} ảnh của 37 giống loài.")

    # 3. Chuẩn bị nhóm ảnh không liên quan đến chó mèo (Non-pet objects / Wild animals)
    non_pet_dir = DEMO_DIR / "unknown_and_non_pet" / "non_pet_objects"
    non_pet_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Bước 3] Tải 10 ảnh không liên quan chó mèo vào: {non_pet_dir}")

    non_pet_count = 0
    for name, query in NON_PET_SEARCH_QUERIES:
        out_path = non_pet_dir / f"{name}.jpg"
        if out_path.exists():
            non_pet_count += 1
            continue
        urls = fetch_wikimedia_urls(query, limit=5)
        for u in urls:
            if download_and_save_image(u, out_path):
                non_pet_count += 1
                time.sleep(0.3)
                break

    print(f"[OK] Đã lưu {non_pet_count} ảnh không liên quan chó mèo.")

    # 4. Chuẩn bị nhóm ảnh các giống chó lạ chưa từng học (Unseen novel dog breeds)
    unseen_dogs_dir = DEMO_DIR / "unknown_and_non_pet" / "unseen_dog_breeds"
    unseen_dogs_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Bước 4] Trích xuất ảnh các giống chó lạ ngoài danh mục vào: {unseen_dogs_dir}")

    unseen_count = 0
    for dog_name, st_label in UNSEEN_STANFORD_DOGS.items():
        candidates = st_buckets[st_label]
        # Lấy 2 ảnh mỗi giống lạ
        chosen = rng.sample(candidates, min(2, len(candidates)))
        for idx, pil_img in enumerate(chosen, start=1):
            out_path = unseen_dogs_dir / f"{dog_name}_{idx:02d}.jpg"
            if not out_path.exists():
                img = pil_img.convert("RGB")
                if max(img.size) > 1024:
                    img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                img.save(out_path, "JPEG", quality=90)
            unseen_count += 1

    print(f"[OK] Đã lưu {unseen_count} ảnh giống chó lạ (Husky, Golden Retriever, Rottweiler, German Shepherd, French Bulldog, Doberman).")

    print("\n" + "=" * 80)
    print("HOÀN TẤT! Toàn bộ ảnh demo đã sẵn sàng tại thư mục:")
    print(f"  -> {DEMO_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
