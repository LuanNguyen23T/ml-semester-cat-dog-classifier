"""
Pipeline tạo New Dataset 100% Ảnh Mới (Không trùng lặp với Oxford-IIIT Pet).
Nguồn dữ liệu độc lập:
1. 22 Giống Chó: Stanford Dogs Dataset (ImageNet WordNet Synsets, ~150-180 ảnh/giống)
2. 3 Giống Chó còn thiếu (American Bulldog, Havanese, Shiba Inu): Wikimedia High-Res Media Archive
3. 12 Giống Mèo: Wikimedia High-Res Feline Media Archive
Mỗi ảnh được tải về qua CDN tốc độ cao, chuẩn hóa định dạng RGB JPEG, lưu vào dataset/new_dataset/images.
Toàn bộ annotations (trainval.txt, list.txt) được sinh mới 100%.
"""

import os
import sys
import json
import time
import shutil
import urllib.request
import urllib.parse
from PIL import Image

OXFORD_CLASSES = [
    # 12 Cats (0..11)
    ("Abyssinian", "Cat", 1, "Abyssinian cat"),
    ("Bengal", "Cat", 1, "Bengal cat"),
    ("Birman", "Cat", 1, "Birman cat"),
    ("Bombay", "Cat", 1, "Bombay cat"),
    ("British_Shorthair", "Cat", 1, "British Shorthair cat"),
    ("Egyptian_Mau", "Cat", 1, "Egyptian Mau cat"),
    ("Maine_Coon", "Cat", 1, "Maine Coon cat"),
    ("Persian", "Cat", 1, "Persian cat"),
    ("Ragdoll", "Cat", 1, "Ragdoll cat"),
    ("Russian_Blue", "Cat", 1, "Russian Blue cat"),
    ("Siamese", "Cat", 1, "Siamese cat"),
    ("Sphynx", "Cat", 1, "Sphynx cat"),
    # 25 Dogs (12..36)
    ("american_bulldog", "Dog", 2, "American Bulldog"),
    ("american_pit_bull_terrier", "Dog", 2, "American Staffordshire Terrier"),
    ("basset_hound", "Dog", 2, "Basset Hound"),
    ("beagle", "Dog", 2, "Beagle"),
    ("boxer", "Dog", 2, "Boxer dog"),
    ("chihuahua", "Dog", 2, "Chihuahua dog"),
    ("english_cocker_spaniel", "Dog", 2, "English Cocker Spaniel"),
    ("english_setter", "Dog", 2, "English Setter"),
    ("german_shorthaired", "Dog", 2, "German Shorthaired Pointer"),
    ("great_pyrenees", "Dog", 2, "Great Pyrenees"),
    ("havanese", "Dog", 2, "Havanese dog"),
    ("japanese_chin", "Dog", 2, "Japanese Chin"),
    ("keeshond", "Dog", 2, "Keeshond"),
    ("leonberger", "Dog", 2, "Leonberger"),
    ("miniature_pinscher", "Dog", 2, "Miniature Pinscher"),
    ("newfoundland", "Dog", 2, "Newfoundland dog"),
    ("pomeranian", "Dog", 2, "Pomeranian dog"),
    ("pug", "Dog", 2, "Pug dog"),
    ("saint_bernard", "Dog", 2, "Saint Bernard dog"),
    ("samoyed", "Dog", 2, "Samoyed dog"),
    ("scottish_terrier", "Dog", 2, "Scottish Terrier"),
    ("shiba_inu", "Dog", 2, "Shiba Inu dog"),
    ("staffordshire_bull_terrier", "Dog", 2, "Staffordshire Bull Terrier"),
    ("wheaten_terrier", "Dog", 2, "Soft-coated Wheaten Terrier"),
    ("yorkshire_terrier", "Dog", 2, "Yorkshire Terrier"),
]

STANFORD_MAPPING = {
    'american_pit_bull_terrier': 'American Staffordshire Terrier',
    'basset_hound': 'Basset',
    'beagle': 'Beagle',
    'boxer': 'Boxer',
    'chihuahua': 'Chihuahua',
    'english_cocker_spaniel': 'Cocker Spaniel',
    'english_setter': 'English Setter',
    'german_shorthaired': 'German Short Haired Pointer',
    'great_pyrenees': 'Great Pyrenees',
    'japanese_chin': 'Japanese Spaniel',
    'keeshond': 'Keeshond',
    'leonberger': 'Leonberg',
    'miniature_pinscher': 'Miniature Pinscher',
    'newfoundland': 'Newfoundland',
    'pomeranian': 'Pomeranian',
    'pug': 'Pug',
    'saint_bernard': 'Saint Bernard',
    'samoyed': 'Samoyed',
    'scottish_terrier': 'Scotch Terrier',
    'staffordshire_bull_terrier': 'Staffordshire Bullterrier',
    'wheaten_terrier': 'Soft Coated Wheaten Terrier',
    'yorkshire_terrier': 'Yorkshire Terrier'
}


def clear_and_init_new_dataset(output_dir: str):
    """Xóa sạch mọi liên kết ảnh cũ của Oxford, khởi tạo new_dataset thuần túy 100% ảnh mới."""
    images_dst = os.path.join(output_dir, "images")
    annos_dst = os.path.join(output_dir, "annotations")
    os.makedirs(images_dst, exist_ok=True)
    os.makedirs(annos_dst, exist_ok=True)

    removed = 0
    for f in os.listdir(images_dst):
        p = os.path.join(images_dst, f)
        if os.path.islink(p):
            os.remove(p)
            removed += 1
    if removed > 0:
        print(f">> Đã xóa toàn bộ {removed} symlinks ảnh Oxford cũ ra khỏi {images_dst}")

    for f in ["trainval.txt", "list.txt"]:
        with open(os.path.join(annos_dst, f), "w") as fp:
            fp.write("# New Dataset 100% External Fresh Images (Stanford Dogs + Open Archives)\n")


def fetch_wikimedia_images(query: str, target_count: int = 150):
    """Tìm kiếm đa trang qua Wikimedia API để lấy danh sách URL ảnh kích thước chuẩn."""
    urls = []
    offset = 0
    batch_size = 50
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"}

    while len(urls) < target_count:
        url = (
            f"https://commons.wikimedia.org/w/api.php?action=query&format=json"
            f"&generator=search&gsrsearch={urllib.parse.quote(query)}&gsrnamespace=6"
            f"&gsrlimit={batch_size}&gsroffset={offset}&prop=imageinfo&iiprop=url&iiurlwidth=800"
        )
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.load(resp)
                pages = data.get("query", {}).get("pages", {})
                if not pages:
                    break
                for p in pages.values():
                    if "imageinfo" in p and p["imageinfo"]:
                        info = p["imageinfo"][0]
                        img_url = info.get("thumburl", info.get("url"))
                        clean_url = img_url.split("?")[0]
                        if clean_url.lower().endswith((".jpg", ".jpeg", ".png")):
                            urls.append(img_url)
                            if len(urls) >= target_count:
                                break
                if "continue" in data and "gsroffset" in data["continue"]:
                    offset = data["continue"]["gsroffset"]
                else:
                    break
        except Exception:
            break
        time.sleep(0.3)
    return urls


def download_image(url: str, save_path: str):
    """Tải và chuyển đổi ảnh về chuẩn RGB JPEG."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            img = Image.open(resp).convert("RGB")
            if img.width >= 120 and img.height >= 120:
                img.save(save_path, "JPEG", quality=90)
                return True
    except Exception:
        pass
    return False


def build_new_dataset(output_dir: str = "dataset/new_dataset", images_per_class: int = 150):
    """Thu thập ảnh mới hoàn toàn cho cả 37 giống loài (Mục tiêu 150 ảnh/giống)."""
    clear_and_init_new_dataset(output_dir)
    images_dst = os.path.join(output_dir, "images")
    trainval_file = os.path.join(output_dir, "annotations", "trainval.txt")
    list_file = os.path.join(output_dir, "annotations", "list.txt")

    print(f"\n==================================================================")
    print(f">> KHỞI ĐỘNG TIẾN TRÌNH TẢI {images_per_class} ẢNH MỚI/GIỐNG CHO 37 LOÀI")
    print(f"   Thư mục đích: {os.path.abspath(output_dir)}")
    print(f"   (100% Ảnh mới, hoàn toàn không sử dụng ảnh của Oxford-IIIT Pet)")
    print(f"==================================================================\n")

    # 1. Tải 22 giống chó từ Stanford Dogs Dataset (ImageNet WordNet)
    print(f">> [PHẦN 1/2] Thu thập 22 giống chó từ Stanford Dogs Dataset (Hugging Face Stream)...")
    try:
        from datasets import load_dataset
        stanford_stream = load_dataset('amaye15/stanford-dogs', split='train', streaming=True)
        label_names = stanford_stream.features['label'].names
        stanford_to_oxford = {}
        for ox_name, stan_name in STANFORD_MAPPING.items():
            if stan_name in label_names:
                idx = label_names.index(stan_name)
                ox_cid = next(i + 1 for i, (b, _, _, _) in enumerate(OXFORD_CLASSES) if b == ox_name)
                stanford_to_oxford[idx] = (ox_name, ox_cid)

        dog_counts = {idx: 0 for idx in stanford_to_oxford}
        new_annotations = []
        start_t = time.time()

        for item in stanford_stream:
            lbl = item['label']
            if lbl in stanford_to_oxford and dog_counts[lbl] < images_per_class:
                ox_name, ox_cid = stanford_to_oxford[lbl]
                img = item['pixel_values'].convert('RGB')
                seq = dog_counts[lbl] + 1
                img_name = f"{ox_name}_new_{seq}"
                save_path = os.path.join(images_dst, f"{img_name}.jpg")
                img.save(save_path, "JPEG", quality=90)
                dog_counts[lbl] += 1
                new_annotations.append(f"{img_name} {ox_cid} 2 1\n")

                if sum(dog_counts.values()) % 100 == 0:
                    print(f"   ... Đã lưu {sum(dog_counts.values())} ảnh chó Stanford ({time.time() - start_t:.1f}s)")

                if all(c >= images_per_class for c in dog_counts.values()):
                    break

        with open(trainval_file, "a") as f:
            f.writelines(new_annotations)
        with open(list_file, "a") as f:
            f.writelines(new_annotations)

        total_dogs = sum(dog_counts.values())
        print(f">> [PHẦN 1/2 HOÀN TẤT] Đã tải thành công {total_dogs} ảnh chó mới từ Stanford Dogs!\n")
    except Exception as e:
        print(f">> [Lỗi Phần 1] Stanford Dogs: {e}")

    # 2. Tải 12 giống mèo và 3 giống chó thiếu
    print(f">> [PHẦN 2/2] Thu thập 12 giống mèo & 3 giống chó còn lại qua kho ảnh mở...")
    web_breeds = [cls for cls in OXFORD_CLASSES if cls[0] not in STANFORD_MAPPING]

    web_annotations = []
    web_counts = {}
    for i, (ox_name, species, spec_id, search_q) in enumerate(web_breeds, 1):
        ox_cid = next(idx + 1 for idx, (b, _, _, _) in enumerate(OXFORD_CLASSES) if b == ox_name)
        print(f"   [{i}/{len(web_breeds)}] Tìm nạp [{ox_name}] ({species}) ...", end=" ", flush=True)
        urls = fetch_wikimedia_images(search_q, target_count=images_per_class)
        saved = 0
        for u in urls:
            seq = saved + 1
            img_name = f"{ox_name}_new_{seq}"
            save_path = os.path.join(images_dst, f"{img_name}.jpg")
            if download_image(u, save_path):
                saved += 1
                web_annotations.append(f"{img_name} {ox_cid} {spec_id} 1\n")
            if saved >= images_per_class:
                break
        web_counts[ox_name] = saved
        print(f"Lưu thành công: {saved}/{images_per_class} ảnh.")

    with open(trainval_file, "a") as f:
        f.writelines(web_annotations)
    with open(list_file, "a") as f:
        f.writelines(web_annotations)

    # 3. Tổng kết dữ liệu
    total_imgs = len([f for f in os.listdir(images_dst) if f.endswith(".jpg")])
    summary = {
        "dataset_name": "New Dataset 100% External Fresh Images",
        "description": "Toàn bộ ảnh mới 100%, độc lập hoàn toàn với Oxford-IIIT Pet",
        "total_new_images": total_imgs,
        "total_classes": 37,
        "images_dir": os.path.abspath(images_dst),
        "annotations_dir": os.path.abspath(os.path.join(output_dir, "annotations"))
    }
    summary_path = os.path.join(output_dir, "dataset_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n==================================================================")
    print(f">> TẢI VỀ THÀNH CÔNG TỔNG CỘNG: {total_imgs} ẢNH MỚI CHO 37 LOÀI!")
    print(f"- Toàn bộ lưu tại: {os.path.abspath(images_dst)}")
    print(f"- File tổng kết: {summary_path}")
    print(f"==================================================================\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="dataset/new_dataset")
    parser.add_argument("--images_per_class", type=int, default=150, help="Số ảnh mới cho mỗi giống")
    args = parser.parse_args()

    build_new_dataset(args.output_dir, images_per_class=args.images_per_class)
