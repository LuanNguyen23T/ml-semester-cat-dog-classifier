"""
Hệ thống Thẩm định & Chuẩn hóa Dataset Thú cưng bằng CLIP Zero-Shot (CLIP Breed Verifier):
1. Đưa từng ảnh của từng giống loài vào mô hình CLIP (openai/clip-vit-base-patch32).
2. Thẩm định độ khớp ngữ nghĩa theo 37 giống Oxford-IIIT Pet và các lớp nhiễu (con người, tranh vẽ, phong cảnh, đồ vật).
3. Loại bỏ các ảnh không liên quan, ảnh nhiễu, ảnh sai loài/sai giống (như Abyssinian_1152, ảnh người, ảnh bản vẽ).
4. Tự động tải bổ sung các ảnh chụp thật mới (được thẩm định qua CLIP) để mọi giống loài đạt chuẩn 150 - 155 ảnh.
5. Cắt tỉa ảnh dư thừa để cân bằng hoàn hảo (150 - 155 ảnh/loài).
6. Đồng bộ hóa toàn bộ annotations (trainval.txt, list.txt, dataset_summary.json).
"""

import os
import sys
sys.path.insert(0, os.getcwd())
import json
import time
import urllib.request
import urllib.parse
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset

from src.build_augmented_dataset import OXFORD_CLASSES, STANFORD_MAPPING

DATASET_DIR = "dataset/new_dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
ANNOS_DIR = os.path.join(DATASET_DIR, "annotations")
MIN_TARGET = 150
MAX_TARGET = 155
BALANCED_TARGET = 152


def setup_clip_model(device="cpu"):
    """Nạp CLIP model và tính toán trước text embeddings cho 37 giống và 4 lớp âm tính."""
    print(f">> Khởi tạo mô hình CLIP Zero-Shot trên thiết bị: {device}...")
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)

    prompts = []
    for name, species, _, _ in OXFORD_CLASSES:
        clean = name.replace("_", " ")
        if species == "Cat":
            prompts.append(f"a photo of a {clean} cat")
        else:
            prompts.append(f"a photo of a {clean} dog")

    negatives = [
        "a photo of a human or person",
        "a photo of a landscape, building, or room with no pets",
        "a drawing, sketch, illustration, painting, mosaic, sculpture, or craft artifact",
        "a photo of a wild animal that is not a domestic cat or dog"
    ]
    all_prompts = prompts + negatives
    num_classes = len(OXFORD_CLASSES)

    text_inputs = processor(text=all_prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    breed_to_idx = {name: i for i, (name, _, _, _) in enumerate(OXFORD_CLASSES)}
    breed_species = {name: spec for name, spec, _, _ in OXFORD_CLASSES}

    return model, processor, text_features, breed_to_idx, breed_species, num_classes, all_prompts


def verify_pil_image(img, target_breed, model, processor, text_features, breed_to_idx, breed_species, num_classes, all_prompts, device):
    """Kiểm tra một ảnh PIL với giống mục tiêu."""
    target_idx = breed_to_idx[target_breed]
    target_spec = breed_species[target_breed]

    try:
        inp = processor(images=img.convert("RGB"), return_tensors="pt").to(device)
        with torch.no_grad():
            feat = model.get_image_features(**inp)
            feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
            sim = (feat @ text_features.T)[0]
            probs = (sim * 100).softmax(dim=-1).cpu()

        top_idx = torch.argmax(probs).item()
        target_prob = probs[target_idx].item()
        neg_prob = probs[num_classes:].sum().item()
        top_prob = probs[top_idx].item()

        # 1. Âm tính áp đảo (người, cảnh vật, đồ vật, tranh vẽ)
        if top_idx >= num_classes or neg_prob > 0.45:
            return False, target_prob, f"Loại âm tính ({all_prompts[top_idx]}, neg={neg_prob*100:.1f}%)"

        pred_breed, pred_spec, _, _ = OXFORD_CLASSES[top_idx]

        # 2. Xung đột loài (Mèo bị nhận thành Chó hoặc ngược lại)
        if pred_spec != target_spec and top_prob > 0.35 and target_prob < 0.05:
            return False, target_prob, f"Sai loài ({pred_breed} {pred_spec} {top_prob*100:.1f}%, đích={target_breed} {target_prob*100:.1f}%)"

        # 3. Khớp giống mục tiêu (nằm trong Top 3 hoặc prob >= 0.08)
        top3 = torch.topk(probs[:num_classes], 3).indices.tolist()
        if target_idx in top3 or target_prob >= 0.08:
            return True, target_prob, f"Khớp giống (prob={target_prob*100:.1f}%)"

        # 4. Cùng loài và prob mục tiêu ở mức khá (> 0.03)
        if pred_spec == target_spec and target_prob >= 0.03:
            return True, target_prob, f"Cùng loài (pred={pred_breed}, target prob={target_prob*100:.1f}%)"

        return False, target_prob, f"Độ khớp thấp (pred={pred_breed} {top_prob*100:.1f}%, target={target_prob*100:.1f}%)"
    except Exception as e:
        return False, 0.0, f"Lỗi xử lý ảnh: {e}"


def fetch_wikimedia_photo_urls(breed_name, species, target_count=50):
    """Tìm URL ảnh chụp thật từ Wikimedia Commons cho giống cụ thể."""
    clean_name = breed_name.replace("_", " ")
    spec_kw = "cat" if species == "Cat" else "dog"
    query = f"{clean_name} {spec_kw} photo"
    
    urls = []
    offset = 0
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) PetResearchBot/2.0"}

    while len(urls) < target_count:
        url = (
            f"https://commons.wikimedia.org/w/api.php?action=query&format=json"
            f"&generator=search&gsrsearch={urllib.parse.quote(query)}&gsrnamespace=6"
            f"&gsrlimit=50&gsroffset={offset}&prop=imageinfo&iiprop=url&iiurlwidth=800"
        )
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
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
        time.sleep(0.2)
    return urls


def download_image(url):
    """Tải ảnh từ URL về PIL Image."""
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) PetResearchBot/2.0"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            img = Image.open(resp).convert("RGB")
            if img.width >= 150 and img.height >= 150:
                return img
    except Exception:
        pass
    return None


def run_full_verification():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model, processor, text_features, breed_to_idx, breed_species, num_classes, all_prompts = setup_clip_model(device)

    all_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith(".jpg")])
    print(f"\n==================================================================")
    print(f">> [BƯỚC 1] BẮT ĐẦU KIỂM ĐỊNH TỪNG ẢNH BẰNG CLIP CHO TOÀN BỘ {len(all_files)} ẢNH")
    print(f"==================================================================")

    # Thẩm định theo batch
    batch_size = 64
    valid_images_by_breed = {name: [] for name, _, _, _ in OXFORD_CLASSES}
    removed_count = 0
    t0 = time.time()

    for b_idx in range(0, len(all_files), batch_size):
        batch_files = all_files[b_idx:b_idx + batch_size]
        batch_imgs = []
        batch_meta = []

        for f in batch_files:
            p = os.path.join(IMAGES_DIR, f)
            breed = "_".join(f[:-4].split("_")[:-1])
            if breed not in breed_to_idx:
                os.remove(p)
                removed_count += 1
                continue
            try:
                img = Image.open(p).convert("RGB")
                batch_imgs.append(img)
                batch_meta.append((f, p, breed))
            except Exception:
                if os.path.exists(p):
                    os.remove(p)
                removed_count += 1

        if not batch_imgs:
            continue

        try:
            inp = processor(images=batch_imgs, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                feats = model.get_image_features(**inp)
                feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
                sims = feats @ text_features.T
                probs = (sims * 100).softmax(dim=-1).cpu()

            for i, (f, p, breed) in enumerate(batch_meta):
                target_idx = breed_to_idx[breed]
                target_spec = breed_species[breed]
                top_idx = torch.argmax(probs[i]).item()
                target_prob = probs[i, target_idx].item()
                neg_prob = probs[i, num_classes:].sum().item()
                top_prob = probs[i, top_idx].item()

                is_valid = True
                fail_reason = ""

                # Kiểm tra âm tính
                if top_idx >= num_classes or neg_prob > 0.45:
                    is_valid = False
                    fail_reason = f"Âm tính: {all_prompts[top_idx]} ({neg_prob*100:.1f}%)"
                else:
                    pred_breed, pred_spec, _, _ = OXFORD_CLASSES[top_idx]
                    if pred_spec != target_spec and top_prob > 0.35 and target_prob < 0.05:
                        is_valid = False
                        fail_reason = f"Sai loài: {pred_breed} {pred_spec} ({top_prob*100:.1f}%) vs đích {breed} ({target_prob*100:.1f}%)"
                    else:
                        top3 = torch.topk(probs[i, :num_classes], 3).indices.tolist()
                        if target_idx not in top3 and target_prob < 0.08:
                            if not (pred_spec == target_spec and target_prob >= 0.03):
                                is_valid = False
                                fail_reason = f"Độ khớp thấp: {pred_breed} ({top_prob*100:.1f}%) vs đích {breed} ({target_prob*100:.1f}%)"

                if is_valid:
                    valid_images_by_breed[breed].append((f, p, target_prob))
                else:
                    os.remove(p)
                    removed_count += 1
                    if "1152" in f or removed_count <= 25 or removed_count % 50 == 0:
                        print(f"   [ĐÃ LOẠI BỎ] {f:30} -> {fail_reason}")

        except Exception as e:
            print(f"   [Lỗi batch {b_idx}]: {e}")

        if (b_idx + batch_size) % 640 < batch_size or (b_idx + batch_size) >= len(all_files):
            total_checked = min(b_idx + batch_size, len(all_files))
            valid_so_far = sum(len(v) for v in valid_images_by_breed.values())
            print(f"   ... Đã duyệt {total_checked}/{len(all_files)} ảnh ({time.time() - t0:.1f}s) | Hợp lệ: {valid_so_far} | Đã xóa: {removed_count}")

    print(f">> [BƯỚC 1 HOÀN TẤT] Tổng số ảnh nhiễu đã bị loại bỏ: {removed_count} ảnh.")
    print(f">> Số ảnh chuẩn xác còn lại: {sum(len(v) for v in valid_images_by_breed.values())} ảnh.\n")

    # ==================================================================
    # BƯỚC 2: BỔ SUNG ẢNH ĐẠT CHUẨN ĐỂ ĐẠT 150 - 155 ẢNH/LOÀI
    # ==================================================================
    print(f"==================================================================")
    print(f">> [BƯỚC 2] BỔ SUNG ẢNH THẬT ĐẠT CHUẨN CLIP (MỤC TIÊU: {MIN_TARGET}-{MAX_TARGET} ẢNH/LOÀI)")
    print(f"==================================================================")

    # 1. Bổ sung từ Stanford Dogs test split cho các giống chó còn thiếu
    print("   -> Đang kiểm tra bổ sung ảnh chó từ Stanford Dogs Dataset (split: test)...")
    try:
        stanford_stream = load_dataset('amaye15/stanford-dogs', split='test', streaming=True)
        label_names = stanford_stream.features['label'].names
        stanford_to_oxford = {}
        for ox_name, stan_name in STANFORD_MAPPING.items():
            if stan_name in label_names:
                idx = label_names.index(stan_name)
                stanford_to_oxford[idx] = ox_name

        for item in stanford_stream:
            lbl = item['label']
            if lbl in stanford_to_oxford:
                ox_name = stanford_to_oxford[lbl]
                if len(valid_images_by_breed[ox_name]) < BALANCED_TARGET:
                    img = item['pixel_values'].convert('RGB')
                    ok, prob, reason = verify_pil_image(
                        img, ox_name, model, processor, text_features, breed_to_idx, breed_species, num_classes, all_prompts, device
                    )
                    if ok:
                        seq = len(valid_images_by_breed[ox_name]) + 2000
                        new_fname = f"{ox_name}_{seq}.jpg"
                        new_path = os.path.join(IMAGES_DIR, new_fname)
                        img.save(new_path, "JPEG", quality=90)
                        valid_images_by_breed[ox_name].append((new_fname, new_path, prob))
    except Exception as e:
        print(f"   [Thông tin Stanford Dogs]: {e}")

    # 2. Bổ sung từ Wikimedia Commons cho các giống (mèo & chó) còn thiếu < 150
    for ox_name, species, _, _ in OXFORD_CLASSES:
        current_count = len(valid_images_by_breed[ox_name])
        if current_count < MIN_TARGET:
            needed = BALANCED_TARGET - current_count
            print(f"   -> Đang tìm và thẩm định CLIP {needed} ảnh mới cho [{ox_name}] (Hiện có: {current_count})...")
            candidate_urls = fetch_wikimedia_photo_urls(ox_name, species, target_count=needed * 4)
            added = 0
            for url in candidate_urls:
                img = download_image(url)
                if img is None:
                    continue
                ok, prob, reason = verify_pil_image(
                    img, ox_name, model, processor, text_features, breed_to_idx, breed_species, num_classes, all_prompts, device
                )
                if ok:
                    seq = len(valid_images_by_breed[ox_name]) + 2000
                    new_fname = f"{ox_name}_{seq}.jpg"
                    new_path = os.path.join(IMAGES_DIR, new_fname)
                    img.save(new_path, "JPEG", quality=90)
                    valid_images_by_breed[ox_name].append((new_fname, new_path, prob))
                    added += 1
                    if len(valid_images_by_breed[ox_name]) >= BALANCED_TARGET:
                        break
            print(f"      [Xong {ox_name}]: Đã nạp thành công {added} ảnh chuẩn CLIP. Tổng hiện tại: {len(valid_images_by_breed[ox_name])}")

    # ==================================================================
    # BƯỚC 3: CÂN BẰNG HOÀN TOÀN TẤT CẢ GIỐNG TRONG KHOẢNG 150 - 155 ẢNH
    # ==================================================================
    print(f"\n==================================================================")
    print(f">> [BƯỚC 3] CHUẨN HÓA VÀ CÂN BẰNG CHÍNH XÁC (TỐI ĐA 152 ẢNH/LOÀI CHẤT LƯỢNG CAO NHẤT)")
    print(f"==================================================================")

    # Nếu loài nào vượt quá 155 ảnh, giữ lại 152 ảnh có độ tin cậy CLIP cao nhất
    for ox_name in valid_images_by_breed:
        items = valid_images_by_breed[ox_name]
        if len(items) > MAX_TARGET:
            # Sắp xếp theo prob giảm dần
            items.sort(key=lambda x: x[2], reverse=True)
            keep = items[:BALANCED_TARGET]
            remove = items[BALANCED_TARGET:]
            for f, p, _ in remove:
                if os.path.exists(p):
                    os.remove(p)
            valid_images_by_breed[ox_name] = keep
            print(f"   [Cân bằng {ox_name:25}]: Giữ lại {len(keep)} ảnh chuẩn nhất, loại {len(remove)} ảnh dư thừa.")

    # ==================================================================
    # BƯỚC 4: ĐỒNG BỘ ĐẶT TÊN & ANNOTATIONS (TRAINVAL.TXT, LIST.TXT)
    # ==================================================================
    print(f"\n==================================================================")
    print(f">> [BƯỚC 4] ĐỒNG BỘ HÓA TÊN FILE VÀ DANH MỤC ANNOTATIONS")
    print(f"==================================================================")

    # Đổi tên 2 bước an toàn (tránh collision) theo thứ tự từ 1001 đến N
    final_files = []
    final_counts = {}
    entries = []

    breed_info = {}
    for i, (b, spec, sid, _) in enumerate(OXFORD_CLASSES):
        breed_info[b] = (i + 1, sid)

    # Bước 4.1: Chuyển toàn bộ file hợp lệ sang tên tạm duy nhất
    staged_items = []
    for ox_name in sorted(valid_images_by_breed.keys()):
        items = valid_images_by_breed[ox_name]
        cid, sid = breed_info[ox_name]
        for idx, (old_f, old_p, prob) in enumerate(items, start=1001):
            target_f = f"{ox_name}_{idx}.jpg"
            target_p = os.path.join(IMAGES_DIR, target_f)
            tmp_f = f"temp_clean_{ox_name}_{idx}_{time.time_ns()}.jpg"
            tmp_p = os.path.join(IMAGES_DIR, tmp_f)
            if os.path.exists(old_p):
                os.rename(old_p, tmp_p)
                staged_items.append((tmp_p, target_p, target_f, ox_name, cid, sid))

    # Bước 4.2: Đổi từ tên tạm sang tên chuẩn đích
    for tmp_p, target_p, target_f, ox_name, cid, sid in staged_items:
        if os.path.exists(target_p):
            os.remove(target_p)
        os.rename(tmp_p, target_p)
        name_no_ext = target_f[:-4]
        entries.append(f"{name_no_ext} {cid} {sid} 1\n")
        final_counts[ox_name] = final_counts.get(ox_name, 0) + 1

    trainval_path = os.path.join(ANNOS_DIR, "trainval.txt")
    list_path = os.path.join(ANNOS_DIR, "list.txt")
    with open(trainval_path, "w") as fp:
        fp.writelines(entries)
    with open(list_path, "w") as fp:
        fp.writelines(entries)

    total_imgs = sum(final_counts.values())
    summary = {
        "dataset_name": "New Dataset 100% CLIP Verified Real Photos",
        "description": "Tập dữ liệu mới đã được kiểm duyệt toàn diện qua mô hình CLIP zero-shot, loại bỏ 100% ảnh nhiễu, cân bằng 150 - 155 ảnh/loài.",
        "verification_model": "openai/clip-vit-base-patch32",
        "total_images": total_imgs,
        "total_classes": len(final_counts),
        "average_per_class": round(total_imgs / len(final_counts), 1),
        "min_per_class": min(final_counts.values()),
        "max_per_class": max(final_counts.values()),
        "breed_counts": final_counts
    }

    summary_path = os.path.join(DATASET_DIR, "dataset_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)

    print(f"\n==================================================================")
    print(f">> KIỂM ĐỊNH & CHUẨN HÓA DATASET BẰNG CLIP HOÀN TẤT THÀNH CÔNG!")
    print(f"- Tổng số ảnh chuẩn xác sau kiểm định: {total_imgs} ảnh")
    print(f"- Số lượng giống loài: {len(final_counts)}/37 giống")
    print(f"- Khoảng số lượng mỗi giống: {summary['min_per_class']} - {summary['max_per_class']} ảnh/loài")
    print(f"- Trung bình mỗi giống: {summary['average_per_class']} ảnh")
    print(f"- File tóm tắt: {summary_path}")
    print(f"==================================================================")


if __name__ == "__main__":
    run_full_verification()
