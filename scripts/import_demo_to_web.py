"""Tự động chạy và nạp ảnh mẫu vào hệ thống Web với TOÀN BỘ 5 MÔ HÌNH (Batch Mode).

Các mô hình được chạy đồng thời cho mỗi bức ảnh:
1. Dual-Stream EfficientNet-B0 (mô hình tối ưu, hỗ trợ phát hiện giống lạ)
2. EfficientNet-B0
3. DenseNet-121
4. SE-ResNet-18
5. ResNet-18

Mỗi bức ảnh được gán chung một batch_id để:
- Người dùng khi nhấn vào xem ảnh có thể chọn từng mô hình để so sánh kết quả.
- Bảng so sánh BatchComparisonTable hiển thị đồng thuận/bất đồng giữa 5 mô hình.
- Phân loại OOD hiển thị trạng thái 'Chưa xác định'.
"""

from __future__ import annotations

import shutil
import sys
import time
import uuid
from pathlib import Path

from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.ai.base import ClassifierResult  # type: ignore # noqa: E402
from app.ai.registry import ClassifierRegistry, ModelMethod  # type: ignore # noqa: E402
from app.core.database import SessionLocal  # type: ignore # noqa: E402
from app.models.prediction import Prediction, PredictionStatus  # type: ignore # noqa: E402
from app.repositories import prediction as prediction_repo  # type: ignore # noqa: E402

DEMO_DIR = REPO_ROOT / "demo_samples"
UPLOAD_DIR = REPO_ROOT / "backend" / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MODEL_METHODS = [
    ModelMethod.DUAL_STREAM_EFFICIENTNET,
    ModelMethod.EFFICIENTNET_B0,
    ModelMethod.DENSENET121,
    ModelMethod.SERESNET18,
    ModelMethod.RESNET18,
]


def collect_demo_images() -> list[tuple[str, Path]]:
    """Thu thập danh sách ảnh mẫu demo đại diện:
    - 24 ảnh giống chó ngoài danh mục (bao gồm chó Việt Nam: Phú Quốc, H'mông cộc, Bắc Hà, Lài).
    - 10 ảnh đồ vật & động vật hoang dã ngoài phạm vi thú cưng.
    - 74 ảnh thuộc 37 giống đã học (2 ảnh tiêu biểu mỗi giống).
    Tổng cộng: 108 ảnh x 5 mô hình = 540 lượt dự đoán.
    """
    items: list[tuple[str, Path]] = []

    # 1. Toàn bộ 24 ảnh giống chó lạ (bao gồm các giống chó Việt Nam)
    unseen_dir = DEMO_DIR / "unknown_and_non_pet" / "unseen_dog_breeds"
    if unseen_dir.exists():
        for p in sorted(unseen_dir.glob("*.jpg")):
            items.append(("unseen_dog_breed", p))

    # 2. Toàn bộ 10 ảnh vật thể & động vật hoang dã
    non_pet_dir = DEMO_DIR / "unknown_and_non_pet" / "non_pet_objects"
    if non_pet_dir.exists():
        for p in sorted(non_pet_dir.glob("*.jpg")):
            items.append(("non_pet_object", p))

    # 3. 37 Giống thú cưng đã biết (lấy 2 ảnh tiêu biểu mỗi giống để tối ưu thời gian chạy 5 mô hình)
    known_dir = DEMO_DIR / "known_breeds"
    if known_dir.exists():
        for breed_dir in sorted(known_dir.iterdir()):
            if breed_dir.is_dir():
                samples = sorted(breed_dir.glob("*.jpg"))[:2]
                for p in samples:
                    items.append((f"known_{breed_dir.name}", p))

    return items


def get_demo_note(category: str, filename: str) -> str:
    fn = filename.lower()
    # 1. Chó Việt Nam ngoài danh mục
    if "phu_quoc" in fn:
        return "Chó Phú Quốc (Giống VN ngoài danh mục)"
    if "hmong" in fn:
        return "Chó H'mông cộc đuôi (Giống VN ngoài danh mục)"
    if "bac_ha" in fn:
        return "Chó Bắc Hà (Giống VN ngoài danh mục)"
    if "indochina_dingo" in fn or "dingo" in fn:
        return "Chó Lài / Dingo Đông Dương (Giống VN ngoài danh mục)"

    # 2. Các giống chó quốc tế ngoài danh mục 37 giống
    if "golden_retriever" in fn:
        return "Golden Retriever (Giống chó ngoài danh mục)"
    if "german_shepherd" in fn:
        return "German Shepherd (Giống chó ngoài danh mục)"
    if "doberman" in fn:
        return "Doberman (Giống chó ngoài danh mục)"
    if "french_bulldog" in fn:
        return "French Bulldog (Giống chó ngoài danh mục)"
    if "rottweiler" in fn:
        return "Rottweiler (Giống chó ngoài danh mục)"
    if "siberian_husky" in fn or "husky" in fn:
        return "Siberian Husky (Giống chó ngoài danh mục)"

    # 3. Vật thể & động vật hoang dã ngoài danh mục thú cưng
    non_pet_map = {
        "airplane": "Máy bay (Vật thể ngoài danh mục)",
        "sports_car": "Ô tô thể thao (Vật thể ngoài danh mục)",
        "bicycle": "Xe đạp (Vật thể ngoài danh mục)",
        "guitar": "Đàn guitar (Vật thể ngoài danh mục)",
        "landscape": "Phong cảnh thiên nhiên (Ngoài danh mục)",
        "bear": "Gấu hoang dã (Động vật ngoài danh mục)",
        "elephant": "Voi hoang dã (Động vật ngoài danh mục)",
        "lion": "Sư tử (Động vật ngoài danh mục)",
        "tiger": "Hổ (Động vật ngoài danh mục)",
        "eagle": "Đại bàng (Động vật ngoài danh mục)",
    }
    for k, v in non_pet_map.items():
        if k in fn:
            return v

    # 4. 37 giống đã học (Oxford-IIIT Pet)
    if category.startswith("known_"):
        raw_name = category.replace("known_", "").replace("_", " ").title()
        return f"{raw_name} (Thuộc 37 giống chuẩn)"

    return category.replace("_", " ").title()


def main() -> None:
    items = collect_demo_images()
    print(f"Tổng số ảnh được chọn để chạy toàn bộ 5 mô hình: {len(items)} ảnh.")

    print("\nĐang khởi tạo toàn bộ 5 mô hình phân loại...")
    loaded_models = []
    for m in MODEL_METHODS:
        print(f" - Tải mô hình: {m.value}")
        clf = ClassifierRegistry.get(m)
        loaded_models.append((m, clf))
    print("Toàn bộ 5 mô hình đã sẵn sàng.\n")

    db = SessionLocal()
    try:
        # Làm sạch các bản ghi dự đoán cũ để chuẩn hóa cấu trúc batch 5 mô hình
        deleted_count = db.query(Prediction).delete()
        db.commit()
        print(f"Đã làm sạch {deleted_count} bản ghi cũ trong cơ sở dữ liệu.")

        total_predictions = 0
        total_images = len(items)

        for category, img_path in tqdm(items, desc="Chạy 5 mô hình/ảnh"):
            # Sao chép ảnh vào thư mục upload tĩnh
            upload_name = f"batch_{category}_{img_path.name}"
            dest_path = UPLOAD_DIR / upload_name
            if not dest_path.exists():
                shutil.copy2(img_path, dest_path)

            image_url = f"http://localhost:8000/static/uploads/{upload_name}"

            # Đọc ảnh
            try:
                with Image.open(img_path) as opened:
                    pil_img = opened.convert("RGB")
            except Exception as e:
                print(f"Lỗi mở ảnh {img_path.name}: {e}")
                continue

            batch_id = uuid.uuid4()
            demo_note = get_demo_note(category, img_path.name)

            # Chạy tuần tự qua cả 5 mô hình
            for method, clf in loaded_models:
                t0 = time.perf_counter()
                result: ClassifierResult = clf.predict(pil_img)
                proc_time = time.perf_counter() - t0

                top_k_dicts = [item.model_dump() for item in result.top_k_results]
                if result.unknown_assessment:
                    top_k_dicts.append({"_unknown_assessment": result.unknown_assessment})

                pred = prediction_repo.create_prediction(
                    db,
                    image_key=image_url,
                    method=method.value,
                    batch_id=batch_id,
                    note=demo_note,
                )
                prediction_repo.update_prediction_status(
                    db,
                    pred,
                    status=PredictionStatus.COMPLETED,
                    predicted_breed_id=result.predicted_breed_id,
                    score=result.score,
                    top_k_results=top_k_dicts,
                    model_version=result.model_version,
                    processing_time=proc_time,
                )
                total_predictions += 1

            # Commit sau mỗi ảnh (lưu 5 bản ghi)
            db.commit()

        print("\n" + "=" * 65)
        print("HOÀN TẤT NẠP ẢNH BATCH TOÀN BỘ 5 MÔ HÌNH VÀO WEB")
        print("=" * 65)
        print(f"- Tổng số bức ảnh đã chạy: {total_images} ảnh")
        print(f"- Tổng số kết quả dự đoán tạo lập: {total_predictions} lượt (5 mô hình/ảnh)")
        print("- Xem kết quả so sánh và chọn mô hình tại: http://localhost:5173/history")

    finally:
        db.close()


if __name__ == "__main__":
    main()
