# Các Quyết Định Kỹ Thuật Đã Chốt (`DECISIONS.md`)

Tài liệu này ghi lại các quyết định đã thống nhất để chuyển đổi web nhận diện 37 giống chó mèo sang 5 mô hình tự huấn luyện. Mọi thay đổi phát sinh trong từng giai đoạn triển khai phải được cập nhật tại đây.

---

## 1. Danh sách mô hình trên Web
- Web chỉ phục vụ **5 mô hình**:
  1. `ResNet18`
  2. `SEResNet18`
  3. `DenseNet121`
  4. `EfficientNet-B0`
  5. `Dual-Stream EfficientNet-B0` *(mô hình đề xuất chính, mặc định)*
- Toàn bộ code/weights/tài liệu của **CLIP** và **ResNet-50 pretrained cũ** sẽ được chuyển dần sang `archive/` trong quá trình triển khai, và chỉ **xoá hẳn** ở giai đoạn dọn dẹp cuối cùng sau khi người dùng xác nhận web chạy ổn.

## 2. Cơ sở dữ liệu & Seed
- **Reset sạch DB** (không giữ dữ liệu dự đoán cũ).
- **Giữ nguyên schema** 3 bảng hiện tại (`species`, `breeds`, `predictions` — không cần migration mới).
- Chạy PostgreSQL bằng Docker qua [docker-compose.yml](file:///Users/admin/Developer/Machine%20Learning/CLIP-Animal-Classifier/docker-compose.yml) (Docker Desktop).
- Sửa [scripts/seed_breeds.py](file:///Users/admin/Developer/Machine%20Learning/CLIP-Animal-Classifier/scripts/seed_breeds.py) để nạp lại 37 giống chó mèo trực tiếp mà **không phụ thuộc thư mục `dataset/`**.

## 3. Nạp Checkpoint & Kiến trúc Model
- Mọi model bắt buộc phải nạp weights bằng `load_state_dict(..., strict=True)`. Hoàn toàn không dùng `strict=False`.
- Phải dùng đúng class khớp với checkpoint `.pth`:
  - `SEResNet18` **phải dùng class `CustomSEResNet`** từ notebook `fg-seresnet18-newdataset-al.ipynb` (head tên `linear`), **không dùng** `SEResNet18` trong `my_training/src/models.py` (head tên `fc`).
  - `DualStreamEfficientNet` dùng đúng định nghĩa trong `fg16-dual-stream-efficientnet.ipynb` (`SqueezeExcitation` dùng `bias=True`).
  - `ResNet18`, `DenseNet121`, `EfficientNetB0` khớp với cả notebook lẫn `my_training/src/models.py`.
- **Giữ nguyên 5 file checkpoint gốc** (`*_best.pth`) để dự phòng; đồng thời tạo thêm bộ file weights nhẹ chỉ chứa `model_state_dict` phục vụ web.
- Đưa `backend/weights/`, `*.pt`, `*.pth` vào `.gitignore`.

## 4. Tiền xử lý, Ảnh đầu Dual-Stream & TTA lúc suy luận
- **Tách nền tự động thay Trimap (Hướng A — Cấu hình 3 + 4b)**:
  - Sử dụng `deeplabv3_mobilenet_v3_large` (sẵn trong `torchvision`, trọng số ~42 MB) lọc lớp `cat` (8) và `dog` (12), nạp **lazy-load một lần cho cả tiến trình** và có khóa thread-safe.
  - Tạo mask tiền cảnh → cắt bounding box (+4% padding mỗi chiều) và tô pixel nền thành màu xám `(128, 128, 128)`.
  - **Cơ chế an toàn (Fallback)**: nếu không tìm thấy pixel `cat`/`dog` hoặc việc nạp/chạy segmentation gặp lỗi, tự động dùng ảnh thô ban đầu để không làm hỏng request.
- **Tiền xử lý chuẩn sau khi cắt/nền xám**: `Resize((328, 328))` → `CenterCrop(288)` → `ToTensor()` → `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`. Không dùng CLAHE/Bilateral ở bước suy luận.
- **Luồng đầu của Dual-Stream (4b)**: dùng vùng thú cưng đã crop (+4% padding, **giữ nền gốc**, không tô xám) rồi cắt đầu theo quy tắc (`xmin = 0.10 * w`, `xmax = 0.90 * w`, `ymin = 0`, `ymax = 0.65 * h`), sau đó qua tiền xử lý chuẩn `288 × 288`. (Nếu segmentation fallback về ảnh thô thì cắt 65% trên ảnh thô).
- **Module hóa**: Toàn bộ logic đặt tập trung trong `backend/app/ai/preprocessing.py`.
- **4-View TTA** (4 view: Gốc, Lật ngang, Center Zoom `0.9x`, Lật ngang + Zoom `0.9x`; trọng số `0.40 / 0.30 / 0.15 / 0.15`):
  - **Mặc định TẮT** để ưu tiên tốc độ.
  - Có tham số bật/tắt trên API và công tắc **"Độ chính xác cao (TTA)"** trên giao diện.

## 5. Giao diện & Lưu trữ
- **Bỏ trang `PromptLog`** cùng các component/service/type mock LLM dư thừa trong frontend.
- **Giữ Cloudinary** để lưu ảnh upload. Khoá bí mật nằm trong `backend/.env`.
- Nghiêm ngặt không đọc, in ra hay commit file `.env`; đảm bảo `.env` và `.env.*` có trong `.gitignore`.

---

## Nhật ký thay đổi theo giai đoạn
- **2026-11-25 (Khởi tạo)**: Chốt các quyết định ban đầu dựa trên [AUDIT.md](file:///Users/admin/Developer/Machine%20Learning/CLIP-Animal-Classifier/docs/AUDIT.md) và thiết lập lại thư mục `.agent/`.
- **2026-11-25 (Giai đoạn 1)**: Phát hiện ảnh thô không tách nền làm Top-1 tụt xuống 7.33%–15.33% vì 5 model học phụ thuộc nền xám `(128,128,128)` của Trimap. Đã thử nghiệm Hướng A với `deeplabv3_mobilenet_v3_large` (đưa `EfficientNet-B0` lên 79.00%/82.33% và `Dual-Stream` lên 84.00%/85.00%) và tích hợp Cấu hình (3) + (4b) dạng lazy-load + fallback ảnh thô vào `backend/app/ai/preprocessing.py`.
- **2026-11-25 (Giai đoạn 2)**: Kết nối 5 model mới vào FastAPI, Registry (`threading.Lock` + báo `available: false` nếu thiếu weights), API (`/predictions` mặc định `dual-stream-efficientnet`, tham số `tta`), Batch (`/predictions/batch` chia sẻ kết quả segmentation DeepLabV3 một lần qua `PreprocessedInputs` dạng PIL để parity `0.00e+00` với đường đơn lẻ), ghi hậu tố `+tta` vào `Prediction.model_version` để không cần migration DB, giới hạn ảnh upload (`10 MB`, `8000 px`, kiểm tra MIME & Pillow trước Cloudinary), đổi Cloudinary folder sang `pet-breed-classifier/users/{user_id}`, viết lại `scripts/seed_breeds.py` không phụ thuộc `dataset/`, chuyển code/weights CLIP & ResNet-50 cũ sang `archive/`, và gỡ `transformers==5.17.0` khỏi `backend/requirements.txt`.
- **2026-11-25 (Giai đoạn 3)**: Cập nhật Frontend cho 5 model mới: xoá `BackboneType`/CoOp/PromptLog (`PromptLog` chuyển sang `archive/frontend_old/`), mặc định `dual-stream-efficientnet`, chia nhóm Hai luồng / Một luồng và vô hiệu hoá model `available=false` trong `ConfigPanel`, thêm công tắc **"Độ chính xác cao (TTA)"** mặc định tắt, thêm `BatchComparisonTable` so sánh 5 kết quả cạnh nhau (tô sáng đồng thuận/bất đồng và hiển thị trạng thái riêng từng dòng khi đang chạy hoặc lỗi, cuộn ngang trên mobile), đồng bộ lọc `method` + nhãn `TTA` ở `History`/`Statistics`, chuẩn hoá thông báo lỗi tiếng Việt (`formatApiError`), và cập nhật thương hiệu sang `Pet Breed Classifier — 37 giống chó và mèo`. Không thêm gói `npm` mới và không sửa backend.
- **2026-11-25 (Giai đoạn 4)**: Dọn dẹp cuối: xoá hẳn `archive/`, `notebooks/`, `R_D/`, `dataset/`, và 3 script cũ (`export_clip_weights.py`, `export_resnet50_weights.py`, `verify_clip_parity.py`); giữ lại `my_training/` và 3 script mới (`seed_breeds.py`, `extract_state_dicts.py`, `verify_new_models.py`); tách `opencv-python` khỏi `backend/requirements.txt` và tạo `my_training/requirements.txt` cho nhu cầu huấn luyện lại; viết lại [README.md](file:///Users/admin/Developer/Machine%20Learning/CLIP-Animal-Classifier/README.md) và [SETUP.md](file:///Users/admin/Developer/Machine%20Learning/CLIP-Animal-Classifier/SETUP.md).
