# Hướng dẫn cài đặt & Khởi chạy (SETUP.md)

Tài liệu dành cho người mới, hướng dẫn cài đặt từ máy sạch để chạy hệ thống **Pet Breed Classifier** trên máy cá nhân.

---

## 1. Phần mềm cần cài sẵn

1. **Docker Desktop** (dùng để chạy PostgreSQL).
2. **Python >= 3.10** (khuyến nghị Python 3.12).
3. **Node.js >= 20.19** (đi kèm `npm`).
4. *(Tuỳ chọn trên macOS)*: **macOS 14+ với Apple Silicon** nếu muốn chạy tăng tốc **MPS**. Nếu chạy CPU hoặc CUDA, hệ thống tự động chọn theo cấu hình `AI_DEVICE` trong `.env`.

---

## 2. Khởi chạy PostgreSQL bằng Docker

> **Quan trọng:** Mở ứng dụng **Docker Desktop** trước và đợi đến khi Docker báo trạng thái sẵn sàng (*Engine running*) rồi mới chạy lệnh dưới đây.

Từ thư mục gốc của dự án:

```bash
docker compose up -d
```

Lệnh này khởi tạo container PostgreSQL 16 (xem tên container, database, user và cổng kết nối trong [docker-compose.yml](docker-compose.yml)).

---

## 3. Cài đặt Backend & Tạo file cấu hình `.env`

Mở terminal tại thư mục `backend/`:

```bash
cd backend

# Tạo môi trường ảo Python (.venv)
python3 -m venv .venv

# Kích hoạt môi trường ảo
source .venv/bin/activate      # macOS / Linux
# .\.venv\Scripts\Activate.ps1 # Windows PowerShell

# Cài đặt thư viện chạy Web Backend
pip install -r requirements.txt
```

*(Ghi chú: các notebook huấn luyện chi tiết 5 mô hình và dữ liệu thử nghiệm nằm tại thư mục [notebooks/](notebooks/).)*

### Tạo `backend/.env` từ file mẫu

```bash
cp .env.example .env
```

Mở file `backend/.env` và điền đầy đủ giá trị theo [backend/.env.example](backend/.env.example):
- Thông tin PostgreSQL (`POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_SERVER`, `POSTGRES_PORT`): điền khớp với cấu hình trong [docker-compose.yml](docker-compose.yml).
- Thông tin tài khoản **Cloudinary** của riêng bạn (`CLOUDINARY_CLOUD_NAME`, `CLOUDINARY_API_KEY`, `CLOUDINARY_API_SECRET`, `CLOUDINARY_URL`) lấy từ Cloudinary Dashboard.

> **Bảo mật:** Không bao giờ chia sẻ hoặc commit file `backend/.env` lên Git (file này đã được `.gitignore` bỏ qua).

---

## 4. Chuẩn bị file trọng số (Weights) cho 5 mô hình

Vì các file checkpoint `.pth` có dung lượng lớn nên bị `.gitignore` chặn và **không nằm sẵn trong Git repository**.

1. Đặt đủ **5 file checkpoint gốc** vào thư mục `backend/weights/`:
   - `resnet18_augmented_best.pth`
   - `seresnet18_augmented_best.pth`
   - `densenet121_augmented_best.pth`
   - `efficientnet_b0_augmented_best.pth`
   - `dual_stream_efficientnet_best.pth`
   *(Trong dự án này, các file trên được lấy tương ứng từ thư mục `notebooks/New Dataset /`)*.

2. Quay về **thư mục gốc của dự án** và chạy script trích xuất bản nhẹ `*_state_dict.pth` phục vụ web (script không làm thay đổi 5 file gốc):

```bash
python3 scripts/extract_state_dicts.py
```

Sau bước này, `backend/weights/` sẽ có thêm 5 file `*_state_dict.pth` mà backend sử dụng. *(Nếu thiếu file của model nào, server vẫn khởi động bình thường và đánh dấu model đó là "Không khả dụng" trên giao diện.)*

---

## 5. Khởi tạo bảng Database & Seed 37 giống chó mèo

Từ thư mục gốc của dự án:

```bash
# Chạy DB migrations
cd backend
.venv/bin/alembic upgrade head

# Quay lại thư mục gốc và nạp 2 loài (Cat, Dog) cùng 37 giống vào PostgreSQL
cd ..
backend/.venv/bin/python scripts/seed_breeds.py
```

Script `seed_breeds.py` tự lấy danh sách chuẩn 37 giống từ backend (không cần tải thư mục `dataset/`).

---

## 6. Khởi chạy Backend và Frontend

### Terminal 1 — Chạy Backend (FastAPI)

```bash
cd backend
.venv/bin/uvicorn app.main:app --reload --port 8000
```

> **Lưu ý ở lần chạy đầu tiên:** Ngay khi nhận request phân loại ảnh đầu tiên, backend sẽ tự động tải trọng số mô hình tách nền `DeepLabV3-MobileNetV3-Large` (**~42 MB**) từ PyTorch Hub về cache trên máy, vì vậy **lần chạy đầu cần có kết nối Internet**. Những lần sau dùng bản cache offline.

### Terminal 2 — Chạy Frontend (React Vite)

```bash
cd frontend
npm install
npm run dev
```

---

## 7. Kiểm tra hoạt động

- **Giao diện Web:** `http://localhost:5173`
- **Tài liệu API (Swagger UI):** `http://localhost:8000/docs`
- **Kiểm tra danh sách model:** `http://localhost:8000/api/v1/models`

*(Tuỳ chọn — kiểm tra nhanh độ chính xác 5 model với `torchvision`):*

```bash
backend/.venv/bin/python scripts/verify_new_models.py --num-samples 300 --seed 42
```
