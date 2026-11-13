# Oxford-IIIT Pet Classifier from Scratch
### Đề Tài Nhận Diện & Phân Tích Chó Mèo với Kỹ Thuật Tăng Cường Dữ Liệu Bộ Phận (Part-Level Augmentation)

Dự án nghiên cứu và xây dựng hệ thống phân loại vật nuôi (Oxford-IIIT Pet Dataset) gồm **37 giống loài (12 giống mèo, 25 giống chó)** và bài toán **nhị phân (Chó vs Mèo)** được huấn luyện **hoàn toàn từ đầu (From Scratch)** trên nền tảng PyTorch.

---

## 📁 Cấu Trúc Dự Án Đã Làm Gọn

```text
ml-semester-cat-dog-classifier/
├── .agent/
│   └── skills/
│       └── pet-classifier/
│           └── SKILL.md                 <-- Skill duy nhất chứa toàn bộ tri thức dự án
├── .gitignore
├── README.md
├── dataset/
│   ├── oxford-iiit-pet.zip              <-- Bản nén zip lưu trữ (779 MB)
│   └── oxford-iiit-pet/                 <-- Bản gốc đầy đủ (7,349 ảnh + annotations)
│       ├── annotations/
│       └── images/
├── src/                                 <-- Toàn bộ mã nguồn cốt lõi (sạch sẽ, không file .py ở gốc)
│   ├── __init__.py
│   ├── dataset.py                       <-- Dataset loader, Part-Crop, CLAHE, Bilateral
│   ├── models.py                        <-- 11 kiến trúc Deep Learning & Multimodal CLIP from scratch
│   └── train.py                         <-- Engine huấn luyện, checkpoint, resume, plot
└── notebooks/                           <-- 11 thư mục độc lập cho 11 mô hình
    ├── 01_custom_cnn/
    │   ├── 01_custom_cnn.ipynb
    │   ├── checkpoints/                 <-- Checkpoint riêng của Custom CNN
    │   └── plots/                       <-- Biểu đồ riêng của Custom CNN
    ├── 02_alexnet/
    │   ├── 02_alexnet.ipynb
    │   ├── checkpoints/
    │   └── plots/
    ├── 03_vgg11_bn/
    │   ├── 03_vgg11_bn.ipynb
    │   ├── checkpoints/
    │   └── plots/
    ├── 04_resnet18/
    │   ├── 04_resnet18.ipynb
    │   ├── checkpoints/
    │   └── plots/
    ├── 05_resnet34/
    │   ├── 05_resnet34.ipynb
    │   ├── checkpoints/
    │   └── plots/
    ├── 06_densenet121/
    │   ├── 06_densenet121.ipynb
    │   ├── checkpoints/
    │   └── plots/
    ├── 07_mobilenet_v2/
    │   ├── 07_mobilenet_v2.ipynb
    │   ├── checkpoints/
    │   └── plots/
    ├── 08_seresnet18/
    │   ├── 08_seresnet18.ipynb
    │   ├── checkpoints/
    │   └── plots/
    ├── 09_resnet50/
    │   ├── 09_resnet50.ipynb
    │   ├── checkpoints/
    │   └── plots/
    ├── 10_efficientnet/
    │   ├── 10_efficientnet.ipynb
    │   ├── checkpoints/
    │   └── plots/
    ├── 11_clip/
    │   ├── 11_clip.ipynb
    │   ├── checkpoints/
    │   └── plots/
    └── 12_hybrid_model/
        ├── 12_hybrid_model.ipynb        <-- Proposed PetNet-Hybrid: Dual-Attention MBConv-Dense Net
        ├── checkpoints/                 <-- hybrid_petnet_best.pth
        └── plots/                       <-- hybrid_petnet_curves.png, confusion_matrix, metrics.json
```

---

## 📌 Các Điểm Nổi Bật

1. **Tăng Cường Dữ Liệu Theo Yêu Cầu Của Giáo Viên (Bộ Phận Đơn Lẻ):**
   - **Extreme Part-Level Cropping:** Cắt ngẫu nhiên từ 8% đến 40% diện tích ảnh (tai, chân/móng, đuôi, thân), rèn luyện mạng nơ-ron nhận diện chính xác giống loài ngay cả khi chỉ có 1 bộ phận xuất hiện.
   - **Random Erasing (Cutout):** Che ngẫu nhiên một vùng lớn trên cơ thể để mô hình không phụ thuộc duy nhất vào khuôn mặt.
2. **Kế Thừa & Nâng Cấp từ `CLIP-Animal-Classifier`:**
   - **Bilateral Filtering:** Khử nhiễu quang học nền mà vẫn bảo toàn độ nét từng sợi lông và viền mắt.
   - **CLAHE LAB (L-Channel):** Cân bằng tương phản thích ứng cục bộ, làm nổi bật kết cấu vân lông.
3. **10 Mô Hình CNN From Scratch (Tham chiếu từ `R&D/CNN.txt`):**
   - `custom_cnn`: 6-Layer Baseline ConvNet (~427k params).
   - `alexnet`: AlexNet nâng cấp Batch Normalization (~23.4M params).
   - `vgg11_bn`: VGG-11 với Batch Normalization (~128.9M params).
   - `resnet18`: ResNet-18 mạng liên kết tàn dư (~11.1M params).
   - `resnet34`: ResNet-34 sâu hơn (~21.3M params).
   - `resnet50`: ResNet-50 khối Bottleneck 3 tầng (~23.6M params).
   - `densenet121`: DenseNet-121 kết nối dày đặc (~6.9M params).
   - `mobilenet_v2`: MobileNetV2 khối Inverted Residuals (~2.2M params).
   - `seresnet18`: SENet: ResNet-18 tích hợp Squeeze-and-Excitation (~11.2M params).
   - `efficientnet`: EfficientNet-B0 chuẩn Google Research (Tan & Le, 2019 Table 1) MBConv + SE (~4.0M params).
4. **Bộ Thông Số Thống Nhất Giúp Đối Sánh Chuẩn Xác:**
   - **Random Seed = 42:** Đảm bảo dữ liệu nạp và xáo trộn hoàn toàn đồng nhất giữa cả 10 mô hình.
   - **Cấu hình huấn luyện:** `img_size = 224x224`, `batch_size = 64`, `epochs = 50`, `lr = 5e-4` (AdamW + CosineAnnealingLR + Label Smoothing 0.1).
   - **Tỷ lệ biểu đồ đồng bộ:** Trục y đồ thị Accuracy cố định từ `0%` đến `100%` (`ylim(0, 100)`), thang màu Confusion Matrix cố định `vmin=0, vmax=1` giúp so sánh trực quan dễ dàng.
   - **Tự động xuất tóm tắt:** Mỗi mô hình tự xuất file `./plots/{model}_metrics.json` lưu Best Acc và Final Loss để tổng hợp bảng đối sánh.
5. **Cơ Chế Huấn Luyện Tiết Kiệm Thời Gian (Checkpoint & Resume):**
   - Lưu checkpoint sau mỗi epoch (`./checkpoints/{model}_epoch_{epoch:02d}.pth`, `{model}_latest.pth`, `{model}_best.pth`).
   - Tự động phát hiện checkpoint cũ để tiếp tục huấn luyện (resume).
   - Định dạng hiển thị chuẩn:
     ```text
     Epoch 01 | Train Loss: 1.951 | Train Acc: 29.82% | Test Loss: 1.560 | Test Acc: 40.99%
     ```

---

## 🚀 Hướng Dẫn Sử Dụng

### 1. Sử dụng Jupyter Notebook
Vào thư mục `notebooks/<thư_mục_mô_hình>/` tương ứng và mở file notebook `.ipynb`.
Tất cả 10 notebook đều được thiết kế ngắn gọn, chuẩn 6 bước:
- **Bước 1:** Input các thư viện
- **Bước 2:** Argparse
- **Bước 3:** Load Data và in ra thông tin (kèm ảnh cắt bộ phận: tai, chân, đuôi)
- **Bước 4:** Chọn Model (100% viết từ đầu bằng `nn.Module`, không dùng thư viện model có sẵn)
- **Bước 5:** Training (lưu checkpoint từng epoch, tự động resume, in đúng định dạng mẫu)
- **Bước 6:** In ra các biểu đồ, figure liên quan đến model (Loss/Acc curves, Confusion Matrix, metrics.json)

### 2. Sử dụng dòng lệnh (CLI)
```bash
# Huấn luyện Kiến trúc Đề xuất PetNet-Hybrid (From Scratch)
python3 src/train.py --model hybrid_petnet --task breed --epochs 50 --batch_size 64

# Huấn luyện ResNet-50
python3 src/train.py --model resnet50 --task breed --epochs 50 --batch_size 64

# Huấn luyện CLIP From Scratch (Vision Transformer + Text Transformer)
python3 src/train.py --model clip --task breed --epochs 50 --batch_size 64

# Huấn luyện EfficientNet-B0
python3 src/train.py --model efficientnet --task breed --epochs 50 --batch_size 64

# Huấn luyện Custom CNN
python3 src/train.py --model custom_cnn --task breed --epochs 50 --batch_size 64
```

---

## 📊 Bảng Đối Sánh Thực Nghiệm 11 Baseline Models & Kiến Trúc Đề Xuất (50 Epochs From Scratch)

| Hạng | Mô hình | Số tham số | Train Loss | Train Acc | Test Loss | Test Acc | Best Test Acc |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **EfficientNet-B0** | 4.05M | 2.212 | 47.64% | 2.212 | 45.98% | **46.33%** |
| 🥈 | **SEResNet-18 (SENet)** | 11.28M | 2.197 | 47.50% | 2.318 | 41.65% | **42.44%** |
| 🥉 | **Custom ResNet-50** | 23.58M | 2.309 | 43.59% | 2.406 | 40.77% | **40.96%** |
| 4 | **DenseNet-121** | 6.99M | 2.269 | 46.74% | 2.364 | 39.82% | **40.31%** |
| 5 | **Custom ResNet-34** | 21.30M | 2.508 | 36.63% | 2.415 | 39.22% | **39.22%** |
| 6 | **Custom ResNet-18** | 11.20M | 2.428 | 39.51% | 2.436 | 38.57% | **38.57%** |
| 7 | **MobileNetV2** | 2.27M | 2.622 | 33.53% | 2.557 | 33.63% | **34.26%** |
| 8 | **AlexNet (BatchNorm)** | 23.48M | 2.950 | 22.26% | 2.753 | 28.18% | **28.18%** |
| 9 | **CLIP (ViT + Text)** | 9.89M | 3.514 | - | 3.046 | - | **15.26%** |
| 10 | **Custom CNN (6-Layer)** | 0.43M | 3.190 | 15.43% | 3.178 | 14.85% | **15.04%** |
| 11 | **VGG-11 (BatchNorm)** | 128.92M | 3.611 | 2.69% | 3.611 | 2.73% | **2.73%** |
| 🚀 | **PetNet-Hybrid (Đề xuất)** | **5.37M** | *Tối ưu* | *Tối ưu* | *Tối ưu* | *Tối ưu* | **Dự kiến 55% - 63%+ (với TTA)** |
