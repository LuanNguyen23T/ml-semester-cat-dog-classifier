---
name: pet-classifier
description: >-
  Pipeline for Oxford-IIIT Pet classification (37 breeds and binary Cat vs Dog) trained from scratch.
  Includes part-level data augmentation (ears, tail, paws), CLAHE/Bilateral texture filtering,
  10 scratch architectures (Custom CNN, AlexNet, VGG-11, ResNet-18, ResNet-34, ResNet-50, DenseNet-121, MobileNetV2, SENet, EfficientNet-B0),
  dedicated notebooks per model, 50-epoch training with per-epoch checkpointing and resume,
  and publication-ready visualization generation.
---

# Pet Classifier: Oxford-IIIT Pet Pipeline from Scratch

Tài liệu hướng dẫn và quy chuẩn toàn diện cho hệ thống phân loại chó mèo với tập dữ liệu **Oxford-IIIT Pet**, đáp ứng yêu cầu nhận diện vật nuôi theo từng bộ phận (tai, chân, đuôi, thân), khởi tạo mô hình **From Scratch**, cấu trúc thư mục gọn gàng, tách biệt từng model thành notebook riêng và cơ chế lưu checkpoint / tự động resume sau mỗi epoch.

---

## 1. Cấu Trúc Dự Án (Workspace Architecture)

Toàn bộ mã nguồn thực thi được tổ chức trong thư mục `src/`, loại bỏ toàn bộ các file `.py` rác ở thư mục gốc:

```text
ml-semester-cat-dog-classifier/
├── .agent/
│   └── skills/
│       └── pet-classifier/
│           └── SKILL.md                 <-- Skill duy nhất chứa toàn bộ tri thức dự án
├── .gitignore                           <-- Bỏ qua checkpoint nặng, zip, dataset images thô
├── README.md                            <-- Hướng dẫn tổng quan
├── dataset/
│   ├── oxford-iiit-pet.zip              <-- Bản nén zip lưu trữ (779 MB)
│   └── oxford-iiit-pet/                 <-- Bản gốc đầy đủ (7,349 ảnh + annotations)
│       ├── annotations/
│       └── images/
├── src/                                 <-- Gói mã nguồn dùng chung
│   ├── __init__.py
│   ├── dataset.py                       <-- Dataset loader, Part-Crop, CLAHE, Bilateral
│   ├── models.py                        <-- 10 kiến trúc Deep CNN from scratch
│   └── train.py                         <-- Engine huấn luyện, checkpoint, resume, vẽ đồ thị
└── notebooks/                           <-- Thư mục chứa 10 folder độc lập cho 10 mô hình
    ├── 01_custom_cnn/
    │   ├── 01_custom_cnn.ipynb
    │   ├── checkpoints/                 <-- Lưu checkpoint riêng của Custom CNN
    │   └── plots/                       <-- Lưu biểu đồ riêng của Custom CNN
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
    └── 10_efficientnet/
        ├── 10_efficientnet.ipynb
        ├── checkpoints/
        └── plots/
```

---

## 2. 10 Kiến Trúc Deep CNN From Scratch (Tham Chiếu `R&D/CNN.txt`)
Tất cả các mô hình được thiết lập `pretrained=False`, khởi tạo trọng số ngẫu nhiên theo **Kaiming He Normal**:
1. `custom_cnn`: 6-Layer Baseline ConvNet (Conv-BN-ReLU-MaxPool + GAP + MLP, ~427k params).
2. `alexnet`: AlexNet hiện đại hóa với Batch Normalization sau mỗi Conv (~23.4M params).
3. `vgg11_bn`: VGG-11 với cấu trúc tích chập 3x3 kèm BatchNorm (~128.9M params).
4. `resnet18`: ResNet-18 mạng liên kết tàn dư (Residual Skip Connections, ~11.1M params).
5. `resnet34`: ResNet-34 phiên bản sâu hơn (~21.3M params).
6. `resnet50`: ResNet-50 cấu trúc Bottleneck 3 tầng (1x1 -> 3x3 -> 1x1, ~23.6M params).
7. `densenet121`: DenseNet-121 kết nối dày đặc ghép nối đặc trưng đa tầng (~6.9M params).
8. `mobilenet_v2`: MobileNetV2 khối Inverted Residual & Linear Bottlenecks tối ưu biên (~2.2M params).
9. `seresnet18`: SENet: ResNet-18 tích hợp Squeeze-and-Excitation Channel Attention (~11.2M params).
10. `efficientnet`: EfficientNet-B0 chuẩn Google Research (Tan & Le, 2019 Table 1) MBConv + SE (~4.0M params).

---

## 3. Chiến Lược Tăng Cường Dữ Liệu
### A. Yêu Cầu Của Giáo Viên: Nhận Diện Qua Từng Bộ Phận (Tai, Chân, Đuôi, Thân)
- **Extreme Part-Level Crop (`PartCropTransform`):** 50% số ảnh trong tập huấn luyện được crop ngẫu nhiên chỉ từ **8% đến 40% diện tích**, ép mô hình học cách phân loại giống loài khi chỉ nhìn thấy một góc của tai, chân/móng, đuôi hoặc mảng lông.
- **Random Erasing (Cutout):** Che ngẫu nhiên một phần lớn khuôn mặt hoặc thân thể để mô hình không phụ thuộc vào một đặc trưng đơn lẻ.

### B. Tái Sử Dụng & Nâng Cấp từ `CLIP-Animal-Classifier`
- **CLAHE trên kênh L (LAB Space):** Tăng cường tương phản thích ứng cục bộ, làm nổi bật kết cấu vân lông và viền mắt.
- **Bilateral Filtering:** Làm mịn nhiễu nền nhưng giữ độ sắc cạnh của sợi lông và ria mép.

---

## 4. Quy Trình 6 Bước Chuẩn Trong Từng Notebook
Mỗi notebook trong `notebooks/<tên_model>/` được chia làm 6 bước:
1. **Bước 1: Input các thư viện:** PyTorch, Torchvision, OpenCV, NumPy, Matplotlib, PIL. Tự động nhận diện `mps` (Apple Silicon), `cuda` hoặc `cpu`.
2. **Bước 2: Argparse:** Cấu hình tham số riêng của mô hình, trỏ output về `./checkpoints` và `./plots` nội bộ.
3. **Bước 3: Load Data và in ra thông tin:** Thống kê tập trainval (3,680) / test (3,669), trực quan hóa các bộ phận tai, chân, đuôi, thân.
4. **Bước 4: Chọn Model:** Khởi tạo kiến trúc from scratch tương ứng với thư mục.
5. **Bước 5: Huấn luyện 25 Epochs với Checkpoint & Resume:**
   - Lưu checkpoint sau mỗi epoch: `{model}_{task}_epoch_{epoch:02d}.pth`, `latest.pth`, `best.pth`.
   - Tự động nhận diện và resume từ epoch tiếp theo nếu bị dừng.
   - Định dạng log: `Epoch 01 | Train Loss: 1.951 | Train Acc: 29.82% | Test Loss: 1.560 | Test Acc: 40.99%`
6. **Bước 6: In ra biểu đồ & figures:** Xuất biểu đồ đường cong Loss/Acc, ma trận nhầm lẫn (Confusion Matrix) và ảnh dự đoán kiểm thử vào `./plots/`.

---

## 5. Tiêu Chuẩn Thống Nhất Cho Đối Sánh (Benchmark Standardization)
Để việc so sánh (đối sánh) giữa 10 mô hình hoàn toàn công bằng, khoa học và trực quan:
1. **Random Seed = 42:** Cố định seed cho `random`, `np.random`, `torch.manual_seed` để đảm bảo thứ tự xáo trộn dữ liệu của DataLoader là hoàn toàn như nhau trên cả 10 notebook.
2. **Kích thước ảnh & Batch Size:** Cố định `img_size = 224x224`, `batch_size = 64`.
3. **Số epoch & Tối ưu hóa:** Cố định `epochs = 50`, `lr = 5e-4`, `weight_decay = 1e-2` với thuật toán AdamW và CosineAnnealingLR + Label Smoothing 0.1.
4. **Mẫu hiển thị Bước 3:** Thống nhất cùng 1 mẫu ảnh (sample index = 10) để quan sát sự khác biệt của ảnh gốc so với các lát cắt bộ phận (tai, chân, đuôi, CLAHE).
5. **Thống nhất tỷ lệ biểu đồ Bước 6:**
   - Trục y của biểu đồ Accuracy được cố định từ **0% đến 100%** (`plt.ylim(0, 100)`), giúp đặt 10 biểu đồ cạnh nhau là thấy ngay tốc độ hội tụ và khoảng cách giữa các model.
   - Thang màu của Ma trận nhầm lẫn chuẩn hóa (Confusion Matrix) cố định `vmin=0.0, vmax=1.0`.
   - Mỗi notebook tự động xuất file `./plots/{model}_metrics.json` lưu trữ các chỉ số tốt nhất (Best Acc, Final Loss) để tổng hợp thành bảng đối sánh.

---

## 6. Lệnh Chạy Huấn Luyện CLI (Tùy Chọn)
```bash
# Huấn luyện ResNet-50
python3 src/train.py --model resnet50 --task breed --epochs 50 --batch_size 64

# Huấn luyện EfficientNet-B0
python3 src/train.py --model efficientnet --task breed --epochs 50 --batch_size 64

# Huấn luyện SENet (SEResNet-18)
python3 src/train.py --model seresnet18 --task breed --epochs 50 --batch_size 64
```
