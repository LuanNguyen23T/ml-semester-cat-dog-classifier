# Báo Cáo Thực Nghiệm: Cross-Dataset Zero-Shot Benchmark Toàn Diện Trên Stanford Dogs

## 1. Mục Đích & Phương Pháp Thử Nghiệm

Thử nghiệm **Cross-Dataset Benchmark** (Cấp độ 2) nhằm kiểm chứng năng lực tổng quát hóa (Generalization) và khả năng kháng dịch chuyển miền dữ liệu (Domain Shift Robustness) của 5 mô hình CNN được huấn luyện từ đầu (Train from Scratch trên Oxford-IIIT Pet):
- **Nguyên tắc thử nghiệm:** Lấy trực tiếp trọng số từ checkpoint tối ưu nhất (`*_state_dict.pth`) của 5 mô hình để suy luận trực tiếp (**Zero-Shot**, không huấn luyện lại, không fine-tuning, không cập nhật bất kỳ tham số nào trên Stanford Dogs).
- **Tập kiểm thử:** Lọc ra **21 giống chó giao nhau** giữa Oxford-IIIT Pet (37 lớp) và Stanford Dogs (120 lớp).
- **Quy mô dữ liệu đánh giá:** **Toàn bộ 3.730 ảnh** (kết hợp cả hai phân tập chính thức `train` gồm 2.100 ảnh và `test` gồm 1.630 ảnh) của toàn bộ 21 giống chó giao nhau, đảm bảo tính đại diện thống kê đầy đủ nhất và loại bỏ hoàn toàn sai số chọn mẫu ngẫu nhiên.
- **Chỉ số đánh giá cốt lõi:**
  - **Top-1 Accuracy & Top-5 Accuracy (%)** trên toàn bộ 3.730 ảnh Stanford Dogs.
  - **Mức suy giảm hiệu năng ($\Delta_{drop}$):**
    $$\Delta_{drop} = \text{Top-1}_{\text{Oxford}} - \text{Top-1}_{\text{Stanford Dogs}}$$
  - **Tiêu chuẩn kiểm định khoa học:** Nếu $\Delta_{drop} < 10\%$, mô hình chứng minh đã học được các biểu diễn hình thái giải phẫu sinh học thực chất của loài vật thay vì học vẹt bối cảnh nền xung quanh.

---

## 2. Bảng Thống Kê Số Lượng Ảnh Của 21 Giống Chó Giao Nhau

| STT | Mã Lớp Oxford | Tên Giống (Oxford-IIIT Pet) | Mã Nhãn Stanford | Tên Nhãn (Stanford Dogs) | Số Ảnh Train | Số Ảnh Test | Tổng Số Ảnh Đánh Giá |
| :---: | :---: | :--- | :---: | :--- | :---: | :---: | :---: |
| 1 | 4 | `basset_hound` | 10 | `n02088238-basset` | 100 | 75 | **175** |
| 2 | 5 | `beagle` | 11 | `n02088364-beagle` | 100 | 95 | **195** |
| 3 | 9 | `boxer` | 91 | `n02108089-boxer` | 100 | 51 | **151** |
| 4 | 11 | `chihuahua` | 0 | `n02085620-Chihuahua` | 100 | 52 | **152** |
| 5 | 13 | `english_cocker_spaniel` | 68 | `n02102318-cocker_spaniel` | 100 | 59 | **159** |
| 6 | 14 | `english_setter` | 61 | `n02100735-English_setter` | 100 | 61 | **161** |
| 7 | 15 | `german_shorthaired` | 59 | `n02100236-German_short-haired_pointer` | 100 | 52 | **152** |
| 8 | 16 | `great_pyrenees` | 105 | `n02111500-Great_Pyrenees` | 100 | 113 | **213** |
| 9 | 18 | `japanese_chin` | 1 | `n02085782-Japanese_spaniel` | 100 | 85 | **185** |
| 10 | 19 | `keeshond` | 109 | `n02112350-keeshond` | 100 | 58 | **158** |
| 11 | 20 | `leonberger` | 103 | `n02111129-Leonberg` | 100 | 110 | **210** |
| 12 | 22 | `miniature_pinscher` | 86 | `n02107312-miniature_pinscher` | 100 | 84 | **184** |
| 13 | 23 | `newfoundland` | 104 | `n02111277-Newfoundland` | 100 | 95 | **195** |
| 14 | 25 | `pomeranian` | 107 | `n02112018-Pomeranian` | 100 | 119 | **219** |
| 15 | 26 | `pug` | 102 | `n02110958-pug` | 100 | 100 | **200** |
| 16 | 29 | `saint_bernard` | 96 | `n02109525-Saint_Bernard` | 100 | 70 | **170** |
| 17 | 30 | `samoyed` | 106 | `n02111889-Samoyed` | 100 | 118 | **218** |
| 18 | 31 | `scottish_terrier` | 48 | `n02097298-Scotch_terrier` | 100 | 58 | **158** |
| 19 | 35 | `staffordshire_bull_terrier` | 28 | `n02093256-Staffordshire_bullterrier` | 100 | 55 | **155** |
| 20 | 36 | `wheaten_terrier` | 51 | `n02098105-soft-coated_wheaten_terrier` | 100 | 56 | **156** |
| 21 | 37 | `yorkshire_terrier` | 36 | `n02094433-Yorkshire_terrier` | 100 | 64 | **164** |
| **Tổng** | | **21 Giống Giao Nhau** | | | **2.100** | **1.630** | **3.730** |

---

## 3. Bảng Kết Quả Thực Nghiệm Toàn Diện (N = 3.730 Ảnh)

Dưới đây là kết quả đánh giá thực tế trên toàn bộ 3.730 ảnh của 21 giống chó giao nhau từ Stanford Dogs:

| Kiến Trúc Mô Hình | Số Tham Số (#Params) | Oxford Top-1 (Test Gốc) | Stanford Top-1 (N=3.730) | Độ Sụt Giảm ($\Delta_{drop}$) | Stanford Top-5 | Độ Trễ Trung Bình (Inference/ảnh) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-18** | 11.20M | 67.84% | **65.17%** | 2.67% | 89.57% | 7.0 ms |
| **SEResNet-18** | 11.28M | 70.62% | **63.49%** | 7.13% | 87.88% | 7.3 ms |
| **DenseNet-121** | 6.99M | 74.54% | **71.47%** | 3.07% | 91.90% | 21.8 ms |
| **EfficientNet-B0** | 4.05M | 83.32% | **77.51%** | 5.81% | 94.16% | 11.7 ms |
| **Dual-Stream PetNet** | 5.08M | **85.72%** | **85.74%** | **-0.02%** | **96.54%** | 22.0 ms |

> [!IMPORTANT]
> **Điểm nhấn khoa học đột phá:** Kiến trúc **Dual-Stream EfficientNet-B0 (PetNet)** đạt độ chính xác **85.74% Top-1** và **96.54% Top-5** trên toàn bộ 3.730 ảnh Stanford Dogs mà không hề bị sụt giảm độ chính xác ($\Delta_{drop} = -0.02\%$), duy trì kết quả tương đương (thậm chí nhỉnh hơn 0.02%) so với tập test gốc của Oxford-IIIT Pet.

---

## 4. Phân Tích Hiệu Năng Từng Giống Của Dual-Stream PetNet Trên 3.730 Ảnh

| Nhóm Phân Hạng Độ Chính Xác | Tên Giống Thú Cưng | Số Lượng Đúng / Tổng Số Ảnh | Độ Chính Xác (%) |
| :--- | :--- | :---: | :---: |
| **Nhóm Siêu Cao ($\ge 95\%$)** | Keeshond | 153 / 158 | **96.84%** |
| | Saint Bernard | 164 / 170 | **96.47%** |
| | Yorkshire Terrier | 157 / 164 | **95.73%** |
| | Scottish Terrier | 151 / 158 | **95.57%** |
| | Wheaten Terrier | 149 / 156 | **95.51%** |
| | Japanese Chin | 176 / 185 | **95.14%** |
| **Nhóm Rất Cao ($90\% - 95\%$)** | English Setter | 151 / 161 | **93.79%** |
| | German Shorthaired | 142 / 152 | **93.42%** |
| | Miniature Pinscher | 170 / 184 | **92.39%** |
| **Nhóm Tốt ($80\% - 90\%$)** | English Cocker Spaniel | 143 / 159 | **89.94%** |
| | Beagle | 173 / 195 | **88.72%** |
| | Samoyed | 188 / 218 | **86.24%** |
| | Leonberger | 179 / 210 | **85.24%** |
| | Pomeranian | 180 / 219 | **82.19%** |
| **Nhóm Khá ($70\% - 80\%$)** | Boxer | 120 / 151 | **79.47%** |
| | Great Pyrenees | 166 / 213 | **77.93%** |
| | Basset Hound | 135 / 175 | **77.14%** |
| | Staffordshire Bull Terrier | 118 / 155 | **76.13%** |
| | Pug | 146 / 200 | **73.00%** |
| **Nhóm Chịu Thử Thách Miền ($< 70\%$)** | Chihuahua | 106 / 152 | **69.74%** |
| | Newfoundland | 131 / 195 | **67.18%** |

---

## 5. Kết Luận Khoa Học

1. **Kiểm định giả thuyết nghiên cứu hoàn tất thắng lợi:**
   - Cả 5 kiến trúc đều vượt qua tiêu chuẩn kiểm định khoa học khắt khe ($\Delta_{drop} < 10\%$) trên toàn bộ 3.730 ảnh độc lập từ Stanford Dogs.
   - Thử nghiệm trên kích thước mẫu toàn phần ($N = 3.730$) xác thực tính ổn định vượt bậc so với các bài test mẫu nhỏ trước đó.

2. **Ưu thế của cơ chế Dual-Stream kết hợp Gated Fusion:**
   - Các mô hình đơn luồng truyền thống phụ thuộc nhiều vào bối cảnh nền xung quanh vật nuôi, dẫn đến việc chuyển dịch miền dữ liệu (Domain Shift từ ảnh sinh hoạt gia đình sang ảnh dã ngoại ngoài trời) làm giảm từ 2.67% đến 7.13% độ chính xác.
   - **Dual-Stream PetNet** tách riêng luồng nhận diện toàn thân (đã tách nền bằng DeepLabV3 MobileNetV3) và luồng nhận diện đặc trưng khuôn mặt/đầu (Head Crop Heuristic), phối hợp qua cổng Gated Fusion. Cơ chế này giúp mô hình tập trung vào hình thái giải phẫu của giống loài, mang lại khả năng thích ứng miền dữ liệu vượt trội.

3. **Tài nguyên & File dữ liệu:**
   - Dữ liệu kết quả chi tiết dạng JSON: [report/stanford_dogs_cross_dataset_benchmark.json](file:///Users/nguyenhuuhoangluan/Git_Code/ml-semester-cat-dog-classifier/report/stanford_dogs_cross_dataset_benchmark.json)
   - Mã nguồn thực thi tái hiện kết quả: [scripts/benchmark_stanford_dogs.py](file:///Users/nguyenhuuhoangluan/Git_Code/ml-semester-cat-dog-classifier/scripts/benchmark_stanford_dogs.py)
