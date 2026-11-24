# Tổng Hợp Các Bài Báo Nghiên Cứu (Research Papers) Cho 4 Kiến Trúc Mô Hình Cốt Lõi

Tài liệu này tổng hợp cơ sở lý thuyết, công trình nghiên cứu gốc (Original Papers), công thức toán học, cơ chế hoạt động và vai trò thực nghiệm của 4 kiến trúc mạng học sâu chủ chốt được triển khai và huấn luyện từ đầu (Train from Scratch) trong đồ án:
1. **ResNet-18** (CVPR 2016)
2. **SEResNet-18 / SENet** (CVPR 2018)
3. **DenseNet-121** (CVPR 2017)
4. **EfficientNet-B0** (ICML 2019)

---

## 1. Danh Mục File PDF Bài Báo Đã Tải Về Offline

Tất cả các bài báo gốc dạng PDF truy cập mở từ arXiv đã được lưu trữ sẵn trong thư mục [R&D/papers/](file:///Users/nguyenhuuhoangluan/Git_Code/ml-semester-cat-dog-classifier/R&D/papers):

| STT | Mô Hình | Tên File PDF Lưu Trữ | Nguồn Gốc & Liên Kết Gốc |
| :--- | :--- | :--- | :--- |
| 1 | **ResNet** | [resnet_cvpr2016.pdf](file:///Users/nguyenhuuhoangluan/Git_Code/ml-semester-cat-dog-classifier/R&D/papers/resnet_cvpr2016.pdf) | [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) |
| 2 | **SENet / SEResNet** | [senet_cvpr2018.pdf](file:///Users/nguyenhuuhoangluan/Git_Code/ml-semester-cat-dog-classifier/R&D/papers/senet_cvpr2018.pdf) | [arXiv:1709.01507](https://arxiv.org/abs/1709.01507) |
| 3 | **DenseNet** | [densenet_cvpr2017.pdf](file:///Users/nguyenhuuhoangluan/Git_Code/ml-semester-cat-dog-classifier/R&D/papers/densenet_cvpr2017.pdf) | [arXiv:1608.06993](https://arxiv.org/abs/1608.06993) |
| 4 | **EfficientNet** | [efficientnet_icml2019.pdf](file:///Users/nguyenhuuhoangluan/Git_Code/ml-semester-cat-dog-classifier/R&D/papers/efficientnet_icml2019.pdf) | [arXiv:1905.11946](https://arxiv.org/abs/1905.11946) |

---

## 2. Chi Tiết Từng Công Trình Nghiên Cứu

### 2.1. ResNet-18: Deep Residual Learning for Image Recognition

- **Tác giả:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (Microsoft Research).
- **Hội nghị:** IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2016) - **Best Paper Award**.
- **Liên kết arXiv:** [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)
- **File PDF cục bộ:** [resnet_cvpr2016.pdf](file:///Users/nguyenhuuhoangluan/Git_Code/ml-semester-cat-dog-classifier/R&D/papers/resnet_cvpr2016.pdf)

#### Đóng góp khoa học cốt lõi:
- **Vấn đề giải quyết:** Hiện tượng suy thoái độ chính xác (Degradation Problem) và biến mất đạo hàm (Vanishing Gradient) khi xếp chồng quá nhiều tầng tích chập thông thường (Plain Net).
- **Cơ chế Residual (Skip Connection):** Thay vì bắt các tầng học trực tiếp ánh xạ mục tiêu $H(x)$, mạng được tái cấu trúc để học phần thặng dư $F(x) = H(x) - x$. Đầu ra khối được tính bởi:
  $$y = \mathcal{F}(x, \{W_i\}) + x$$
  Khi gradient truyền ngược, dòng thông tin có thể truyền thẳng qua đường tắt (identity shortcut) $\frac{\partial y}{\partial x} = \frac{\partial \mathcal{F}}{\partial x} + 1$, ngăn chặn triệt để triệt tiêu gradient.
- **Biến thể ResNet-18:** Sử dụng BasicBlock (2 lớp Conv $3 \times 3$ với Batch Normalization và ReLU), tổng số tham số khoảng 11.20M, phù hợp làm mốc đối sánh chuẩn (Baseline).

#### Trích dẫn BibTeX:
```bibtex
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}
```

---

### 2.2. SEResNet-18: Squeeze-and-Excitation Networks (SENet)

- **Tác giả:** Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu.
- **Hội nghị:** IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2018) - **Best Paper Award** (Quán quân giải ILSVRC 2017).
- **Liên kết arXiv:** [https://arxiv.org/abs/1709.01507](https://arxiv.org/abs/1709.01507)
- **File PDF cục bộ:** [senet_cvpr2018.pdf](file:///Users/nguyenhuuhoangluan/Git_Code/ml-semester-cat-dog-classifier/R&D/papers/senet_cvpr2018.pdf)

#### Đóng góp khoa học cốt lõi:
- **Vấn đề giải quyết:** Các mạng CNN chuẩn xử lý mọi kênh đặc trưng (channels) với trọng số bình đẳng, bỏ qua mối tương quan phụ thuộc lẫn nhau giữa các kênh (inter-channel relationships).
- **Cơ chế Squeeze-and-Excitation (SE Block):** Tự động tái chuẩn hóa trọng số kênh (Channel Attention Recalibration):
  1. **Squeeze:** Nén đặc trưng không gian $H \times W$ về vector $1 \times 1 \times C$ thông qua Global Average Pooling (GAP):
     $$z_c = \mathbf{F}_{sq}(\mathbf{u}_c) = \frac{1}{H \times W} \sum_{i=1}^H \sum_{j=1}^W u_c(i,j)$$
  2. **Excitation:** Khai thác tương quan phi tuyến giữa các kênh thông qua 2 tầng Fully-Connected với tỉ lệ giảm chiều $r=16$ (reduction ratio):
     $$s = \mathbf{F}_{ex}(z, \mathbf{W}) = \sigma(\mathbf{W}_2 \delta(\mathbf{W}_1 z))$$
     với $\delta$ là hàm ReLU, $\sigma$ là hàm Sigmoid.
  3. **Scale:** Nhân trọng số chú ý $s_c$ ngược lại bản đồ đặc trưng ban đầu:
     $$\mathbf{\tilde{x}}_c = \mathbf{F}_{scale}(\mathbf{u}_c, s_c) = s_c \mathbf{u}_c$$
- **Ứng dụng SEResNet-18:** Tích hợp SE Block vào cuối mỗi BasicBlock của ResNet-18, giúp mô hình tập trung mạnh vào các kênh biểu diễn cấu trúc giải phẫu sinh học (tai, mũi, đốm lông thú cưng).

#### Trích dẫn BibTeX:
```bibtex
@inproceedings{hu2018squeeze,
  title={Squeeze-and-excitation networks},
  author={Hu, Jie and Shen, Li and Sun, Gang},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={7132--7141},
  year={2018}
}
```

---

### 2.3. DenseNet-121: Densely Connected Convolutional Networks

- **Tác giả:** Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger (Cornell University, Tsinghua University, Facebook AI Research).
- **Hội nghị:** IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2017) - **Best Paper Award**.
- **Liên kết arXiv:** [https://arxiv.org/abs/1608.06993](https://arxiv.org/abs/1608.06993)
- **File PDF cục bộ:** [densenet_cvpr2017.pdf](file:///Users/nguyenhuuhoangluan/Git_Code/ml-semester-cat-dog-classifier/R&D/papers/densenet_cvpr2017.pdf)

#### Đóng góp khoa học cốt lõi:
- **Vấn đề giải quyết:** Giảm thiểu sự lãng phí đặc trưng (feature redundancy) và tối ưu hóa dòng chảy thông tin qua các tầng.
- **Cơ chế Dense Connectivity:** Mỗi tầng nhận trực tiếp bản đồ đặc trưng của toàn bộ các tầng trước đó thông qua phép ghép nối kênh (Channel Concatenation), thay vì phép cộng như ResNet:
  $$\mathbf{x}_\ell = H_\ell([\mathbf{x}_0, \mathbf{x}_1, \dots, \mathbf{x}_{\ell-1}])$$
- **Growth Rate ($k=32$):** Mỗi tầng chỉ sinh ra thêm $k$ kênh mới, giúp mạng mỏng nhẹ hơn nhưng vẫn tái sử dụng triệt để biểu diễn đặc trưng đa cấp độ (Multi-scale Feature Reuse).
- **Transition Layer:** Giảm kích thước không gian và giảm số kênh thông qua tích chập $1 \times 1$ và Average Pooling $2 \times 2$.
- **Ưu thế:** Hiệu quả tham số cao (~6.99M params), học tốt các chi tiết vân lông mịn của chó mèo.

#### Trích dẫn BibTeX:
```bibtex
@inproceedings{huang2017densely,
  title={Densely connected convolutional networks},
  author={Huang, Gao and Liu, Zhuang and Van Der Maaten, Laurens and Weinberger, Kilian Q},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={4700--4708},
  year={2017}
}
```

---

### 2.4. EfficientNet-B0: Rethinking Model Scaling for Convolutional Neural Networks

- **Tác giả:** Mingxing Tan, Quoc V. Le (Google Research, Brain Team).
- **Hội nghị:** International Conference on Machine Learning (ICML 2019).
- **Liên kết arXiv:** [https://arxiv.org/abs/1905.11946](https://arxiv.org/abs/1905.11946)
- **File PDF cục bộ:** [efficientnet_icml2019.pdf](file:///Users/nguyenhuuhoangluan/Git_Code/ml-semester-cat-dog-classifier/R&D/papers/efficientnet_icml2019.pdf)

#### Đóng góp khoa học cốt lõi:
- **Vấn đề giải quyết:** Các phương pháp scale mạng truyền thống chỉ tăng đơn lẻ chiều sâu (Depth - $d$), chiều rộng (Width - $w$) hoặc độ phân giải ảnh (Resolution - $r$), dẫn đến bão hòa độ chính xác nhanh chóng và chi phí tính toán tăng vọt.
- **Compound Scaling:** Cân đối đồng thời cả ba chiều theo hệ số mở rộng $\phi$:
  $$\text{depth: } d = \alpha^\phi, \quad \text{width: } w = \beta^\phi, \quad \text{resolution: } r = \gamma^\phi$$
  $$\text{thỏa mãn: } \alpha \cdot \beta^2 \cdot \gamma^2 \approx 2 \quad (\alpha \ge 1, \beta \ge 1, \gamma \ge 1)$$
- **Khối MBConv (Mobile Inverted Bottleneck Conv):** Kết hợp Depthwise Separable Convolution, Inverted Residuals và Squeeze-and-Excitation, giúp giảm FLOPs xuống mức rất thấp nhưng vẫn duy trì năng lực biểu diễn cao cấp.
- **Hiệu quả thực tế:** EfficientNet-B0 chỉ cần **4.05M tham số** và ~0.39 GFLOPs nhưng vượt trội toàn diện so với các mạng lớn hơn nhiều lần.

#### Trích dẫn BibTeX:
```bibtex
@inproceedings{tan2019efficientnet,
  title={Efficientnet: Rethinking model scaling for convolutional neural networks},
  author={Tan, Mingxing and Le, Quoc},
  booktitle={International conference on machine learning},
  pages={6105--6114},
  year={2019},
  organization={PMLR}
}
```

---

## 3. Tổng Hợp Đối Sánh Thực Nghiệm Trong Đồ Án (Train From Scratch)

Toàn bộ 4 mô hình đều được huấn luyện từ đầu (Scratch, ngẫu nhiên hóa trọng số, không sử dụng bất kỳ trọng số pretrained nào) trên 2 bộ dữ liệu:
1. **Dataset Chuẩn (Oxford-IIIT Pet gốc):** 37 giống, kích thước $224 \times 224$, 3,680 ảnh train.
2. **Dataset Mới (Data Cải Tiến Toàn Diện):** 37 giống, kích thước $288 \times 288$, 9,301 ảnh train (CLAHE, Bilateral, Part-level Augmentation).

### Bảng Chỉ Số Thực Nghiệm Tổng Hợp

| Kiến Trúc | Số Tham Số (#Params) | Độ Phức Tạp (GFLOPs) | Oxford Gốc (Raw Acc) | Oxford Gốc (TTA Acc) | Dataset Mới (Raw Acc) | Dataset Mới (TTA Acc) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-18** | 11.20M | ~1.82 | 71.65% | 72.50% | 67.84% | 68.47% |
| **SEResNet-18** | 11.28M | ~1.83 | 68.00% | 68.82% | 70.62% | 71.55% |
| **DenseNet-121** | 6.99M | ~2.88 | 72.14% | 73.51% | 74.54% | 75.33% |
| **EfficientNet-B0** | **4.05M** | **~0.39** | **74.19%** | **75.28%** | **83.32%** | **83.78%** |
| *Dual-Stream PetNet* | *5.08M* | *~0.78* | *-* | *-* | **85.72%** | **86.54%** |

---

## 4. Liên Kết Tài Liệu Tham Khảo Trong Dự Án

- Báo cáo tổng thể cải tiến: [report/bao_cao_cai_tien_toan_dien.md](file:///Users/nguyenhuuhoangluan/Git_Code/ml-semester-cat-dog-classifier/report/bao_cao_cai_tien_toan_dien.md)
- Hướng dẫn và benchmark chính: [README.md](file:///Users/nguyenhuuhoangluan/Git_Code/ml-semester-cat-dog-classifier/README.md)
- Notebook thực nghiệm Mai Anh Luân: [notebooks/Anh_Luan/](file:///Users/nguyenhuuhoangluan/Git_Code/ml-semester-cat-dog-classifier/notebooks/Anh_Luan)
- Notebook thực nghiệm Hoàng Lương Phước: [notebooks/Phuoc/](file:///Users/nguyenhuuhoangluan/Git_Code/ml-semester-cat-dog-classifier/notebooks/Phuoc)
- Thư mục chứa bài báo PDF: [R&D/papers/](file:///Users/nguyenhuuhoangluan/Git_Code/ml-semester-cat-dog-classifier/R&D/papers)
