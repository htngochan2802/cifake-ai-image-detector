# CIFAKE – Phát hiện ảnh thật và ảnh do AI tạo bằng Fine-tuning CNN

Project phân loại nhị phân **REAL vs. FAKE** trên bộ dữ liệu [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images).
So sánh 4 backbone ImageNet (**ResNet50, VGG16, EfficientNetV2B0, MobileNetV3Small**) dùng chung một đầu phân loại,
cài đặt bằng TensorFlow / Keras.

Project được phát triển từ notebook Kaggle [Fine-tuning CNN](https://www.kaggle.com/code/guptaachal02/fine-tuning-cnn)
của guptaachal02, tách thành các module có thể tái sử dụng, có test và CI.

## Dữ liệu

CIFAKE gồm ảnh thật lấy từ CIFAR-10 và ảnh giả do Stable Diffusion 1.4 sinh ra, kích thước 32x32:
100.000 ảnh train (50k mỗi lớp) và 20.000 ảnh test (10k mỗi lớp). Cấu trúc thư mục cần có:

```
cifake/
├── train/{FAKE,REAL}/
└── test/{FAKE,REAL}/
```

Nếu không truyền `--data_dir`, chương trình tự tải dataset bằng `kagglehub` (cần tài khoản Kaggle).

## Cấu trúc dự án

```
cifake-ai-image-detector/
├── cifake/
│   ├── data.py        # nạp dữ liệu, chia train/val/test
│   ├── models.py      # 4 backbone + đầu phân loại chung (thêm mô hình = thêm 1 dòng vào SPECS)
│   ├── train.py       # huấn luyện + đánh giá, hỗ trợ chạy nhiều mô hình và tạo bảng so sánh
│   ├── evaluate.py    # metrics, confusion matrix, đường cong huấn luyện, bảng so sánh
│   └── predict.py     # dự đoán ảnh mới bằng mô hình đã train
├── tests/             # smoke test (không cần tải dữ liệu hay trọng số pretrained)
├── .github/workflows/ci.yml
├── outputs/           # kết quả chạy (bị .gitignore, giữ lại thư mục)
└── pyproject.toml
```

## Cài đặt

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e .          # thêm ".[dev]" nếu muốn chạy test và lint
```

## Sử dụng

```bash
# Train một mô hình
cifake-train --models efficientnetv2b0 --data_dir /path/to/cifake

# Train cả 4 mô hình và tạo bảng so sánh (outputs/comparison.md)
cifake-train --models all

# Chạy thử nhanh để kiểm tra pipeline
cifake-train --models mobilenetv3small --epochs 2 --weights none

# Dự đoán ảnh mới
cifake-predict --model_dir outputs/efficientnetv2b0 anh1.png anh2.jpg
```

Có thể thay `cifake-train` bằng `python -m cifake.train`. Các tham số chính: `--models`, `--epochs` (mặc định 100,
có early stopping), `--batch_size` (500), `--lr` (1e-3), `--val_split` (0.1), `--patience` (10),
`--weights {imagenet,none}`, `--optimizer {adam,adamax}`, `--output_dir`. Xem đầy đủ bằng `cifake-train -h`.

Mỗi mô hình ghi vào `outputs/<model>/`: `metrics.json`, `history.json`, `config.json`, `training_curves.png`,
`confusion_matrix.png` và file mô hình `<model>_cifake.keras`. Bảng so sánh nằm ở `outputs/comparison.md`.

## Kết quả

**Kết quả ghi trong notebook gốc** (đọc từ output của notebook; tập test được dùng luôn làm validation cho early stopping nên
số liệu có thể hơi lạc quan):

| Mô hình | Accuracy | Precision | Recall |
|---|---|---|---|
| ResNet50 | 0.9590 | 0.9626 | 0.9551 |
| VGG16 | 0.9590 | 0.9698 | 0.9476 |
| EfficientNetV2B0 | 0.9708 | 0.9831 | 0.9581 |
| MobileNetV3Small | 0.9528 | 0.9399 | 0.9676 |

(Phần chữ của notebook nêu 97,29% cho EfficientNet, trong khi output đã lưu cho thấy 97,08%.)

**Kết quả của project này** (validation tách riêng, test chỉ dùng một lần): _điền sau khi chạy_ từ `outputs/comparison.md`.

| Mô hình | Accuracy | Precision | Recall | AUC |
|---|---|---|---|---|
| ResNet50 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| VGG16 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| EfficientNetV2B0 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| MobileNetV3Small | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

## Ghi chú thiết kế

Những điểm khác với notebook gốc (có thể làm kết quả khác một chút):

- **Validation tách khỏi test:** 10% của thư mục `train` làm validation; `test` chỉ dùng một lần ở cuối.
- **Tiền xử lý đúng theo từng backbone:** ResNet50 và VGG16 dùng `preprocess_input` tương ứng (đặt trong pipeline dữ liệu,
  không nằm trong mô hình). EfficientNetV2 và MobileNetV3 đã có bước chuẩn hóa bên trong nên giữ nguyên ảnh 0–255.
  `cifake-predict` tự áp dụng đúng bước này; nếu tự gọi mô hình đã lưu, cần tiền xử lý tương tự.
- **Tập test không bị shuffle**, nên nhãn và dự đoán luôn khớp khi tính confusion matrix, AUC.
- **Tên metric cố định** (`precision`, `recall`, `auc`) thay vì `precision_1`, `recall_2`... như trong notebook.
- **Lưu mô hình định dạng `.keras`** thay cho `.h5` (định dạng cũ). Không commit file mô hình vào git (đã có trong `.gitignore`).

Giữ nguyên từ notebook gốc: đầu phân loại chung (BatchNorm → Dense 256 có L1/L2 → Dropout 0,4 → Dense 64 → sigmoid),
`pooling="max"`, fine-tune toàn bộ backbone với `training=False` cho BatchNorm, early stopping theo `val_loss`,
Adamax (MobileNetV3Small dùng Adam), learning rate 1e-3, batch 500.

Quy ước nhãn: thư mục xếp theo chữ cái nên `FAKE = 0`, `REAL = 1`; đầu ra sigmoid là xác suất ảnh là **REAL**,
và Precision/Recall của Keras tính cho lớp REAL (`metrics.json` có thêm báo cáo từng lớp).

## Hạn chế

Ảnh REAL của CIFAKE vốn là ảnh CIFAR-10 32x32 và toàn bộ ảnh FAKE chỉ do một generator (Stable Diffusion 1.4) tạo ra.
Điểm cao trên CIFAKE vì vậy chưa chứng minh mô hình phát hiện được ảnh AI nói chung: ảnh chụp độ phân giải cao hoặc ảnh
từ generator khác có thể cho kết quả kém hơn nhiều. `cifake-predict` luôn thu nhỏ ảnh về 32x32 trước khi dự đoán.

## Kiểm thử và lint

```bash
pip install -e ".[dev]"
ruff check .
pytest
```

Test dựng cả 4 mô hình (không tải trọng số) và chạy trọn quy trình train → so sánh → dự đoán trên dữ liệu giả.

## Mở rộng

Thêm backbone mới bằng cách thêm một `ModelSpec` vào `SPECS` trong `cifake/models.py` (constructor, `preprocess_input`,
optimizer mặc định). Tên mô hình mới tự xuất hiện trong `--models`.

## Trích dẫn

- Bird, J.J., Lotfi, A. (2023). *CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images.* arXiv:2303.14126.
- Krizhevsky, A., Hinton, G. (2009). *Learning multiple layers of features from tiny images.*
- Notebook gốc: guptaachal02, [Fine-tuning CNN](https://www.kaggle.com/code/guptaachal02/fine-tuning-cnn) (Kaggle).

Bộ dữ liệu CIFAKE được phát hành theo giấy phép MIT, cùng giấy phép với CIFAR-10. Hãy chọn giấy phép cho repo của bạn
và giữ phần ghi công cho notebook gốc.
