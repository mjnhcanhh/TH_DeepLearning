# HƯỚNG DẪN CHẠY - THỰC HÀNH 5: AUTOENCODER
=========================================

## CÀI ĐẶT THƯ VIỆN

```bash
pip install tensorflow scikit-learn matplotlib seaborn flask pillow numpy
pip install tensorflow-datasets   # (tuỳ chọn, cho bài 2/4 khi không có dữ liệu)
```

---

## BÀI 1 – CIFAR10

```bash
python bai1_autoencoder_cifar10.py
```

- Tự động tải CIFAR10 từ Keras
- Train 50 epochs, batch 256
- Xuất file: `cifar10_reconstruction.png`, `cifar10_loss.png`
- Đánh giá bằng KNN trên encoded features

---

## BÀI 2 – Cat vs Dog

```bash
python bai2_autoencoder_catdog.py
```

**Nếu có dữ liệu giảng viên**, đặt vào:
```
data/catdog/
  train/
    cat/   ← ảnh .jpg
    dog/   ← ảnh .jpg
  test/
    cat/
    dog/
```

**Nếu không có dữ liệu**, script tự tải từ `tensorflow_datasets` (cats_vs_dogs ~800MB).

---

## BÀI 3 – Fashion-MNIST

```bash
python bai3_autoencoder_fashion.py
```

- Tự động tải Fashion-MNIST từ Keras nếu không có dữ liệu giảng viên
- Nếu giảng viên cung cấp file `.npz`, đặt vào `data/fashion_mnist/fashion_mnist.npz`
- Xuất: `fashion_reconstruction.png`, `fashion_confusion.png`, `fashion_training.png`

---

## BÀI 4 – Face Gender (Nam/Nữ)

```bash
python bai4_autoencoder_face_gender.py
```

**Nếu có dữ liệu giảng viên**, đặt vào:
```
data/face/
  train/
    male/
    female/
  test/
    male/
    female/
```

**Nếu không có dữ liệu**, script tự tải CelebA (attribute Male/Female) từ `tensorflow_datasets`.

---

## BÀI 5 – Flask Web App

### Bước 1: Lưu model sau khi train
Sau khi chạy xong bài 1–4, thêm đoạn sau vào cuối mỗi script:

```python
import os
os.makedirs('models', exist_ok=True)

# Bài 1
autoencoder.save('models/cifar10_ae.h5')
# classifier.save('models/cifar10_clf.h5')   # nếu có

# Bài 2
autoencoder.save('models/catdog_ae.h5')
classifier.save('models/catdog_clf.h5')

# Bài 3
autoencoder.save('models/fashion_ae.h5')
classifier.save('models/fashion_clf.h5')

# Bài 4
autoencoder.save('models/face_ae.h5')
classifier.save('models/face_clf.h5')
```

### Bước 2: Chạy Flask
```bash
python bai5_flask_app.py
```

### Bước 3: Mở trình duyệt
```
http://localhost:5000
```

---

## CẤU TRÚC THƯ MỤC

```
project/
├── bai1_autoencoder_cifar10.py
├── bai2_autoencoder_catdog.py
├── bai3_autoencoder_fashion.py
├── bai4_autoencoder_face_gender.py
├── bai5_flask_app.py
├── README.md
├── models/                  ← lưu .h5 sau khi train
├── data/
│   ├── catdog/
│   ├── fashion_mnist/
│   └── face/
└── output images/           ← ảnh kết quả
```

---

## GHI CHÚ

| Bài | Dataset | Input size | Kiến trúc |
|-----|---------|-----------|-----------|
| 1 | CIFAR10 (tự động) | 32×32×3 | Conv AE + KNN |
| 2 | Cat/Dog (tfds/GV) | 64×64×3 | Conv AE + Dense CLF |
| 3 | Fashion-MNIST (tự động) | 28×28×1 | Conv AE + Dense CLF |
| 4 | CelebA/GV | 64×64×3 | Conv AE + Dense CLF + ROC |
| 5 | – | – | Flask REST API + HTML UI |
