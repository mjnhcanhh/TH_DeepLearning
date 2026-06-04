"""
==============================================
BÀI TẬP 4: Autoencoder nhận dạng khuôn mặt {Nam, Nữ}
==============================================
Cấu trúc thư mục dữ liệu (do giảng viên cung cấp):
  data/face/
    train/
      male/    (ảnh .jpg)
      female/  (ảnh .jpg)
    test/
      male/
      female/

Nếu chưa có → tự tải CelebA subset từ tensorflow_datasets.
==============================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
import seaborn as sns

# ─────────────────────────────────────────────
# CẤU HÌNH
# ─────────────────────────────────────────────
IMG_SIZE   = (64, 64)
BATCH_SIZE = 32
EPOCHS_AE  = 50
EPOCHS_CLF = 20
DATA_DIR   = 'data/face'            # Thay đường dẫn giảng viên
CLASSES    = ['female', 'male']     # 0=female, 1=male

# ─────────────────────────────────────────────
# 1. LOAD DỮ LIỆU
# ─────────────────────────────────────────────
def load_from_directory(data_dir, img_size, batch_size):
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        horizontal_flip=True,
        validation_split=0.2
    )
    test_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_ds = datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=img_size, batch_size=batch_size,
        class_mode='binary', subset='training', shuffle=True
    )
    val_ds = datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=img_size, batch_size=batch_size,
        class_mode='binary', subset='validation'
    )
    test_ds = test_gen.flow_from_directory(
        os.path.join(data_dir, 'test'),
        target_size=img_size, batch_size=batch_size,
        class_mode='binary', shuffle=False
    )
    return train_ds, val_ds, test_ds


def load_from_celeba(img_size, max_samples=5000):
    """Load CelebA subset từ tensorflow_datasets (male attribute)"""
    try:
        import tensorflow_datasets as tfds
        print("Đang tải CelebA...")
        ds = tfds.load('celeb_a', split='train', as_supervised=False)

        images, labels = [], []
        for sample in ds.take(max_samples):
            img = tf.image.resize(sample['image'], img_size).numpy() / 255.0
            lbl = int(sample['attributes']['Male'].numpy())
            images.append(img)
            labels.append(lbl)

        images = np.array(images, dtype='float32')
        labels = np.array(labels)

        split = int(len(images) * 0.8)
        return images[:split], labels[:split], images[split:], labels[split:]

    except Exception as e:
        print(f"Không tải được CelebA: {e}")
        print("Dùng dữ liệu ngẫu nhiên để demo...")
        x_train = np.random.rand(400, *img_size, 3).astype('float32')
        y_train = np.random.randint(0, 2, 400)
        x_test  = np.random.rand(100, *img_size, 3).astype('float32')
        y_test  = np.random.randint(0, 2, 100)
        return x_train, y_train, x_test, y_test


USE_DIRECTORY = os.path.exists(DATA_DIR)
if USE_DIRECTORY:
    train_ds, val_ds, test_ds = load_from_directory(DATA_DIR, IMG_SIZE, BATCH_SIZE)
    print("✅ Loaded from directory")
else:
    x_train, y_train, x_test, y_test = load_from_celeba(IMG_SIZE)
    print(f"Train: {x_train.shape} | Test: {x_test.shape}")

# ─────────────────────────────────────────────
# 2. XÂY DỰNG AUTOENCODER (kiến trúc sâu hơn cho khuôn mặt)
# ─────────────────────────────────────────────
def build_face_autoencoder(img_size):
    h, w = img_size
    input_img = keras.Input(shape=(h, w, 3))

    # ── Encoder ──
    x = layers.Conv2D(32,  (3,3), activation='relu', padding='same')(input_img)
    x = layers.Conv2D(32,  (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)                          # 32x32x32
    x = layers.Dropout(0.1)(x)

    x = layers.Conv2D(64,  (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(64,  (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)                          # 16x16x64
    x = layers.Dropout(0.1)(x)

    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2,2), name='encoded')(x)    # 8x8x128

    # ── Decoder ──
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2,2))(x)                          # 16x16

    x = layers.Conv2D(64,  (3,3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2,2))(x)                          # 32x32

    x = layers.Conv2D(32,  (3,3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2,2))(x)                          # 64x64

    decoded = layers.Conv2D(3, (3,3), activation='sigmoid', padding='same')(x)

    autoencoder = keras.Model(input_img, decoded, name='face_autoencoder')
    encoder     = keras.Model(input_img, encoded, name='face_encoder')
    return autoencoder, encoder

autoencoder, encoder = build_face_autoencoder(IMG_SIZE)
autoencoder.summary()

# ─────────────────────────────────────────────
# 3. HUẤN LUYỆN AUTOENCODER
# ─────────────────────────────────────────────
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

callbacks = [
    keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, verbose=1)
]

if USE_DIRECTORY:
    # Tạo generator cho autoencoder (target = input)
    ae_gen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255, horizontal_flip=True, validation_split=0.2)
    ae_train = ae_gen.flow_from_directory(
        os.path.join(DATA_DIR, 'train'), target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode='input', subset='training')
    ae_val = ae_gen.flow_from_directory(
        os.path.join(DATA_DIR, 'train'), target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode='input', subset='validation')
    ae_history = autoencoder.fit(ae_train, epochs=EPOCHS_AE,
                                  validation_data=ae_val, callbacks=callbacks)
else:
    ae_history = autoencoder.fit(
        x_train, x_train,
        epochs=EPOCHS_AE, batch_size=BATCH_SIZE, shuffle=True,
        validation_split=0.2, callbacks=callbacks
    )

# ─────────────────────────────────────────────
# 4. CLASSIFIER: GENDER (Nam/Nữ)
# ─────────────────────────────────────────────
def build_gender_classifier(encoder):
    # Fine-tune: unlock top conv layers
    for layer in encoder.layers:
        layer.trainable = False
    for layer in encoder.layers[-6:]:
        layer.trainable = True

    x = layers.GlobalAveragePooling2D()(encoder.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64,  activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation='sigmoid', name='gender')(x)

    model = keras.Model(encoder.input, output, name='gender_classifier')
    return model

classifier = build_gender_classifier(encoder)
classifier.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)
classifier.summary()

if USE_DIRECTORY:
    clf_history = classifier.fit(train_ds, epochs=EPOCHS_CLF,
                                  validation_data=val_ds, callbacks=callbacks)
else:
    clf_history = classifier.fit(
        x_train, y_train,
        epochs=EPOCHS_CLF, batch_size=BATCH_SIZE,
        validation_split=0.2, callbacks=callbacks
    )

# ─────────────────────────────────────────────
# 5. ĐÁNH GIÁ & HIỂN THỊ
# ─────────────────────────────────────────────
if not USE_DIRECTORY:
    decoded_imgs = autoencoder.predict(x_test)
    y_prob = classifier.predict(x_test).flatten()
    y_pred = (y_prob > 0.5).astype(int)

    print(f"\n✅ Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=CLASSES))

    # Hiển thị kết quả
    n = 8
    fig, axes = plt.subplots(3, n, figsize=(18, 7))
    indices = np.random.choice(len(x_test), n, replace=False)
    for j, idx in enumerate(indices):
        # Gốc
        axes[0, j].imshow(x_test[idx])
        true_lbl = CLASSES[int(y_test[idx])]
        axes[0, j].set_title(f"True: {true_lbl}", fontsize=7)
        axes[0, j].axis('off')
        # Tái tạo
        axes[1, j].imshow(decoded_imgs[idx])
        pred_lbl = CLASSES[int(y_pred[idx])]
        color = 'green' if y_pred[idx] == y_test[idx] else 'red'
        axes[1, j].set_title(f"Pred: {pred_lbl}", fontsize=7, color=color)
        axes[1, j].axis('off')
        # Prob bar
        axes[2, j].bar(['female','male'], [1-y_prob[idx], y_prob[idx]],
                        color=['pink','steelblue'])
        axes[2, j].set_ylim(0, 1); axes[2, j].tick_params(labelsize=6)

    plt.suptitle('Face Gender Recognition - Original | Reconstructed | Probability', fontsize=11)
    plt.tight_layout()
    plt.savefig('face_gender_result.png', dpi=150)
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.title('ROC Curve - Gender Classification')
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.legend(); plt.grid(True)
    plt.savefig('face_roc.png', dpi=150)
    plt.show()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Gender Confusion Matrix')
    plt.savefig('face_confusion.png', dpi=150)
    plt.show()

print("✅ Bài 4 hoàn thành!")
