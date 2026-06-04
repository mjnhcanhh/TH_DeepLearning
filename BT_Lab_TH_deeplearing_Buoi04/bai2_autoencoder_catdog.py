"""
==============================================
BÀI TẬP 2: Autoencoder nhận dạng Cat hoặc Dog
==============================================
Cấu trúc thư mục dữ liệu (do giảng viên cung cấp):
  data/
    train/
      cat/  (ảnh .jpg)
      dog/  (ảnh .jpg)
    test/
      cat/
      dog/

Nếu chưa có dữ liệu, script sẽ tự tải từ Kaggle Dogs vs Cats
(cần kaggle API key) hoặc dùng subset từ TF Datasets.
==============================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns

# ─────────────────────────────────────────────
# CẤU HÌNH
# ─────────────────────────────────────────────
IMG_SIZE   = (64, 64)
BATCH_SIZE = 64
EPOCHS     = 50
DATA_DIR   = 'data/catdog'          # Thay đường dẫn theo dữ liệu giảng viên
CLASSES    = ['cat', 'dog']

# ─────────────────────────────────────────────
# 1. LOAD DỮ LIỆU
#    Nếu có thư mục data/ → dùng ImageDataGenerator
#    Nếu không → tải tự động từ tensorflow_datasets
# ─────────────────────────────────────────────
def load_from_directory(data_dir, img_size, batch_size):
    """Load ảnh từ thư mục theo cấu trúc class/"""
    train_gen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )
    test_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_ds = train_gen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='input',     # Autoencoder: target = input
        subset='training'
    )
    val_ds = train_gen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='input',
        subset='validation'
    )
    test_ds = test_gen.flow_from_directory(
        os.path.join(data_dir, 'test'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    return train_ds, val_ds, test_ds


def load_from_tfds(img_size):
    """Tải dữ liệu từ tensorflow_datasets (cats_vs_dogs)"""
    try:
        import tensorflow_datasets as tfds
        ds, info = tfds.load('cats_vs_dogs', with_info=True, as_supervised=True)
        print("Dùng tensorflow_datasets cats_vs_dogs")

        def preprocess(img, label):
            img = tf.image.resize(img, img_size) / 255.0
            return img, label

        full_ds = ds['train'].map(preprocess).shuffle(1000)
        total   = info.splits['train'].num_examples   # 23262
        train_n = int(total * 0.8)

        x_all, y_all = [], []
        for img, lbl in full_ds:
            x_all.append(img.numpy())
            y_all.append(lbl.numpy())
        x_all = np.array(x_all, dtype='float32')
        y_all = np.array(y_all)

        x_train, y_train = x_all[:train_n], y_all[:train_n]
        x_test,  y_test  = x_all[train_n:], y_all[train_n:]
        return x_train, y_train, x_test, y_test

    except Exception as e:
        print(f"Không thể dùng tensorflow_datasets: {e}")
        print("Tạo dữ liệu giả để demo kiến trúc...")
        x_train = np.random.rand(500,  *img_size, 3).astype('float32')
        y_train = np.random.randint(0, 2, 500)
        x_test  = np.random.rand(100, *img_size, 3).astype('float32')
        y_test  = np.random.randint(0, 2, 100)
        return x_train, y_train, x_test, y_test


# Chọn cách load
if os.path.exists(DATA_DIR):
    USE_DIRECTORY = True
    train_ds, val_ds, test_ds = load_from_directory(DATA_DIR, IMG_SIZE, BATCH_SIZE)
else:
    USE_DIRECTORY = False
    x_train, y_train, x_test, y_test = load_from_tfds(IMG_SIZE)
    print(f"Train: {x_train.shape} | Test: {x_test.shape}")

# ─────────────────────────────────────────────
# 2. XÂY DỰNG AUTOENCODER
# ─────────────────────────────────────────────
def build_catdog_autoencoder(img_size):
    h, w = img_size
    input_img = keras.Input(shape=(h, w, 3))

    # ── Encoder ──
    x = layers.Conv2D(32,  (3,3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2,2), padding='same')(x)   # 32x32

    x = layers.Conv2D(64,  (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2), padding='same')(x)   # 16x16

    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2,2), padding='same', name='encoded')(x)  # 8x8

    # ── Decoder ──
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2,2))(x)

    x = layers.Conv2D(64,  (3,3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2,2))(x)

    x = layers.Conv2D(32,  (3,3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2,2))(x)

    decoded = layers.Conv2D(3, (3,3), activation='sigmoid', padding='same')(x)

    autoencoder = keras.Model(input_img, decoded, name='catdog_autoencoder')
    encoder     = keras.Model(input_img, encoded, name='catdog_encoder')
    return autoencoder, encoder

autoencoder, encoder = build_catdog_autoencoder(IMG_SIZE)
autoencoder.summary()

# ─────────────────────────────────────────────
# 3. HUẤN LUYỆN AUTOENCODER
# ─────────────────────────────────────────────
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
callbacks = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
]

if USE_DIRECTORY:
    history = autoencoder.fit(train_ds, epochs=EPOCHS,
                               validation_data=val_ds, callbacks=callbacks)
else:
    history = autoencoder.fit(
        x_train, x_train,
        epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True,
        validation_split=0.2, callbacks=callbacks
    )

# ─────────────────────────────────────────────
# 4. XÂY DỰNG CLASSIFIER TRÊN ENCODED FEATURES
# ─────────────────────────────────────────────
def build_classifier(encoder, img_size):
    """Thêm classification head lên encoder"""
    for layer in encoder.layers:
        layer.trainable = False      # Freeze encoder

    encoded_out = encoder.output
    x = layers.Flatten()(encoded_out)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64,  activation='relu')(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    classifier = keras.Model(encoder.input, output, name='classifier')
    return classifier

classifier = build_classifier(encoder, IMG_SIZE)
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier.summary()

if USE_DIRECTORY:
    # Tạo lại generator với class_mode='binary'
    clf_gen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255, validation_split=0.2)
    clf_train = clf_gen.flow_from_directory(
        os.path.join(DATA_DIR, 'train'), target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode='binary', subset='training')
    clf_val = clf_gen.flow_from_directory(
        os.path.join(DATA_DIR, 'train'), target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode='binary', subset='validation')
    clf_history = classifier.fit(clf_train, epochs=20,
                                  validation_data=clf_val, callbacks=callbacks)
else:
    clf_history = classifier.fit(
        x_train, y_train,
        epochs=20, batch_size=BATCH_SIZE,
        validation_split=0.2, callbacks=callbacks
    )

# ─────────────────────────────────────────────
# 5. ĐÁNH GIÁ & HIỂN THỊ
# ─────────────────────────────────────────────
if not USE_DIRECTORY:
    decoded_imgs = autoencoder.predict(x_test[:10])
    y_pred_prob  = classifier.predict(x_test).flatten()
    y_pred       = (y_pred_prob > 0.5).astype(int)

    # Hiển thị reconstruction
    fig, axes = plt.subplots(2, 10, figsize=(20, 4))
    for i in range(10):
        axes[0, i].imshow(x_test[i])
        axes[0, i].set_title(CLASSES[y_test[i]], fontsize=8)
        axes[0, i].axis('off')
        axes[1, i].imshow(decoded_imgs[i])
        axes[1, i].set_title(f"pred:{CLASSES[int(y_pred[i])]}", fontsize=7)
        axes[1, i].axis('off')
    plt.suptitle('Cat vs Dog - Original (top) | Reconstructed + Prediction (bottom)')
    plt.tight_layout()
    plt.savefig('catdog_result.png', dpi=150)
    plt.show()

    # Classification report
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=CLASSES))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Cat vs Dog - Confusion Matrix')
    plt.savefig('catdog_confusion.png', dpi=150)
    plt.show()

print("✅ Bài 2 hoàn thành!")
