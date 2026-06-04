"""
==============================================
BÀI TẬP 3: Autoencoder nhận dạng Fashion-MNIST
==============================================
Labels:
  0: T-shirt/top, 1: Trouser,  2: Pullover, 3: Dress,   4: Coat
  5: Sandal,      6: Shirt,    7: Sneaker,  8: Bag,      9: Ankle boot
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
EPOCHS     = 50
BATCH_SIZE = 256
ENCODING_DIM = 64

FASHION_LABELS = {
    0: 'T-shirt/top', 1: 'Trouser',  2: 'Pullover', 3: 'Dress',     4: 'Coat',
    5: 'Sandal',      6: 'Shirt',    7: 'Sneaker',   8: 'Bag',       9: 'Ankle boot'
}

# ─────────────────────────────────────────────
# 1. LOAD DỮ LIỆU
#    Ưu tiên: thư mục giảng viên → keras built-in
# ─────────────────────────────────────────────
DATA_PATH = 'data/fashion_mnist'    # Thay đường dẫn giảng viên nếu có

def load_fashion_mnist(data_path=None):
    if data_path and os.path.exists(data_path):
        # Load từ file npz do giảng viên cung cấp
        d = np.load(os.path.join(data_path, 'fashion_mnist.npz'))
        x_train, y_train = d['x_train'], d['y_train']
        x_test,  y_test  = d['x_test'],  d['y_test']
        print(f"Loaded from {data_path}")
    else:
        # Tải từ keras
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        print("Loaded from keras.datasets.fashion_mnist")

    # Normalize & reshape
    x_train = x_train.astype('float32') / 255.0
    x_test  = x_test.astype('float32')  / 255.0

    # Thêm channel dimension: (N, 28, 28) → (N, 28, 28, 1)
    x_train = x_train[..., np.newaxis]
    x_test  = x_test[..., np.newaxis]

    print(f"Train: {x_train.shape} | Test: {x_test.shape}")
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_fashion_mnist(DATA_PATH)

# ─────────────────────────────────────────────
# 2. XÂY DỰNG CONVOLUTIONAL AUTOENCODER
# ─────────────────────────────────────────────
def build_fashion_autoencoder():
    input_img = keras.Input(shape=(28, 28, 1))

    # ── Encoder ──
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2,2), padding='same')(x)        # 14x14x32

    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2), padding='same')(x)        # 7x7x64

    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2,2), padding='same', name='encoded')(x)  # 4x4x128

    # ── Decoder ──
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2,2))(x)                        # 8x8

    x = layers.Conv2D(64,  (3,3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2,2))(x)                        # 16x16
    x = layers.Cropping2D(((1,1),(1,1)))(x)                  # → 14x14

    x = layers.Conv2D(32,  (3,3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2,2))(x)                        # 28x28

    decoded = layers.Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)

    autoencoder = keras.Model(input_img, decoded, name='fashion_autoencoder')
    encoder     = keras.Model(input_img, encoded, name='fashion_encoder')
    return autoencoder, encoder

autoencoder, encoder = build_fashion_autoencoder()
autoencoder.summary()

# ─────────────────────────────────────────────
# 3. HUẤN LUYỆN AUTOENCODER
# ─────────────────────────────────────────────
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

callbacks = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)
]

history = autoencoder.fit(
    x_train, x_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_data=(x_test, x_test),
    callbacks=callbacks,
    verbose=1
)

# ─────────────────────────────────────────────
# 4. XÂY DỰNG CLASSIFIER (FINE-TUNING)
# ─────────────────────────────────────────────
def build_classifier(encoder):
    # Unfreeze top layers
    for layer in encoder.layers[:-4]:
        layer.trainable = False
    for layer in encoder.layers[-4:]:
        layer.trainable = True

    x = layers.Flatten()(encoder.output)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(encoder.input, output, name='fashion_classifier')
    return model

classifier = build_classifier(encoder)
classifier.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
classifier.summary()

clf_history = classifier.fit(
    x_train, y_train,
    epochs=20,
    batch_size=BATCH_SIZE,
    validation_data=(x_test, y_test),
    callbacks=callbacks
)

# ─────────────────────────────────────────────
# 5. ĐÁNH GIÁ & HIỂN THỊ
# ─────────────────────────────────────────────
decoded_imgs = autoencoder.predict(x_test)
y_pred_probs = classifier.predict(x_test)
y_pred       = np.argmax(y_pred_probs, axis=1)

test_acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Test Accuracy: {test_acc:.4f}")
print(classification_report(y_test, y_pred,
      target_names=[FASHION_LABELS[i] for i in range(10)]))

# Hiển thị reconstruction (chọn 1 ảnh mỗi class)
fig, axes = plt.subplots(3, 10, figsize=(20, 6))
for cls in range(10):
    idx = np.where(y_test == cls)[0][0]
    axes[0, cls].imshow(x_test[idx].squeeze(), cmap='gray')
    axes[0, cls].set_title(FASHION_LABELS[cls], fontsize=7)
    axes[0, cls].axis('off')

    axes[1, cls].imshow(decoded_imgs[idx].squeeze(), cmap='gray')
    axes[1, cls].set_title('recon', fontsize=7)
    axes[1, cls].axis('off')

    axes[2, cls].set_visible(False)

plt.suptitle('Fashion-MNIST: Original (row1) | Reconstructed (row2)', fontsize=12)
plt.tight_layout()
plt.savefig('fashion_reconstruction.png', dpi=150)
plt.show()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[FASHION_LABELS[i] for i in range(10)],
            yticklabels=[FASHION_LABELS[i] for i in range(10)])
plt.title('Fashion-MNIST Confusion Matrix')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('fashion_confusion.png', dpi=150)
plt.show()

# Loss curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history['loss'], label='Train')
ax1.plot(history.history['val_loss'], label='Val')
ax1.set_title('Autoencoder Loss'); ax1.legend(); ax1.grid(True)

ax2.plot(clf_history.history['accuracy'], label='Train')
ax2.plot(clf_history.history['val_accuracy'], label='Val')
ax2.set_title('Classifier Accuracy'); ax2.legend(); ax2.grid(True)

plt.tight_layout()
plt.savefig('fashion_training.png', dpi=150)
plt.show()

print("✅ Bài 3 hoàn thành!")
