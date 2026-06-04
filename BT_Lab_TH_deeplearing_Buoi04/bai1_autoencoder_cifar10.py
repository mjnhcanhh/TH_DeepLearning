"""
==============================================
BÀI TẬP 1: Autoencoder trên CIFAR10
==============================================
Labels:
  0: airplane, 1: automobile, 2: bird, 3: cat, 4: deer
  5: dog,      6: frog,       7: horse, 8: ship, 9: truck
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# ─────────────────────────────────────────────
# 1. LOAD & TIỀN XỬ LÝ DỮ LIỆU
# ─────────────────────────────────────────────
CIFAR10_LABELS = {
    0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat',  4: 'deer',
    5: 'dog',      6: 'frog',       7: 'horse', 8: 'ship', 9: 'truck'
}

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32')  / 255.0

print(f"Train: {x_train.shape} | Test: {x_test.shape}")

# ─────────────────────────────────────────────
# 2. XÂY DỰNG CONVOLUTIONAL AUTOENCODER
# ─────────────────────────────────────────────
def build_autoencoder():
    input_img = keras.Input(shape=(32, 32, 3))

    # Encoder
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(input_img)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2), padding='same')(x)          # 16x16x64

    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2), padding='same')(x)          # 8x8x32

    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    encoded = layers.MaxPooling2D((2,2), padding='same', name='encoded')(x)  # 4x4x16

    # Decoder
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2,2))(x)                          # 8x8x16

    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2,2))(x)                          # 16x16x32

    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2,2))(x)                          # 32x32x64

    decoded = layers.Conv2D(3, (3,3), activation='sigmoid', padding='same')(x)

    autoencoder = keras.Model(input_img, decoded, name='autoencoder')
    encoder     = keras.Model(input_img, encoded, name='encoder')
    return autoencoder, encoder

autoencoder, encoder = build_autoencoder()
autoencoder.summary()

# ─────────────────────────────────────────────
# 3. HUẤN LUYỆN
# ─────────────────────────────────────────────
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = autoencoder.fit(
    x_train, x_train,
    epochs=50,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test),
    verbose=1
)

# ─────────────────────────────────────────────
# 4. DỰ BÁO & HIỂN THỊ
# ─────────────────────────────────────────────
decoded_imgs = autoencoder.predict(x_test)

n = 10
fig, axes = plt.subplots(2, n, figsize=(20, 4))
for i in range(n):
    axes[0, i].imshow(x_test[i])
    axes[0, i].set_title(CIFAR10_LABELS[int(y_test[i])], fontsize=8)
    axes[0, i].axis('off')
    axes[1, i].imshow(decoded_imgs[i])
    axes[1, i].set_title('recon', fontsize=8)
    axes[1, i].axis('off')

plt.suptitle('CIFAR10 - Original (top) vs Reconstructed (bottom)')
plt.tight_layout()
plt.savefig('cifar10_reconstruction.png', dpi=150)
plt.show()

# Loss curve
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'],     label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('CIFAR10 Autoencoder - Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.legend(); plt.grid(True)
plt.savefig('cifar10_loss.png', dpi=150)
plt.show()

# ─────────────────────────────────────────────
# 5. PHÂN LOẠI BẰNG KNN TRÊN ENCODED FEATURES
# ─────────────────────────────────────────────
enc_train = encoder.predict(x_train).reshape(len(x_train), -1)
enc_test  = encoder.predict(x_test).reshape(len(x_test),  -1)

knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn.fit(enc_train, y_train.ravel())
y_pred = knn.predict(enc_test)

print(f"\nKNN Accuracy: {accuracy_score(y_test.ravel(), y_pred):.4f}")
print(classification_report(y_test.ravel(), y_pred,
      target_names=[CIFAR10_LABELS[i] for i in range(10)]))
