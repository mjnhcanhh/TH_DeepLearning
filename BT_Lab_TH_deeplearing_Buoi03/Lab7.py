# -*- coding: utf-8 -*-
# ============================================================
# Lab07 - MNIST Neural Network
# + Early Stopping: dừng nếu val_acc không tăng 5 epoch liên tiếp
# pip install torch torchvision matplotlib
# ============================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision
import torchvision.transforms as transforms

print("=" * 55)
print("  Lab07 — MNIST Handwritten Digit Recognition")
print("=" * 55)

# ── Cấu hình ─────────────────────────────────────────────────
BATCH_SIZE  = 128
EPOCHS      = 100        # tối đa, early stopping sẽ dừng sớm hơn
LR          = 0.001
PATIENCE    = 5          # dừng nếu val_acc không tăng 5 epoch liên tiếp
BEST_MODEL  = 'best_model_lab07.pth'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
print(f"Monitor: val_acc ↑ | Patience: {PATIENCE}")

# ── Bước 1: Loading Training Data ────────────────────────────
print("\n[1] Loading MNIST dataset...")

transform   = transforms.ToTensor()
train_data  = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
test_data   = torchvision.datasets.MNIST(root='./data', train=False,
                                          download=True, transform=transform)

x_train_raw = train_data.data.numpy()     # (60000, 28, 28)
y_train_raw = train_data.targets.numpy()  # (60000,)
x_test_raw  = test_data.data.numpy()      # (10000, 28, 28)
y_test_raw  = test_data.targets.numpy()   # (10000,)

print("x_train shape", x_train_raw.shape)
print("x_test  shape", x_test_raw.shape)

# ── Bước 2: Hiển thị 9 ảnh ngẫu nhiên từ tập train ──────────
print("\n[2] Saving 9 random training images...")
plt.rcParams['figure.figsize'] = (9, 9)
plt.figure()
for i in range(9):
    plt.subplot(3, 3, i + 1)
    num = random.randint(0, len(x_train_raw) - 1)
    plt.imshow(x_train_raw[num].reshape(28, 28), cmap='gray', interpolation=None)
    plt.title('Class {}'.format(y_train_raw[num]))
plt.tight_layout()
plt.savefig('train_samples.png', dpi=120)
plt.close()
print("Saved: train_samples.png")

# ── Bước 3: Chuẩn hóa & flatten ──────────────────────────────
x_train = x_train_raw.reshape(60000, 784).astype('float32') / 255.0
x_test  = x_test_raw.reshape(10000, 784).astype('float32')  / 255.0

print("\n[3] After normalization:")
print("x_train:", x_train.shape, " x_test:", x_test.shape)
print(x_train[:1])

nb_class = 10
def to_categorical(y, num_classes):
    out = np.zeros((len(y), num_classes), dtype='float32')
    for i, val in enumerate(y):
        out[i, int(val)] = 1.0
    return out

y_train_cat = to_categorical(y_train_raw, nb_class)
y_test_cat  = to_categorical(y_test_raw,  nb_class)
print("y_train[0] (label={}): {}".format(y_train_raw[0], y_train_cat[0]))

# ── Bước 4: Xây dựng mô hình ─────────────────────────────────
print("\n[4] Building model...")

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1     = nn.Linear(784, 10)
        self.relu1   = nn.ReLU()
        self.fc2     = nn.Linear(10, 40)
        self.relu2   = nn.ReLU()
        self.fc3     = nn.Linear(40, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

model = MNISTNet().to(DEVICE)

print("\nModel Summary:")
print("-" * 50)
print(f"  Dense(10)  input=784    params = {784*10+10}")
print(f"  Activation ReLU")
print(f"  Dense(40)               params = {10*40+40}")
print(f"  Activation ReLU")
print(f"  Dense(10)               params = {40*10+10}")
print(f"  Activation Softmax")
total = (784*10+10) + (10*40+40) + (40*10+10)
print(f"  Total params: {total}")
print("-" * 50)

# ── Bước 5: Train + Early Stopping theo val_acc ──────────────
print(f"\n[5] Training (batch={BATCH_SIZE}, max_epochs={EPOCHS}, patience={PATIENCE})...")

criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=LR)

x_train_t = torch.tensor(x_train,     dtype=torch.float32)
y_train_t = torch.tensor(y_train_raw, dtype=torch.long)
x_test_t  = torch.tensor(x_test,      dtype=torch.float32)
y_test_t  = torch.tensor(y_test_raw,  dtype=torch.long)

train_loader = DataLoader(TensorDataset(x_train_t, y_train_t),
                          batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(TensorDataset(x_test_t,  y_test_t),
                          batch_size=BATCH_SIZE, shuffle=False)

train_loss_hist = []
train_acc_hist  = []
val_loss_hist   = []
val_acc_hist    = []

# ── Early stopping state (monitor: val_acc ↑) ────────────────
best_val_acc     = 0.0           # lưu acc cao nhất
best_val_loss    = float('inf')  # lưu kèm để hiển thị
patience_counter = 0
best_epoch       = 0

print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7} | Status")
print("-" * 72)

for epoch in range(EPOCHS):

    # ---- Train ----
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        out  = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (out.argmax(1) == yb).sum().item()
        total      += yb.size(0)

    train_loss = total_loss / len(train_loader)
    train_acc  = correct / total
    train_loss_hist.append(train_loss)
    train_acc_hist.append(train_acc)

    # ---- Validation ----
    model.eval()
    val_loss_sum, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            val_loss_sum += criterion(out, yb).item()
            val_correct  += (out.argmax(1) == yb).sum().item()
            val_total    += yb.size(0)

    val_loss = val_loss_sum / len(test_loader)
    val_acc  = val_correct / val_total
    val_loss_hist.append(val_loss)
    val_acc_hist.append(val_acc)

    # ── Kiểm tra cải thiện theo val_acc ──────────────────────
    if val_acc > best_val_acc:          # tăng acc → lưu model
        best_val_acc     = val_acc
        best_val_loss    = val_loss     # lưu kèm loss tương ứng
        best_epoch       = epoch + 1
        patience_counter = 0
        torch.save(model.state_dict(), BEST_MODEL)
        status = '★ BEST'
    else:
        patience_counter += 1           # không tăng → đếm patience
        status = f'no improve ({patience_counter}/{PATIENCE})'

    print(f"{epoch+1:>6} | {train_loss:>10.4f} | {train_acc:>9.4f} | {val_loss:>8.4f} | {val_acc:>7.4f} | {status}")

    # ── Early stopping ────────────────────────────────────────
    if patience_counter >= PATIENCE:
        print(f"\n⚡ Early stopping tại epoch {epoch+1}")
        print(f"   Model tốt nhất: epoch {best_epoch} "
              f"— val_acc={best_val_acc:.4f}, val_loss={best_val_loss:.4f}")
        break
else:
    print(f"\n✓ Hoàn thành {EPOCHS} epochs")
    print(f"  Model tốt nhất: epoch {best_epoch} "
          f"— val_acc={best_val_acc:.4f}, val_loss={best_val_loss:.4f}")

# ── Bước 6: Load model tốt nhất & đánh giá ───────────────────
print(f"\n[6] Loading best model (epoch {best_epoch})...")
model.load_state_dict(torch.load(BEST_MODEL, map_location=DEVICE))
model.eval()

all_preds = []
with torch.no_grad():
    for xb, _ in test_loader:
        out = model(xb.to(DEVICE))
        all_preds.append(out.cpu().numpy())

predicted_classes = np.vstack(all_preds)   # (10000, 10)
test_acc = (predicted_classes.argmax(1) == y_test_raw).mean()
print(f"Test Accuracy (best model): {test_acc:.4f}")

# ── Bước 7: Biểu đồ training history ─────────────────────────
print("\n[7] Saving training history plot...")
epochs_ran = len(train_loss_hist)
x_axis     = range(1, epochs_ran + 1)

fig, ax = plt.subplots(2, 1, figsize=(18, 10))

ax[0].plot(x_axis, train_loss_hist, color='b',  label="Training loss")
ax[0].plot(x_axis, val_loss_hist,   color='r', linestyle='--', label="Val loss")
ax[0].axvline(best_epoch, color='green', linestyle=':', linewidth=1.5,
              label=f'Best epoch {best_epoch}')
ax[0].legend(loc='best', shadow=True)

ax[1].plot(x_axis, train_acc_hist, color='b',  label="Training accuracy")
ax[1].plot(x_axis, val_acc_hist,   color='g', linestyle='--', label="Val accuracy")
ax[1].axvline(best_epoch, color='green', linestyle=':', linewidth=1.5,
              label=f'Best epoch {best_epoch}')
ax[1].legend(loc='best', shadow=True)

plt.suptitle(f'Best epoch: {best_epoch} | Val Acc: {best_val_acc:.4f}', fontsize=12)
plt.tight_layout()
plt.savefig('training_history.png', dpi=120)
plt.close()
print("Saved: training_history.png")

# ── Bước 8: 9 ảnh ngẫu nhiên tập test + nhãn dự báo ─────────
print("\n[8] Saving 9 random test predictions...")
plt.rcParams['figure.figsize'] = (9, 9)
plt.figure()
for i in range(9):
    plt.subplot(3, 3, i + 1)
    num = random.randint(0, len(x_test) - 1)
    plt.imshow(x_test[num].reshape(28, 28), cmap='gray', interpolation=None)
    plt.title('Class {}'.format(predicted_classes[num].argmax()))
plt.tight_layout()
plt.savefig('test_predictions.png', dpi=120)
plt.close()
print("Saved: test_predictions.png")

# ── Bước 9: Lưới 4x4 Real vs Predict ────────────────────────
print("\n[9] Saving real vs predict grid (4x4)...")
x_test_2d = x_test.reshape(x_test.shape[0], 28, 28)

fig, axis = plt.subplots(4, 4, figsize=(12, 14))
for i, ax in enumerate(axis.flat):
    ax.imshow(x_test_2d[i], cmap='binary')
    real_lbl = y_test_raw[i]
    pred_lbl = predicted_classes[i].argmax()
    color    = 'green' if real_lbl == pred_lbl else 'red'
    ax.set_title(
        f"Real Number is {real_lbl}\nPredict Number is {pred_lbl}",
        color=color, fontsize=10
    )
    ax.axis('off')
plt.suptitle('Green = Correct   |   Red = Wrong', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('real_vs_predict.png', dpi=120, bbox_inches='tight')
plt.close()
print("Saved: real_vs_predict.png")

print("\n" + "=" * 55)
print("  Done! Files saved:")
print("  train_samples.png      — 9 anh ngau nhien tap train")
print("  training_history.png   — Loss & Accuracy (train vs val)")
print("  test_predictions.png   — 9 anh test + nhan du bao")
print("  real_vs_predict.png    — luoi 4x4 Real vs Predict")
print(f"  {BEST_MODEL}  — model tot nhat (epoch {best_epoch})")
print("=" * 55)