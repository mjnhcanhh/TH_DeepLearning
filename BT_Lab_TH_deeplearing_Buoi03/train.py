# ============================================================
# CNN Feature Map Visualization
# Theo yêu cầu: cnn_feature_map_colab.ipynb
# + Early Stopping: lưu model tốt nhất (loss thấp nhất / acc cao nhất)
# ============================================================

# ── Bước 1: Import thư viện ──────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())


# ── Bước 2: Cấu hình tham số ─────────────────────────────────
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE    = 64
EPOCHS        = 100       # tăng lên để early stopping phát huy
LEARNING_RATE = 0.001
NUM_CLASSES   = 10

# Early Stopping
MONITOR       = 'val_loss'   # 'val_loss' hoặc 'val_acc'
PATIENCE      = 5            # dừng nếu không cải thiện sau 5 epoch liên tiếp
BEST_MODEL    = 'best_model.pth'

print('Using device:', DEVICE)
print(f'Monitor: {MONITOR} | Patience: {PATIENCE}')


# ── Bước 3: Load dataset digits ──────────────────────────────
digits = load_digits()

X = digits.images / 16.0
X = X.astype(np.float32)[:, None, :, :]
y = digits.target.astype(np.int64)

print('X shape:', X.shape)
print('y shape:', y.shape)
print('Classes:', np.unique(y))


# ── Bước 4: Hiển thị một vài ảnh mẫu ────────────────────────
plt.figure(figsize=(10, 3))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X[i].squeeze(), cmap='gray')
    plt.title(f'Label: {y[i]}')
    plt.axis('off')
plt.tight_layout()
plt.savefig('sample_images.png', dpi=120, bbox_inches='tight')
plt.close()
print('Saved: sample_images.png')


# ── Bước 5: Chia train/test và tạo DataLoader ────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test  = torch.tensor(y_test,  dtype=torch.long)

train_dataset = TensorDataset(X_train, y_train)
test_dataset  = TensorDataset(X_test,  y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print('Train size:', len(train_dataset))
print('Test size:', len(test_dataset))


# ── Bước 6: Xây dựng mô hình CNN ─────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1   = nn.Linear(16 * 2 * 2, 32)
        self.relu3 = nn.ReLU()
        self.fc2   = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.conv1(x);  x = self.relu1(x);  x = self.pool1(x)
        x = self.conv2(x);  x = self.relu2(x);  x = self.pool2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.relu3(self.fc1(x))
        return self.fc2(x)


model = SimpleCNN(num_classes=NUM_CLASSES).to(DEVICE)
print(model)


# ── Bước 7: Train CNN + Early Stopping ───────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Lịch sử
train_loss_history = []
train_acc_history  = []
val_loss_history   = []
val_acc_history    = []

# Early stopping state
best_val_loss = float('inf')
best_val_acc  = 0.0
patience_counter = 0
best_epoch    = 0

print(f'\n{"Epoch":>6} | {"Train Loss":>10} | {"Train Acc":>9} | {"Val Loss":>8} | {"Val Acc":>7} | {"Status"}')
print('-' * 70)

for epoch in range(EPOCHS):

    # ---- Training ----
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss    = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct    += (torch.argmax(outputs, 1) == y_batch).sum().item()
        total      += y_batch.size(0)

    train_loss = total_loss / len(train_loader)
    train_acc  = correct / total
    train_loss_history.append(train_loss)
    train_acc_history.append(train_acc)

    # ---- Validation (trên test set) ----
    model.eval()
    val_loss_sum, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs   = model(X_batch)
            val_loss_sum += criterion(outputs, y_batch).item()
            val_correct  += (torch.argmax(outputs, 1) == y_batch).sum().item()
            val_total    += y_batch.size(0)

    val_loss = val_loss_sum / len(test_loader)
    val_acc  = val_correct / val_total
    val_loss_history.append(val_loss)
    val_acc_history.append(val_acc)

    # ---- Kiểm tra cải thiện & lưu model tốt nhất ----
    improved = False

    if MONITOR == 'val_loss' and val_loss < best_val_loss:
        best_val_loss    = val_loss
        best_val_acc     = val_acc
        best_epoch       = epoch + 1
        patience_counter = 0
        improved         = True
        torch.save(model.state_dict(), BEST_MODEL)

    elif MONITOR == 'val_acc' and val_acc > best_val_acc:
        best_val_acc     = val_acc
        best_val_loss    = val_loss
        best_epoch       = epoch + 1
        patience_counter = 0
        improved         = True
        torch.save(model.state_dict(), BEST_MODEL)

    else:
        patience_counter += 1

    status = '★ BEST' if improved else f'no improve ({patience_counter}/{PATIENCE})'

    print(f'{epoch+1:>6} | {train_loss:>10.4f} | {train_acc:>9.4f} | {val_loss:>8.4f} | {val_acc:>7.4f} | {status}')

    # ---- Early stopping ----
    if patience_counter >= PATIENCE:
        print(f'\n⚡ Early stopping tại epoch {epoch+1}')
        print(f'   Model tốt nhất tại epoch {best_epoch} — val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f}')
        break

else:
    print(f'\n✓ Hoàn thành {EPOCHS} epochs')
    print(f'  Model tốt nhất tại epoch {best_epoch} — val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f}')


# ── Bước 8: Load lại model tốt nhất & đánh giá ───────────────
model.load_state_dict(torch.load(BEST_MODEL, map_location=DEVICE))
model.eval()

correct, total = 0, 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        correct += (model(X_batch).argmax(1) == y_batch).sum().item()
        total   += y_batch.size(0)

print(f'\nTest Accuracy (best model): {correct/total:.4f}')
print(f'Saved: {BEST_MODEL}')


# ── Biểu đồ training curves ───────────────────────────────────
epochs_ran = len(train_loss_history)
x_axis     = range(1, epochs_ran + 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss
axes[0].plot(x_axis, train_loss_history, 'b-',  label='Train Loss')
axes[0].plot(x_axis, val_loss_history,   'r--', label='Val Loss')
if best_epoch:
    axes[0].axvline(best_epoch, color='green', linestyle=':', linewidth=1.5,
                    label=f'Best epoch {best_epoch}')
axes[0].set_title('Loss')
axes[0].set_xlabel('Epoch')
axes[0].legend()

# Accuracy
axes[1].plot(x_axis, train_acc_history, 'b-',  label='Train Acc')
axes[1].plot(x_axis, val_acc_history,   'g--', label='Val Acc')
if best_epoch:
    axes[1].axvline(best_epoch, color='green', linestyle=':', linewidth=1.5,
                    label=f'Best epoch {best_epoch}')
axes[1].set_title('Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].legend()

plt.suptitle(f'Monitor: {MONITOR} | Best epoch: {best_epoch} | Val Acc: {best_val_acc:.4f}',
             fontsize=11)
plt.tight_layout()
plt.savefig('training_curves.png', dpi=120, bbox_inches='tight')
plt.close()
print('Saved: training_curves.png')


# ── Bước 9–10: Hàm lấy và vẽ feature map ────────────────────
def get_feature_maps(model, image_tensor):
    model.eval()
    feature_maps = {}
    with torch.no_grad():
        x = image_tensor.to(DEVICE)
        x = model.conv1(x);  feature_maps['conv1'] = x.cpu()
        x = model.relu1(x);  feature_maps['relu1'] = x.cpu()
        x = model.pool1(x);  feature_maps['pool1'] = x.cpu()
        x = model.conv2(x);  feature_maps['conv2'] = x.cpu()
        x = model.relu2(x);  feature_maps['relu2'] = x.cpu()
        x = model.pool2(x);  feature_maps['pool2'] = x.cpu()
    return feature_maps


def plot_feature_maps(feature_maps, original_image, true_label, max_channels=8):
    num_layers = len(feature_maps)
    plt.figure(figsize=(14, 3 * (num_layers + 1)))

    plt.subplot(num_layers + 1, max_channels, 1)
    plt.imshow(original_image.squeeze(), cmap='gray')
    plt.title(f'Original\nLabel: {true_label}')
    plt.axis('off')

    for row, (layer_name, fmap) in enumerate(feature_maps.items(), start=1):
        fmap = fmap.squeeze(0)
        for i in range(min(fmap.shape[0], max_channels)):
            plt.subplot(num_layers + 1, max_channels, row * max_channels + i + 1)
            plt.imshow(fmap[i], cmap='viridis')
            plt.title(f'{layer_name}\nch {i}')
            plt.axis('off')

    plt.tight_layout()
    plt.savefig('feature_maps.png', dpi=130, bbox_inches='tight')
    plt.close()


# ── Bước 11: Chọn ảnh mẫu và hiển thị feature map ───────────
sample_index = 0
sample_image = X_test[sample_index:sample_index + 1]
sample_label = y_test[sample_index].item()

feature_maps = get_feature_maps(model, sample_image)
for layer_name, fmap in feature_maps.items():
    print(layer_name, 'shape:', tuple(fmap.shape))

plot_feature_maps(
    feature_maps,
    original_image=sample_image.numpy(),
    true_label=sample_label,
    max_channels=8
)
print('Saved: feature_maps.png')


# ── Bước 12: Giải thích ───────────────────────────────────────
# conv1 : tạo bản đồ đặc trưng từ ảnh gốc, mỗi channel = 1 filter.
# relu1 : loại bỏ giá trị âm, giữ vùng kích hoạt mạnh.
# pool1 : giảm kích thước 8x8 → 4x4, giữ thông tin nổi bật.
# conv2 : học đặc trưng sâu hơn từ feature map tầng trước.
# relu2 : tiếp tục giữ vùng kích hoạt quan trọng.
# pool2 : giảm kích thước 4x4 → 2x2 để đưa vào fully connected.

print('\n=== Done! Files saved ===')
print('  sample_images.png   — 10 anh mau chu so 0-9')
print('  training_curves.png — bieu do Loss & Accuracy (train vs val)')
print('  feature_maps.png    — feature map qua tung layer')
print(f'  {BEST_MODEL:<20} — trong so model tot nhat (epoch {best_epoch})')