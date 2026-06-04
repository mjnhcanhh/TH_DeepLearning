"""
==============================================
BÀI TẬP 5: Flask Web App cho Autoencoder
Triển khai tất cả bài 1-4 trên nền tảng WEB
==============================================
Cài đặt:
  pip install flask tensorflow scikit-learn pillow numpy

Cấu trúc project:
  flask_app/
  ├── app.py               ← file này
  ├── models/
  │   ├── cifar10_ae.h5
  │   ├── catdog_clf.h5
  │   ├── fashion_clf.h5
  │   └── face_clf.h5
  ├── templates/
  │   └── index.html       ← tự tạo khi chạy
  └── static/
      └── style.css

Chạy:
  python app.py
==============================================
"""

import os
import io
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from PIL import Image
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)

# ─────────────────────────────────────────────
# CẤU HÌNH
# ─────────────────────────────────────────────
MODELS_DIR = 'models'

CIFAR10_LABELS = ['airplane','automobile','bird','cat','deer',
                  'dog','frog','horse','ship','truck']
FASHION_LABELS = ['T-shirt/top','Trouser','Pullover','Dress','Coat',
                  'Sandal','Shirt','Sneaker','Bag','Ankle boot']
CATDOG_LABELS  = ['cat','dog']
GENDER_LABELS  = ['female','male']

# ─────────────────────────────────────────────
# LOAD MODELS (lazy loading)
# ─────────────────────────────────────────────
_models = {}

def get_model(name):
    """Lazy-load model theo tên"""
    if name not in _models:
        path = os.path.join(MODELS_DIR, f'{name}.h5')
        if os.path.exists(path):
            _models[name] = keras.models.load_model(path)
            print(f"✅ Loaded model: {name}")
        else:
            print(f"⚠️  Model {name} not found at {path}")
            _models[name] = None
    return _models[name]

# ─────────────────────────────────────────────
# HELPER: xử lý ảnh từ upload
# ─────────────────────────────────────────────
def preprocess_image(file, target_size, grayscale=False):
    """Đọc file upload, resize, normalize → numpy array"""
    img = Image.open(file)
    if grayscale:
        img = img.convert('L')
    else:
        img = img.convert('RGB')
    img = img.resize(target_size)
    arr = np.array(img, dtype='float32') / 255.0
    if grayscale:
        arr = arr[..., np.newaxis]
    return arr[np.newaxis, ...]   # batch dim

def image_to_base64(arr):
    """Chuyển numpy array → base64 để hiển thị HTML"""
    # arr: (H, W, C) hoặc (H, W, 1)
    if arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    arr = (arr * 255).astype(np.uint8)
    if arr.ndim == 2:
        img = Image.fromarray(arr, 'L')
    else:
        img = Image.fromarray(arr, 'RGB')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()

# ─────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────

@app.route('/api/predict/cifar10', methods=['POST'])
def predict_cifar10():
    """Bài 1: CIFAR10 - 10 classes"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    model = get_model('cifar10_ae')
    if model is None:
        return jsonify({'error': 'Model not found. Train and save model first.'}), 503

    img = preprocess_image(request.files['image'], (32, 32))
    pred = model.predict(img)               # reconstruction
    reconstructed_b64 = image_to_base64(pred[0])

    # Nếu có classifier riêng
    clf = get_model('cifar10_clf')
    if clf:
        probs = clf.predict(img)[0]
        label_idx = int(np.argmax(probs))
        label = CIFAR10_LABELS[label_idx]
        confidence = float(probs[label_idx])
    else:
        label = "N/A (train classifier)"
        confidence = 0.0
        probs = [0.0] * 10

    return jsonify({
        'label': label,
        'confidence': round(confidence * 100, 2),
        'probabilities': {CIFAR10_LABELS[i]: round(float(probs[i])*100, 2) for i in range(10)},
        'reconstructed': reconstructed_b64
    })


@app.route('/api/predict/catdog', methods=['POST'])
def predict_catdog():
    """Bài 2: Cat vs Dog"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    ae  = get_model('catdog_ae')
    clf = get_model('catdog_clf')
    if clf is None:
        return jsonify({'error': 'Model not found'}), 503

    img = preprocess_image(request.files['image'], (64, 64))
    prob = float(clf.predict(img)[0][0])
    label_idx = int(prob > 0.5)
    label = CATDOG_LABELS[label_idx]

    reconstructed_b64 = None
    if ae:
        rec = ae.predict(img)
        reconstructed_b64 = image_to_base64(rec[0])

    return jsonify({
        'label': label,
        'confidence': round(max(prob, 1-prob) * 100, 2),
        'probabilities': {
            'cat': round((1-prob)*100, 2),
            'dog': round(prob*100, 2)
        },
        'reconstructed': reconstructed_b64
    })


@app.route('/api/predict/fashion', methods=['POST'])
def predict_fashion():
    """Bài 3: Fashion-MNIST"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    ae  = get_model('fashion_ae')
    clf = get_model('fashion_clf')
    if clf is None:
        return jsonify({'error': 'Model not found'}), 503

    img = preprocess_image(request.files['image'], (28, 28), grayscale=True)
    probs = clf.predict(img)[0]
    label_idx = int(np.argmax(probs))
    label = FASHION_LABELS[label_idx]

    reconstructed_b64 = None
    if ae:
        rec = ae.predict(img)
        reconstructed_b64 = image_to_base64(rec[0])

    return jsonify({
        'label': label,
        'confidence': round(float(probs[label_idx]) * 100, 2),
        'probabilities': {FASHION_LABELS[i]: round(float(probs[i])*100, 2) for i in range(10)},
        'reconstructed': reconstructed_b64
    })


@app.route('/api/predict/gender', methods=['POST'])
def predict_gender():
    """Bài 4: Face Gender Nam/Nữ"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    ae  = get_model('face_ae')
    clf = get_model('face_clf')
    if clf is None:
        return jsonify({'error': 'Model not found'}), 503

    img = preprocess_image(request.files['image'], (64, 64))
    prob = float(clf.predict(img)[0][0])
    label_idx = int(prob > 0.5)
    label = GENDER_LABELS[label_idx]

    reconstructed_b64 = None
    if ae:
        rec = ae.predict(img)
        reconstructed_b64 = image_to_base64(rec[0])

    return jsonify({
        'label': label,
        'confidence': round(max(prob, 1-prob) * 100, 2),
        'probabilities': {
            'female': round((1-prob)*100, 2),
            'male': round(prob*100, 2)
        },
        'reconstructed': reconstructed_b64
    })


# ─────────────────────────────────────────────
# TRANG WEB CHÍNH
# ─────────────────────────────────────────────
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Autoencoder Demo - Thực Hành 5</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Segoe UI', sans-serif;
    background: #0f172a;
    color: #e2e8f0;
    min-height: 100vh;
  }

  header {
    background: linear-gradient(135deg, #1e3a5f, #0f766e);
    padding: 24px 32px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
  }
  header h1 { font-size: 2rem; font-weight: 700; color: #fff; }
  header p  { color: #94a3b8; margin-top: 6px; }

  .tabs {
    display: flex;
    justify-content: center;
    gap: 8px;
    padding: 20px 16px 0;
    flex-wrap: wrap;
  }
  .tab-btn {
    background: #1e293b;
    border: 2px solid #334155;
    color: #94a3b8;
    padding: 10px 20px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 0.9rem;
    transition: all .2s;
  }
  .tab-btn:hover, .tab-btn.active {
    background: #0f766e;
    border-color: #0f766e;
    color: #fff;
  }

  .tab-content { display: none; }
  .tab-content.active { display: block; }

  .card {
    background: #1e293b;
    border-radius: 14px;
    padding: 28px;
    max-width: 820px;
    margin: 20px auto;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
  }
  .card h2 {
    font-size: 1.25rem;
    color: #38bdf8;
    margin-bottom: 6px;
  }
  .card p.desc {
    color: #64748b;
    font-size: 0.88rem;
    margin-bottom: 18px;
  }

  .upload-zone {
    border: 2px dashed #334155;
    border-radius: 10px;
    padding: 30px;
    text-align: center;
    cursor: pointer;
    transition: border-color .2s;
  }
  .upload-zone:hover { border-color: #0f766e; }
  .upload-zone input { display: none; }
  .upload-zone label {
    cursor: pointer;
    color: #38bdf8;
    font-size: 0.95rem;
  }

  .preview-row {
    display: flex;
    gap: 16px;
    margin: 16px 0;
    justify-content: center;
    flex-wrap: wrap;
  }
  .preview-box {
    text-align: center;
    flex: 1; min-width: 120px; max-width: 200px;
  }
  .preview-box span {
    display: block;
    font-size: 0.75rem;
    color: #64748b;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 1px;
  }
  .preview-box img {
    width: 100%; border-radius: 8px;
    border: 2px solid #334155;
    image-rendering: pixelated;
  }

  .btn-predict {
    width: 100%;
    padding: 12px;
    background: linear-gradient(90deg, #0f766e, #0369a1);
    color: white;
    border: none;
    border-radius: 8px;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: opacity .2s;
    margin-top: 8px;
  }
  .btn-predict:hover { opacity: 0.88; }
  .btn-predict:disabled { opacity: 0.5; cursor: not-allowed; }

  .result-box {
    margin-top: 20px;
    background: #0f172a;
    border-radius: 10px;
    padding: 18px;
    display: none;
  }
  .result-label {
    font-size: 1.6rem;
    font-weight: 700;
    color: #34d399;
    text-align: center;
    margin-bottom: 4px;
  }
  .result-conf {
    text-align: center;
    color: #94a3b8;
    font-size: 0.9rem;
    margin-bottom: 14px;
  }

  .prob-bar-container { margin: 4px 0; }
  .prob-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    color: #cbd5e1;
    margin-bottom: 2px;
  }
  .prob-bar-track {
    background: #334155;
    border-radius: 4px;
    height: 10px;
    overflow: hidden;
  }
  .prob-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #0f766e, #38bdf8);
    border-radius: 4px;
    transition: width .4s ease;
  }

  .loading {
    text-align: center;
    color: #38bdf8;
    padding: 12px;
    display: none;
  }
  .spinner {
    display: inline-block;
    width: 24px; height: 24px;
    border: 3px solid #334155;
    border-top-color: #38bdf8;
    border-radius: 50%;
    animation: spin .8s linear infinite;
    margin-right: 8px;
    vertical-align: middle;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  footer {
    text-align: center;
    padding: 24px;
    color: #475569;
    font-size: 0.8rem;
  }
</style>
</head>
<body>

<header>
  <h1>🧠 Autoencoder Image Recognition</h1>
  <p>Thực Hành 5 – Convolutional Autoencoder Demo</p>
</header>

<div class="tabs">
  <button class="tab-btn active" onclick="switchTab('cifar10', this)">📦 CIFAR-10</button>
  <button class="tab-btn" onclick="switchTab('catdog',  this)">🐱🐶 Cat vs Dog</button>
  <button class="tab-btn" onclick="switchTab('fashion', this)">👗 Fashion-MNIST</button>
  <button class="tab-btn" onclick="switchTab('gender',  this)">👤 Face Gender</button>
</div>

<!-- TAB 1: CIFAR10 -->
<div id="tab-cifar10" class="tab-content active">
  <div class="card">
    <h2>📦 CIFAR-10 – Bài Tập 1</h2>
    <p class="desc">Nhận dạng 10 loại ảnh: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck</p>
    <div class="upload-zone" id="zone-cifar10">
      <input type="file" id="file-cifar10" accept="image/*" onchange="previewImage('cifar10')">
      <label for="file-cifar10">📁 Click để chọn ảnh hoặc kéo thả vào đây</label>
    </div>
    <div class="preview-row" id="preview-cifar10"></div>
    <button class="btn-predict" onclick="predict('cifar10')" id="btn-cifar10">🔍 Dự Đoán</button>
    <div class="loading" id="loading-cifar10"><span class="spinner"></span>Đang xử lý...</div>
    <div class="result-box" id="result-cifar10"></div>
  </div>
</div>

<!-- TAB 2: Cat vs Dog -->
<div id="tab-catdog" class="tab-content">
  <div class="card">
    <h2>🐱🐶 Cat vs Dog – Bài Tập 2</h2>
    <p class="desc">Phân loại ảnh chó hoặc mèo</p>
    <div class="upload-zone">
      <input type="file" id="file-catdog" accept="image/*" onchange="previewImage('catdog')">
      <label for="file-catdog">📁 Click để chọn ảnh</label>
    </div>
    <div class="preview-row" id="preview-catdog"></div>
    <button class="btn-predict" onclick="predict('catdog')" id="btn-catdog">🔍 Dự Đoán</button>
    <div class="loading" id="loading-catdog"><span class="spinner"></span>Đang xử lý...</div>
    <div class="result-box" id="result-catdog"></div>
  </div>
</div>

<!-- TAB 3: Fashion -->
<div id="tab-fashion" class="tab-content">
  <div class="card">
    <h2>👗 Fashion-MNIST – Bài Tập 3</h2>
    <p class="desc">Nhận dạng 10 loại thời trang: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot</p>
    <div class="upload-zone">
      <input type="file" id="file-fashion" accept="image/*" onchange="previewImage('fashion')">
      <label for="file-fashion">📁 Click để chọn ảnh</label>
    </div>
    <div class="preview-row" id="preview-fashion"></div>
    <button class="btn-predict" onclick="predict('fashion')" id="btn-fashion">🔍 Dự Đoán</button>
    <div class="loading" id="loading-fashion"><span class="spinner"></span>Đang xử lý...</div>
    <div class="result-box" id="result-fashion"></div>
  </div>
</div>

<!-- TAB 4: Gender -->
<div id="tab-gender" class="tab-content">
  <div class="card">
    <h2>👤 Face Gender – Bài Tập 4</h2>
    <p class="desc">Nhận dạng giới tính khuôn mặt: Nam (Male) hoặc Nữ (Female)</p>
    <div class="upload-zone">
      <input type="file" id="file-gender" accept="image/*" onchange="previewImage('gender')">
      <label for="file-gender">📁 Click để chọn ảnh khuôn mặt</label>
    </div>
    <div class="preview-row" id="preview-gender"></div>
    <button class="btn-predict" onclick="predict('gender')" id="btn-gender">🔍 Dự Đoán</button>
    <div class="loading" id="loading-gender"><span class="spinner"></span>Đang xử lý...</div>
    <div class="result-box" id="result-gender"></div>
  </div>
</div>

<footer>Thực Hành 5: Autoencoder | Deep Learning | TensorFlow + Flask</footer>

<script>
function switchTab(name, btn) {
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  btn.classList.add('active');
}

function previewImage(task) {
  const file = document.getElementById('file-' + task).files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {
    const box = document.getElementById('preview-' + task);
    box.innerHTML = `
      <div class="preview-box">
        <span>Ảnh gốc</span>
        <img src="${e.target.result}" alt="preview">
      </div>`;
  };
  reader.readAsDataURL(file);
}

async function predict(task) {
  const fileInput = document.getElementById('file-' + task);
  if (!fileInput.files[0]) {
    alert('Vui lòng chọn ảnh trước!'); return;
  }

  const btn     = document.getElementById('btn-' + task);
  const loading = document.getElementById('loading-' + task);
  const result  = document.getElementById('result-' + task);

  btn.disabled = true;
  loading.style.display = 'block';
  result.style.display  = 'none';

  const formData = new FormData();
  formData.append('image', fileInput.files[0]);

  try {
    const res  = await fetch('/api/predict/' + task, { method: 'POST', body: formData });
    const data = await res.json();

    if (data.error) {
      result.innerHTML = `<p style="color:#f87171;text-align:center">${data.error}</p>`;
    } else {
      // Build probability bars
      let barsHTML = '';
      const sorted = Object.entries(data.probabilities).sort((a,b) => b[1]-a[1]);
      sorted.forEach(([lbl, prob]) => {
        barsHTML += `
          <div class="prob-bar-container">
            <div class="prob-bar-label"><span>${lbl}</span><span>${prob}%</span></div>
            <div class="prob-bar-track">
              <div class="prob-bar-fill" style="width:${Math.min(prob,100)}%"></div>
            </div>
          </div>`;
      });

      // Reconstruction image
      let reconHTML = '';
      if (data.reconstructed) {
        reconHTML = `
          <div class="preview-box">
            <span>Ảnh tái tạo</span>
            <img src="data:image/png;base64,${data.reconstructed}">
          </div>`;
        document.getElementById('preview-' + task).innerHTML += reconHTML;
      }

      result.innerHTML = `
        <div class="result-label">${data.label.toUpperCase()}</div>
        <div class="result-conf">Độ tin cậy: ${data.confidence}%</div>
        ${barsHTML}`;
    }
    result.style.display = 'block';
  } catch (err) {
    result.innerHTML = `<p style="color:#f87171;text-align:center">Lỗi kết nối: ${err.message}</p>`;
    result.style.display = 'block';
  } finally {
    btn.disabled = false;
    loading.style.display = 'none';
  }
}
</script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

# ─────────────────────────────────────────────
# SCRIPT XUẤT MODEL (chạy sau khi train)
# ─────────────────────────────────────────────
def save_models_from_training():
    """
    Gọi hàm này sau khi train xong để lưu model vào thư mục models/
    Ví dụ:
        from bai1_autoencoder_cifar10 import autoencoder, encoder
        os.makedirs('models', exist_ok=True)
        autoencoder.save('models/cifar10_ae.h5')
        # Nếu có classifier:
        # classifier.save('models/cifar10_clf.h5')
    """
    print("Xem file README_SAVE_MODELS.txt để biết cách lưu model")


if __name__ == '__main__':
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("=" * 50)
    print("🌐 Flask Autoencoder Demo")
    print("   Truy cập: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
