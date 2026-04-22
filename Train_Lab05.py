import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def main():
    # ==========================================
    # BƯỚC 1: ĐỌC DỮ LIỆU TỪ GITHUB
    # ==========================================
    url = "https://raw.githubusercontent.com/huynhhoc/DataAnalystDeepLearning/main/Data/dulieuxettuyendaihoc.csv"
    print("1. Đang tải dữ liệu từ GitHub...")
    try:
        df = pd.read_csv(url)
        print("=> Tải dữ liệu thành công!\n")
    except Exception as e:
        print(f"Lỗi tải dữ liệu: {e}")
        return

    # Lọc bỏ các dòng bị thiếu nhãn (target) để không bị lỗi lúc vẽ hình và train
    features = ['T1', 'T2', 'T3', 'GT', 'KV']
    target = 'DH1'
    df = df.dropna(subset=[target, 'T1', 'GT', 'KV']) 

    # ==========================================
    # BƯỚC 2: TRỰC QUAN HÓA DỮ LIỆU (Giống Lab 05)
    # ==========================================
    print("2. Đang tạo biểu đồ trực quan hóa dữ liệu (EDA)...")
    print("=> [QUAN TRỌNG]: Hãy xem biểu đồ, sau đó ĐÓNG CỬA SỔ BIỂU ĐỒ LẠI để code tiếp tục chạy phần Train!\n")
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 10))

    # Biểu đồ 1: Histogram
    plt.subplot(2, 2, 1)
    sns.histplot(df['T1'], kde=True, color='royalblue', bins=15)
    plt.title('Phân phối điểm Toán HK1 (T1)')

    # Biểu đồ 2: Boxplot
    plt.subplot(2, 2, 2)
    sns.boxplot(x='GT', y='T1', hue='GT', data=df, palette='Set2', legend=False)
    plt.title('Điểm T1 theo Giới tính')

    # Biểu đồ 3: Scatter
    plt.subplot(2, 2, 3)
    sns.scatterplot(x='T1', y='DH1', hue='KV', data=df, palette='Dark2')
    plt.title('Tương quan giữa Điểm T1 và Điểm DH1')

    # Biểu đồ 4: Countplot
    plt.subplot(2, 2, 4)
    sns.countplot(x='KV', hue='KV', data=df, palette='pastel', legend=False)
    plt.title('Số lượng học sinh theo Khu vực')

    plt.tight_layout()
    plt.show() # Dừng code tại đây chờ user đóng cửa sổ hình

    # ==========================================
    # BƯỚC 3: TIỀN XỬ LÝ & HUẤN LUYỆN MÔ HÌNH
    # ==========================================
    print("3. Bắt đầu tiền xử lý dữ liệu (chuẩn hóa, mã hóa) và huấn luyện mô hình...")
    X = df[features]
    y = df[target]

    # Phân loại cột
    numeric_features = ['T1', 'T2', 'T3']
    categorical_features = ['GT', 'KV']

    # Tiền xử lý Số (Điền khuyết bằng mean + Chuẩn hóa)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Tiền xử lý Chữ (Điền khuyết bằng mode + Mã hóa One-hot)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Gộp bộ tiền xử lý
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Khởi tạo mô hình
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # Chia data và Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\n" + "="*40)
    print("KẾT QUẢ HUẤN LUYỆN MÔ HÌNH")
    print("="*40)
    print(f"Sai số MSE       : {mean_squared_error(y_test, y_pred):.4f}")
    print(f"Độ chính xác (R2): {r2_score(y_test, y_pred):.4f}")
    print("="*40)

    # ==========================================
    # BƯỚC 4: VẼ BIỂU ĐỒ ĐÁNH GIÁ MÔ HÌNH
    # ==========================================
    print("\n4. Đang hiển thị biểu đồ đánh giá kết quả dự đoán...")
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='crimson', alpha=0.6, label='Điểm dự đoán')
    
    # Vẽ đường chuẩn (Đường thẳng lý tưởng nếu dự đoán đúng 100%)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Đường chuẩn (Hoàn hảo)')
    
    plt.xlabel('Điểm Thực Tế (Actual DH1)')
    plt.ylabel('Điểm Dự Đoán (Predicted DH1)')
    plt.title('Đánh giá Mô hình: Thực tế vs Dự đoán')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()