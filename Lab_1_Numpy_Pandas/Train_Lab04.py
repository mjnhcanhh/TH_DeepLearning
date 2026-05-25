import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pickle
import os

def train_model():
    print("--- Bước 2: Đang bắt đầu huấn luyện ---")
    
    # KIỂM TRA FILE: Nếu chưa có thì tự đi lấy dữ liệu gốc về xử lý luôn
    if not os.path.exists("processed_data.csv"):
        print("[!] Không thấy file processed_data.csv, đang tiến hành xử lý dữ liệu gốc...")
        df_raw = pd.read_csv("https://raw.githubusercontent.com/huynhhoc/DataAnalystDeepLearning/main/Data/dulieuxettuyendaihoc.csv")
        # Xử lý nhanh
        df_raw['T1'] = df_raw['T1'].fillna(df_raw['T1'].mean())
        df_raw['KQXT'] = ((df_raw['DH1'] + df_raw['DH2'] + df_raw['DH3']) / 3 >= 5.0).astype(int)
        df_raw.to_csv("processed_data.csv", index=False)
        print("[OK] Đã tạo xong file processed_data.csv")

    # Đọc dữ liệu để train
    df = pd.read_csv("processed_data.csv")
    
    # Chọn biến (Features)
    # Lưu ý: Cảnh có thể dùng TBM1, TBM2, TBM3 nếu đã tạo ở Lab 01
    # Ở đây mình dùng tạm T1, L1, H1 để test
    X = df[['T1', 'L1', 'H1']] 
    y = df['KQXT']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Tạo thư mục models nếu chưa có
    if not os.path.exists('models'): os.makedirs('models')
    
    with open('models/logistic_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Lưu dữ liệu test để tí nữa file evaluate.py dùng
    X_test.to_csv("X_test.csv", index=False)
    y_test.to_csv("y_test.csv", index=False)
    
    print(f"--- HUẤN LUYỆN HOÀN TẤT ---")
    print(f"Độ chính xác: {model.score(X_train, y_train)*100:.2f}%")

if __name__ == "__main__":
    train_model()