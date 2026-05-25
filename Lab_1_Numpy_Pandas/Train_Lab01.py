import pandas as pd
import numpy as np

# 3. Tải dữ liệu và in 10 dòng đầu, 10 dòng cuối
# Đường dẫn này lấy từ file của bạn trong notebook
url = "https://raw.githubusercontent.com/huynhhoc/DataAnalystDeepLearning/main/Data/dulieuxettuyendaihoc.csv"
df = pd.read_csv(url)

print("--- 10 dòng đầu tiên ---")
print(df.head(10))
print("\n--- 10 dòng cuối cùng ---")
print(df.tail(10))

# 4. Thống kê dữ liệu thiếu cho cột dân tộc (DT) và điền giá trị 0
print(f"\nSố lượng dữ liệu thiếu cột DT: {df['DT'].isnull().sum()}")
df['DT'] = df['DT'].fillna(0)

# 5. Thống kê dữ liệu thiếu cho T1 và thay bằng Mean
print(f"Số lượng dữ liệu thiếu cột T1: {df['T1'].isnull().sum()}")
df['T1'] = df['T1'].fillna(df['T1'].mean())

# 6. Xử lý dữ liệu thiếu cho các biến điểm số còn lại (bằng Mean)
# Lấy danh sách các cột điểm số (T, L, H, S, V, X, D, N từ năm 1, 2, 6 và DH1,2,3)
diem_cols = df.select_dtypes(include=[np.number]).columns
df[diem_cols] = df[diem_cols].fillna(df[diem_cols].mean())

# 7. Tạo biến TBM1, TBM2, TBM3
# Công thức: TBM = (T*2 + L + H + S + V*2 + X + D + N) / 10
for i in ['1', '2', '6']:
    tbm_name = f'TBM{i}' if i != '6' else 'TBM3'
    t = f'T{i}'
    l = f'L{i}'
    h = f'H{i}'
    s = f'S{i}'
    v = f'V{i}'
    x = f'X{i}'
    d = f'D{i}'
    n = f'N{i}'
    df[tbm_name] = (df[t]*2 + df[l] + df[h] + df[s] + df[v]*2 + df[x] + df[d] + df[n]) / 10

# 8. Xếp loại XL1, XL2, XL3
def xep_loai(diem):
    if diem < 5.0: return 'Y'
    elif diem < 6.5: return 'TB'
    elif diem < 8.0: return 'K'
    elif diem < 9.0: return 'G'
    else: return 'XS'

for i in ['1', '2', '3']:
    df[f'XL{i}'] = df[f'TBM{i}'].apply(xep_loai)

# 9. Chuyển sang thang điểm 4 Mỹ (Min-Max Normalization)
# Công thức: (x - min) / (max - min) * (new_max - new_min) + new_min
# Với thang điểm 10 sang 4: (x / 10) * 4
for i in ['1', '2', '3']:
    df[f'US_TBM{i}'] = (df[f'TBM{i}'] / 10) * 4

# 10. Biến kết quả xét tuyển KQXT
def xet_tuyen(row):
    kt = row['KT']
    dh1, dh2, dh3 = row['DH1'], row['DH2'], row['DH3']
    
    if kt in ['A', 'A1']:
        diem = (dh1*2 + dh2 + dh3) / 4
    elif kt == 'B':
        diem = (dh1 + dh2*2 + dh3) / 4
    else:
        diem = (dh1 + dh2 + dh3) / 3
        
    return 1 if diem >= 5.0 else 0

df['KQXT'] = df.apply(xet_tuyen, axis=1)

# 11. Lưu file xuống ổ đĩa
df.to_csv('processed_dulieuxettuyendaihoc.csv', index=False)
print("\nĐã lưu file: processed_dulieuxettuyendaihoc.csv")