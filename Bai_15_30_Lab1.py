import pandas as pd
import numpy as np

print("--- Câu 15: Cộng, trừ, nhân, chia 2 Series ---")
s1 = pd.Series([2, 4, 6, 8, 10])
s2 = pd.Series([1, 3, 5, 7, 10])
print("Cộng:\n", s1 + s2)
print("Trừ:\n", s1 - s2)
print("Nhân:\n", s1 * s2)
print("Chia:\n", s1 / s2)

print("\n--- Câu 16: Chuyển cột đầu tiên thành Series ---")
df_16 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
# Lưu ý: Thuộc tính .ix đã bị Pandas loại bỏ hoàn toàn ở các bản mới. 
# Thay vào đó, ta sử dụng .iloc để truy xuất theo vị trí.
series_16 = df_16.iloc[:, 0]
print(type(series_16), "\n", series_16)

print("\n--- Câu 17: Sắp xếp theo cột DataFrame 4x3 ---")
df_17 = pd.DataFrame(np.random.randint(0, 50, size=(4, 3)), columns=['C1', 'C2', 'C3'])
print("Gốc:\n", df_17)
print("Sắp xếp theo C1:\n", df_17.sort_values(by='C1'))

print("\n--- Câu 18: Thay đổi index của Series ---")
s_18 = pd.Series([10, 20, 30], index=['A', 'B', 'C'])
s_18_reindex = s_18.reindex(['B', 'C', 'A'])
print(s_18_reindex)

print("\n--- Câu 19: Phần tử trong sr1 không có trong sr2 ---")
sr1 = pd.Series([1, 2, 3, 4, 5])
sr2 = pd.Series([2, 4, 6, 8, 10])
not_in_sr2 = sr1[~sr1.isin(sr2)]
print("Items of sr1 not present in sr2:\n", not_in_sr2)

print("\n--- Câu 20: numpy.union1d, numpy.intersect1d ---")
x = pd.Series(np.random.randint(1, 20, 20))
y = pd.Series(np.random.randint(1, 20, 20))
print("Giao (intersect1d):", np.intersect1d(x, y))
print("Hội (union1d):", np.union1d(x, y))

print("\n--- Câu 21: Thêm series y vào x (vertical, horizontal) ---")
sx = pd.Series([0, 1, 2, 3])
sy = pd.Series(['p', 'q', 'r', 's'])
# Lưu ý: Hàm append() của Series đã bị gỡ bỏ trong các bản Pandas mới.
# Cần dùng pd.concat() để thay thế.
print("Vertical (nối dọc):\n", pd.concat([sx, sy], axis=0, ignore_index=True))
print("Horizontal (nối ngang thành DataFrame):\n", pd.concat([sx, sy], axis=1))

print("\n--- Câu 22: Tạo DataFrame exam_data ---")
exam_data = {
    'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
    'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
    'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
    'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']
}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
df_exam = pd.DataFrame(exam_data, index=labels)
print(df_exam)

print("\n--- Câu 23: 3 dòng đầu tiên ---")
print(df_exam.head(3))

print("\n--- Câu 24: Chọn cột name và score ---")
print(df_exam[['name', 'score']])

print("\n--- Câu 25: Chọn dòng có attempts > 2 ---")
print(df_exam[df_exam['attempts'] > 2])

print("\n--- Câu 26: Số dòng, số cột ---")
print("Số dòng, số cột:", df_exam.shape)

print("\n--- Câu 27: Các dòng có score là NaN ---")
print(df_exam[df_exam['score'].isnull()])

print("\n--- Câu 28: Dòng có score (15, 20) ---")
print(df_exam[df_exam['score'].between(15, 20)])

print("\n--- Câu 29: Dòng có attempts > 2 VÀ score (15, 20) ---")
print(df_exam[(df_exam['attempts'] > 2) & (df_exam['score'].between(15, 20))])

print("\n--- Câu 30: Đổi giá trị dòng 'd', cột 'score' thành 19 ---")
df_exam.loc['d', 'score'] = 19
print("Kết quả sau thay đổi:\n", df_exam.loc[['d']])