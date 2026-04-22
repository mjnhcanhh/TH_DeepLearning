import numpy as np

print("--- Câu 1 & 2: np.all() và np.any() ---")
# Lưu ý: Yêu cầu ghi "20 số nguyên" nhưng kích thước "3x3" (chỉ có 9 phần tử) là mâu thuẫn. 
# Tôi sẽ tạo mảng 3x3 ngẫu nhiên trong khoảng [0, 20]
arr_1 = np.random.randint(0, 21, size=(3, 3))
print("Mảng 3x3:\n", arr_1)
print("1. Tất cả phần tử đều khác 0? :", np.all(arr_1 != 0))
print("2. Có tồn tại phần tử khác 0? :", np.any(arr_1 != 0))

print("\n--- Câu 3: So sánh mảng (greater, less...) ---")
arr_1d_a = np.array([1, 5, 10])
arr_1d_b = np.array([2, 5, 8])
print("Greater (a > b):", np.greater(arr_1d_a, arr_1d_b))
print("Greater_equal (a >= b):", np.greater_equal(arr_1d_a, arr_1d_b))
print("Less (a < b):", np.less(arr_1d_a, arr_1d_b))
print("Less_equal (a <= b):", np.less_equal(arr_1d_a, arr_1d_b))

print("\n--- Câu 4: Mảng 10 số 1, 10 số 0, 10 số 5 ---")
arr_4 = np.concatenate((np.ones(10), np.zeros(10), np.full(10, 5)))
print(arr_4)

print("\n--- Câu 5: Mảng số nguyên chẵn từ [30, 70] ---")
arr_5 = np.arange(30, 72, 2)
print(arr_5)

print("\n--- Câu 6: Ma trận đơn vị 3x3 ---")
arr_6 = np.identity(3)
print(arr_6)

print("\n--- Câu 7: 10 phần tử [15, 55], in trừ đầu cuối ---")
arr_7 = np.linspace(15, 55, 10)
print("Bỏ đầu cuối:", arr_7[1:-1])

print("\n--- Câu 8: Mảng 20 phần tử [0, 20], đổi dấu trong [9, 15] ---")
arr_8 = np.arange(0, 20)
arr_8[(arr_8 >= 9) & (arr_8 <= 15)] *= -1
print(arr_8)

print("\n--- Câu 9: Ma trận 3x4, giá trị [10, 21] ---")
arr_9 = np.random.randint(10, 22, size=(3, 4))
print(arr_9)

print("\n--- Câu 10: Ma trận 10x10, viền 1, trong 0 ---")
arr_10 = np.ones((10, 10))
arr_10[1:-1, 1:-1] = 0
print(arr_10)

print("\n--- Câu 11: Ma trận 5x5, đường chéo chính 1,2,3,4,5 ---")
arr_11 = np.diag([1, 2, 3, 4, 5])
print(arr_11)

print("\n--- Câu 12: Mảng 3x3x3, tính tổng ---")
arr_12 = np.random.randint(1, 10, size=(3, 3, 3))
print("Tổng theo cột (axis=1):\n", np.sum(arr_12, axis=1))
print("Tổng theo dòng (axis=2):\n", np.sum(arr_12, axis=2))

print("\n--- Câu 13: 2 vector ngẫu nhiên 10 phần tử, inner product ---")
v1 = np.random.randint(1, 10, 10)
v2 = np.random.randint(1, 10, 10)
print("Tích vô hướng (dot):", np.dot(v1, v2))

print("\n--- Câu 14: Thêm vector y vào từng dòng ma trận A ---")
A = np.random.randint(1, 10, size=(4, 3))
y = np.random.randint(1, 10, size=3)
print("Ma trận A:\n", A)
print("Vector y:", y)
print("Kết quả (A + y nhờ cơ chế Broadcasting):\n", A + y)