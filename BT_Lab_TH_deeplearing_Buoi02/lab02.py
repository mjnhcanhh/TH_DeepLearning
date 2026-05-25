# LAB02 - Code chạy trên VS Code: vừa IN KẾT QUẢ RA TERMINAL (màn hình đen), vừa XUẤT ẢNH BIỂU ĐỒ
# Cách chạy:
#   python lab02_vscode_day_du_terminal_va_anh.py
# Lưu ý: Đặt file này cùng thư mục với dulieuxettuyendaihoc.csv

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# Cho pandas in bảng dễ nhìn hơn trên terminal
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 50)

# =========================
# 1. ĐỌC DỮ LIỆU
# =========================
local_file = 'dulieuxettuyendaihoc.csv'
url_file = 'https://raw.githubusercontent.com/huynhhoc/DataAnalystDeepLearning/main/Data/dulieuxettuyendaihoc.csv'

if os.path.exists(local_file):
    df = pd.read_csv(local_file)
elif os.path.exists('/mnt/data/dulieuxettuyendaihoc.csv'):
    df = pd.read_csv('/mnt/data/dulieuxettuyendaihoc.csv')
else:
    df = pd.read_csv(url_file)

# =========================
# 2. TẠO BIẾN PHỤ CẦN DÙNG
# =========================
for i in [1, 2, 3]:
    tbm_col = f'TBM{i}'
    us_col = f'US_TBM{i}'
    xl_col = f'XL{i}'

    if tbm_col not in df.columns:
        df[tbm_col] = (
            df[f'T{i}'] * 2 + df[f'L{i}'] + df[f'H{i}'] + df[f'S{i}'] +
            df[f'V{i}'] * 2 + df[f'X{i}'] + df[f'D{i}'] + df[f'N{i}']
        ) / 10

    if us_col not in df.columns:
        df[us_col] = df[tbm_col] / 10 * 4

    if xl_col not in df.columns:
        bins_xl = [0, 5, 6.5, 8, 9, 10.01]
        labels_xl = ['Y', 'TB', 'K', 'G', 'XS']
        df[xl_col] = pd.cut(df[tbm_col], bins=bins_xl, labels=labels_xl, include_lowest=True, right=False)

# KQXT: tạo biến Đậu/Rớt để trình bày dữ liệu ở phần sau
if 'KQXT' not in df.columns:
    df['KQXT'] = np.where(df[['DH1', 'DH2', 'DH3']].sum(axis=1) >= 15, 'Đậu', 'Rớt')

# Dân tộc: dữ liệu DT bị trống thường hiểu là Kinh
if 'DT_NHOM' not in df.columns:
    df['DT_NHOM'] = np.where(df['DT'].isna(), 'Kinh', 'Khác')

# Phân lớp T1 theo đề phần 4
bins_t1 = [0, 5, 7, 8, np.inf]
labels_t1 = ['kém', 'trung bình', 'khá', 'giỏi']
df['phanlopt1'] = pd.cut(df['T1'], bins=bins_t1, labels=labels_t1, right=False)

# =========================
# 3. HÀM IN RA TERMINAL
# =========================
def title(text):
    print('\n' + '=' * 100)
    print(text)
    print('=' * 100)


def sub_title(text):
    print('\n' + '-' * 100)
    print(text)
    print('-' * 100)


def print_table(name, data):
    print(f'\n{name}')
    print(data.to_string())


def q1(x):
    return x.quantile(0.25)


def q2(x):
    return x.quantile(0.50)


def q3(x):
    return x.quantile(0.75)


agg_funcs = ['count', 'sum', 'mean', 'median', 'min', 'max', 'std', q1, q2, q3]

# =========================
# 4. IN KẾT QUẢ RA TERMINAL
# =========================
title('LAB02 - KẾT QUẢ IN RA TERMINAL')
print('Dữ liệu có số dòng, số cột:', df.shape)
print('\n5 dòng đầu của dữ liệu:')
print(df.head().to_string())

# PHẦN 1
sub_title('PHẦN 1 - THỐNG KÊ DỮ LIỆU')

print('\nCâu 1. Sắp xếp dữ liệu điểm DH1 theo thứ tự tăng dần')
print(df.sort_values(by='DH1', ascending=True).head(20).to_string())

print('\nCâu 2. Sắp xếp dữ liệu điểm DH2 tăng dần theo nhóm giới tính')
print(df.sort_values(by=['GT', 'DH2'], ascending=[True, True]).head(30).to_string())

p1_3 = pd.pivot_table(df, values='DH1', index='KT', aggfunc=agg_funcs)
print_table('Câu 3. Pivot-table DH1 theo KT', p1_3)

p1_4 = pd.pivot_table(df, values='DH1', index=['KT', 'KV'], aggfunc=agg_funcs)
print_table('Câu 4. Pivot-table DH1 theo KT và KV', p1_4)

p1_5 = pd.pivot_table(df, values='DH1', index=['KT', 'KV', 'DT'], aggfunc=agg_funcs)
print_table('Câu 5. Pivot-table DH1 theo KT, KV và DT', p1_5)

# PHẦN 2
sub_title('PHẦN 2 - TRÌNH BÀY DỮ LIỆU')

GT_count = df['GT'].value_counts().sort_index()
GT_percent = df['GT'].value_counts(normalize=True).sort_index() * 100
GT_table = pd.DataFrame({'Tần số': GT_count, 'Tần suất (%)': GT_percent.round(2)})
print_table('Câu 1. Bảng tần số và tần suất biến GT', GT_table)

for col in ['US_TBM1', 'US_TBM2', 'US_TBM3']:
    print_table(f'Câu 2. Thống kê biến {col}', df[col].describe().to_frame())

p2_3 = df[df['GT'] == 'M']['DT'].value_counts(dropna=False)
print_table('Câu 3. Trình bày biến DT với học sinh nam', p2_3.to_frame('Số lượng'))

cond_p2_4 = (
    (df['GT'] == 'M') &
    (df['DT_NHOM'] == 'Kinh') &
    (df['DH1'] >= 5.0) &
    (df['DH2'] >= 4.0) &
    (df['DH3'] >= 4.0)
)
p2_4 = df.loc[cond_p2_4, ['STT', 'GT', 'DT', 'DT_NHOM', 'DH1', 'DH2', 'DH3', 'KT', 'KV']]
print_table('Câu 4. Học sinh nam, dân tộc Kinh, DH1 >= 5, DH2 >= 4, DH3 >= 4', p2_4)

cond_p2_5 = (
    (df['DH1'] > 5.0) &
    (df['DH2'] > 5.0) &
    (df['DH3'] > 5.0) &
    (df['KV'] == '2NT')
)
p2_5 = df.loc[cond_p2_5, ['STT', 'GT', 'KV', 'DH1', 'DH2', 'DH3', 'KT']]
print_table('Câu 5. DH1, DH2, DH3 > 5 và thuộc khu vực 2NT', p2_5)

# PHẦN 3
sub_title('PHẦN 3 - BẢNG TẦN SỐ THEO NHÓM')

df_nu = df[df['GT'] == 'F']
order_xl = ['Y', 'TB', 'K', 'G', 'XS']
xl_table = pd.DataFrame({
    'XL1': df_nu['XL1'].value_counts(),
    'XL2': df_nu['XL2'].value_counts(),
    'XL3': df_nu['XL3'].value_counts()
}).fillna(0).astype(int).reindex(order_xl).fillna(0).astype(int)
print_table('Câu 1. Nữ theo XL1, XL2, XL3', xl_table)

loc_p3_2 = df['KT'].isin(['A', 'A1', 'B']) & df['KV'].isin(['1', '2'])
p3_2 = pd.crosstab(df.loc[loc_p3_2, 'KT'], df.loc[loc_p3_2, 'KQXT'])
p3_3 = pd.crosstab(df['KT'], df['KV'])
p3_4 = pd.crosstab(df['KT'], df['KQXT'])
p3_5 = pd.crosstab(df['KV'], df['KQXT'])
p3_6 = pd.crosstab(df['DT_NHOM'], df['KQXT'])
p3_7 = pd.crosstab(df['GT'], df['KQXT'])

print_table('Câu 2. KQXT theo KT A/A1/B và KV 1/2', p3_2)
print_table('Câu 3. Số lượng khu vực theo từng khối thi', p3_3)
print_table('Câu 4. Số lượng Đậu/Rớt theo khối thi', p3_4)
print_table('Câu 5. Số lượng Đậu/Rớt theo khu vực', p3_5)
print_table('Câu 6. Số lượng Đậu/Rớt theo dân tộc', p3_6)
print_table('Câu 7. Số lượng Đậu/Rớt theo giới tính', p3_7)

# PHẦN 4
sub_title('PHẦN 4 - DỮ LIỆU PHÂN LỚP T1')
print_table('Bảng tần số phanlopt1', df['phanlopt1'].value_counts().reindex(labels_t1).to_frame('Số lượng'))
print('\nĐã vẽ các biểu đồ P4: Simple Line, Multiple Line, Drop-line ở file ảnh bên dưới.')

# PHẦN 5
sub_title('PHẦN 5 - KHẢO SÁT PHÂN PHỐI')
print_table('Câu 1. Thống kê mô tả biến T1', df['T1'].describe().to_frame())
print_table('Câu 2. Thống kê T1 theo phanlopt1', df.groupby('phanlopt1', observed=False)['T1'].describe())
print('\nCâu 3, 4, 5 đã vẽ Scatter Plot ở file ảnh bên dưới.')

# =========================
# 5. HÀM VẼ BIỂU ĐỒ
# =========================
def tan_so(series):
    return series.value_counts(dropna=False).sort_index()


def bar_chart(ax, data, chart_title, xlabel='', ylabel='Tần số', rotation=0):
    ax.bar(data.index.astype(str), data.values)
    ax.set_title(chart_title, fontsize=10, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', rotation=rotation)
    ax.grid(axis='y', alpha=0.25)


def pie_chart(ax, data, chart_title):
    ax.pie(data.values, labels=data.index.astype(str), autopct='%1.1f%%', startangle=90, textprops={'fontsize': 8})
    ax.set_title(chart_title, fontsize=10, fontweight='bold')


def cumulative_chart(ax, data, chart_title, xlabel=''):
    percent = data / data.sum() * 100
    cum_percent = percent.cumsum()
    ax.plot(cum_percent.index.astype(str), cum_percent.values, marker='o')
    ax.set_title(chart_title, fontsize=10, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Tần suất tích lũy (%)')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)


def grouped_bar(ax, table, chart_title, xlabel=''):
    table.plot(kind='bar', ax=ax)
    ax.set_title(chart_title, fontsize=10, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Số lượng')
    ax.tick_params(axis='x', rotation=0)
    ax.grid(axis='y', alpha=0.25)
    ax.legend(fontsize=7)


def qq_plot(ax, data, chart_title):
    stats.probplot(data.dropna(), dist='norm', plot=ax)
    ax.set_title(chart_title, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.25)

# =========================
# 6. VẼ VÀ LƯU TẤT CẢ BIỂU ĐỒ ĐÚNG THEO ĐỀ TRONG 1 ẢNH
# =========================
fig, axes = plt.subplots(8, 3, figsize=(24, 38))
axes = axes.flatten()
fig.suptitle('LAB02 - CÁC BIỂU ĐỒ ĐÚNG THEO ĐỀ', fontsize=22, fontweight='bold')
idx = 0

# PHẦN 2 - Câu 1: GT có yêu cầu rõ 3 biểu đồ
bar_chart(axes[idx], GT_count, 'P2.1 - Biểu đồ tần số GT', 'GT'); idx += 1
pie_chart(axes[idx], GT_count, 'P2.1 - Biểu đồ tần suất GT'); idx += 1
cumulative_chart(axes[idx], GT_count, 'P2.1 - Đa giác tích lũy GT', 'GT'); idx += 1

# PHẦN 3
xl_table.T.plot(kind='bar', ax=axes[idx])
axes[idx].set_title('P3.1 - Nữ theo XL1, XL2, XL3', fontsize=10, fontweight='bold')
axes[idx].set_xlabel('Nhóm xếp loại')
axes[idx].set_ylabel('Số lượng')
axes[idx].tick_params(axis='x', rotation=0)
axes[idx].grid(axis='y', alpha=0.25)
axes[idx].legend(title='Xếp loại', fontsize=7)
idx += 1

grouped_bar(axes[idx], p3_2, 'P3.2 - KQXT theo KT A/A1/B, KV 1/2', 'Khối thi'); idx += 1
grouped_bar(axes[idx], p3_3, 'P3.3 - Số lượng KV theo khối thi', 'Khối thi'); idx += 1
grouped_bar(axes[idx], p3_4, 'P3.4 - Đậu/Rớt theo khối thi', 'Khối thi'); idx += 1
grouped_bar(axes[idx], p3_5, 'P3.5 - Đậu/Rớt theo khu vực', 'Khu vực'); idx += 1
grouped_bar(axes[idx], p3_6, 'P3.6 - Đậu/Rớt theo dân tộc', 'Dân tộc'); idx += 1
grouped_bar(axes[idx], p3_7, 'P3.7 - Đậu/Rớt theo giới tính', 'Giới tính'); idx += 1

# PHẦN 4
axes[idx].plot(df.index, df['T1'], marker='o', markersize=3, linewidth=1)
axes[idx].set_title('P4.1 - Simple Line biến T1', fontsize=10, fontweight='bold')
axes[idx].set_xlabel('Chỉ số dòng')
axes[idx].set_ylabel('T1')
axes[idx].grid(True, alpha=0.3)
idx += 1

for nhom in labels_t1:
    tmp = df[df['phanlopt1'] == nhom]
    axes[idx].plot(tmp.index, tmp['T1'], marker='o', markersize=3, linewidth=1, label=nhom)
axes[idx].set_title('P4.4 - Multiple Line T1 theo phanlopt1', fontsize=10, fontweight='bold')
axes[idx].set_xlabel('Chỉ số dòng')
axes[idx].set_ylabel('T1')
axes[idx].grid(True, alpha=0.3)
axes[idx].legend(fontsize=7)
idx += 1

for nhom in labels_t1:
    tmp = df[df['phanlopt1'] == nhom]
    axes[idx].stem(tmp.index, tmp['T1'], label=nhom, basefmt=' ')
axes[idx].set_title('P4.5 - Drop-line T1 theo phanlopt1', fontsize=10, fontweight='bold')
axes[idx].set_xlabel('Chỉ số dòng')
axes[idx].set_ylabel('T1')
axes[idx].grid(True, alpha=0.3)
axes[idx].legend(fontsize=7)
idx += 1

# PHẦN 5
axes[idx].boxplot(df['T1'].dropna(), vert=False)
axes[idx].set_title('P5.1 - Boxplot T1', fontsize=10, fontweight='bold')
axes[idx].set_xlabel('T1')
axes[idx].set_yticks([])
axes[idx].grid(axis='x', alpha=0.25)
idx += 1

axes[idx].hist(df['T1'].dropna(), bins=10)
axes[idx].axvline(df['T1'].mean(), linestyle='--', label='Mean')
axes[idx].axvline(df['T1'].median(), linestyle='-', label='Median')
axes[idx].set_title('P5.1 - Histogram T1', fontsize=10, fontweight='bold')
axes[idx].set_xlabel('T1')
axes[idx].set_ylabel('Tần số')
axes[idx].grid(axis='y', alpha=0.25)
axes[idx].legend(fontsize=7)
idx += 1

qq_plot(axes[idx], df['T1'], 'P5.1 - QQ-Plot T1')
idx += 1

box_groups = [df.loc[df['phanlopt1'] == nhom, 'T1'].dropna() for nhom in labels_t1]
axes[idx].boxplot(box_groups, labels=labels_t1)
axes[idx].set_title('P5.2 - Boxplot T1 theo phanlopt1', fontsize=10, fontweight='bold')
axes[idx].set_xlabel('phanlopt1')
axes[idx].set_ylabel('T1')
axes[idx].grid(axis='y', alpha=0.25)
idx += 1

for nhom in labels_t1:
    axes[idx].hist(df.loc[df['phanlopt1'] == nhom, 'T1'].dropna(), bins=6, alpha=0.55, label=nhom)
axes[idx].set_title('P5.2 - Histogram T1 theo phanlopt1', fontsize=10, fontweight='bold')
axes[idx].set_xlabel('T1')
axes[idx].set_ylabel('Tần số')
axes[idx].grid(axis='y', alpha=0.25)
axes[idx].legend(fontsize=7)
idx += 1

for nhom in labels_t1:
    data = df.loc[df['phanlopt1'] == nhom, 'T1'].dropna()
    if len(data) >= 3:
        osm, osr = stats.probplot(data, dist='norm')[0]
        axes[idx].scatter(osm, osr, s=12, label=nhom)
axes[idx].set_title('P5.2 - QQ-Plot T1 theo phanlopt1', fontsize=10, fontweight='bold')
axes[idx].set_xlabel('Theoretical quantiles')
axes[idx].set_ylabel('Ordered values')
axes[idx].grid(True, alpha=0.25)
axes[idx].legend(fontsize=7)
idx += 1

axes[idx].scatter(df['T1'], df['DH1'])
z = np.polyfit(df['T1'].dropna(), df['DH1'].dropna(), 1)
p = np.poly1d(z)
x_sorted = np.sort(df['T1'].dropna())
axes[idx].plot(x_sorted, p(x_sorted))
axes[idx].set_title('P5.3 - Scatter DH1 theo T1', fontsize=10, fontweight='bold')
axes[idx].set_xlabel('T1')
axes[idx].set_ylabel('DH1')
axes[idx].grid(True, alpha=0.25)
idx += 1

for kv in sorted(df['KV'].dropna().unique()):
    tmp = df[df['KV'] == kv]
    axes[idx].scatter(tmp['T1'], tmp['DH1'], label=str(kv))
axes[idx].set_title('P5.4 - Scatter DH1 theo T1 từng KV', fontsize=10, fontweight='bold')
axes[idx].set_xlabel('T1')
axes[idx].set_ylabel('DH1')
axes[idx].grid(True, alpha=0.25)
axes[idx].legend(title='KV', fontsize=7)
idx += 1

pairs = [('DH1', 'DH2'), ('DH1', 'DH3'), ('DH2', 'DH3')]
for x, y in pairs:
    axes[idx].scatter(df[x], df[y])
    axes[idx].set_title(f'P5.5 - Scatter {y} theo {x}', fontsize=10, fontweight='bold')
    axes[idx].set_xlabel(x)
    axes[idx].set_ylabel(y)
    axes[idx].grid(True, alpha=0.25)
    idx += 1

for j in range(idx, len(axes)):
    axes[j].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.97])
output_file = 'bieu_do_lab02_dung_de.png'
plt.savefig(output_file, dpi=200, bbox_inches='tight')

print('\n' + '=' * 100)
print('ĐÃ XUẤT ẢNH BIỂU ĐỒ')
print('=' * 100)
print(f'Tên file ảnh: {output_file}')
print(f'Số biểu đồ đã vẽ: {idx}')
print('Bạn mở file ảnh này trong thư mục đang chạy code để xem tất cả biểu đồ trong 1 khung.')

plt.show()
