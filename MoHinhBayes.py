import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Bước 1: Đọc dữ liệu
df = pd.read_csv('data.csv')

# Bước 2: Loại bỏ cột ID nếu có
if 'building_id' in df.columns:
    df = df.drop(columns=['building_id'])

# Bước 3: Xác định X và y
X = df.drop(columns=['labels'])
y = df['labels']

# Bước 4: One-hot encoding
X = pd.get_dummies(X)

# Bước 5: Tách tập train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Bước 6: Huấn luyện Naive Bayes + đo thời gian huấn luyện
start_train = time.time()
model = GaussianNB()
model.fit(X_train, y_train)
end_train = time.time()
training_time = end_train - start_train

# Bước 7: Dự đoán + đo thời gian test
start_test = time.time()
y_pred = model.predict(X_test)
end_test = time.time()
testing_time = end_test - start_test

# Bước 8: Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')  # hoặc 'macro' nếu bạn muốn tính đều các lớp
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

# Bước 9: In kết quả
print("Kết quả đánh giá mô hình Naive Bayes:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"Thời gian huấn luyện: {training_time:.4f} giây")
print(f"Thời gian test:       {testing_time:.4f} giây")

from sklearn.decomposition import PCA

# Giảm chiều dữ liệu để trực quan hóa (chỉ dùng cho trực quan, không dùng để huấn luyện)
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test)

# Trực quan hóa nhãn thực tế
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test_pca[:, 0], y=X_test_pca[:, 1], hue=y_test, palette='Set2', alpha=0.7, edgecolor='k')
plt.title("Phân phối dữ liệu tập test sau PCA - theo nhãn thực")
plt.xlabel("Thành phần chính 1")
plt.ylabel("Thành phần chính 2")
plt.legend(title='Label')
plt.grid(True)
plt.tight_layout()
plt.show()

# Trực quan hóa nhãn dự đoán
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test_pca[:, 0], y=X_test_pca[:, 1], hue=y_pred, palette='Set1', alpha=0.7, edgecolor='k')
plt.title("Phân phối dữ liệu tập test sau PCA - theo nhãn dự đoán")
plt.xlabel("Thành phần chính 1")
plt.ylabel("Thành phần chính 2")
plt.legend(title='Predicted')
plt.grid(True)
plt.tight_layout()
plt.show()
