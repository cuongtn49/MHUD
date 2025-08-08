import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import ConfusionMatrixDisplay

# Đọc dữ liệu
df = pd.read_csv("data.csv", sep=',')

# Mã hóa các cột dạng object
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Tách dữ liệu
X = df.drop(columns=['building_id', 'labels'])
y = df['labels']

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Chọn các đặc trưng quan trọng nhất
selector = SelectKBest(score_func=f_classif, k=8)
X_scaled = selector.fit_transform(X_scaled, y)
print("Cột được chọn:", selector.get_support(indices=True))

# Lấy tên cột từ DataFrame gốc (bỏ các cột không dùng trong X)
feature_columns = df.drop(columns=['building_id', 'labels']).columns
# print("Tên cột:", feature_columns[selector.get_support()])

# Loại bỏ nhiễu bằng z-score
z = np.abs(stats.zscore(X_scaled))
mask = (z < 3).all(axis=1)
X = X_scaled[mask]
y = y[mask].reset_index(drop=True)

# Tách train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Khởi tạo và huấn luyện mô hình
model = GaussianNB(var_smoothing=0.1)

start_train = time.time()
model.fit(X_train, y_train)
end_train = time.time()

start_test = time.time()
y_pred = model.predict(X_test)
end_test = time.time()

# Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

train_error = 1 - accuracy_score(y_train, model.predict(X_train))
test_error = 1 - accuracy

train_time = end_train - start_train
test_time = end_test - start_test

# In kết quả
print(f"Accuracy     : {accuracy:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"F1-score     : {f1:.4f}")
print(f"Train Error  : {train_error:.4f}")
print(f"Test Error   : {test_error:.4f}")
print(f"Train Time   : {train_time:.4f} sec")
print(f"Test Time    : {test_time:.4f} sec")

# Vẽ biểu đồ
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
values = [accuracy, precision, recall, f1]
plt.figure(figsize=(8, 5))
plt.bar(metrics, values)
plt.ylim(0, 1)
plt.title('Evaluation Metrics for GaussianNB')
plt.ylabel('Score')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Ma trận nhầm lẫn
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()


# Tầm quan trọng của các đặc trưng
scores = selector.scores_[selector.get_support()]
selected_names = feature_columns[selector.get_support()]
plt.figure(figsize=(10, 5))
plt.barh(selected_names, scores)
plt.xlabel("F-score")
plt.title("Feature Importance (F-score)")
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()




