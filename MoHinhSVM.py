import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Đọc dữ liệu
df = pd.read_csv("data.csv", delimiter=",")

# Kiểm tra nhanh
print("Kích thước dữ liệu:", df.shape)
print(df.head())

# Bỏ cột không cần thiết
if 'building_id' in df.columns:
    df = df.drop(columns=['building_id'])

# Xử lý missing (nếu có)
df = df.dropna()

# Kiểm tra nhãn
if 'labels' not in df.columns:
    raise ValueError("Không tìm thấy cột 'damage_grade'.")

# Phân chia X, y
X = df.drop(columns=['labels'])
y = df['labels']

# One-hot encoding cho biến phân loại
X = pd.get_dummies(X)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Huấn luyện mô hình SVM
svm_model = SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced')
svm_model.fit(X_train, y_train)

# Dự đoán
y_pred = svm_model.predict(X_test)

# Đánh giá
print("Báo cáo phân loại:\n")
print(classification_report(y_test, y_pred))

# Vẽ ma trận nhầm lẫn
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
plt.title("Confusion Matrix - SVM")
plt.show()


# Bước 5: Lưu mô hình (nếu cần)
import joblib
joblib.dump(svm_model, 'svm_model.pkl')
# Bước 6: Xuất dữ liệu đã chuẩn hóa ra file mới
df.to_csv('train_cleaned_for_svm.csv', index=False)
print("Dữ liệu đã được chuẩn hóa và lưu vào 'train_cleaned_for_svm.csv'")
# Bước 7: Giảm chiều dữ liệu để trực quan hóa (nếu cần)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test)
# Vẽ biểu đồ phân tán
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test_pca[:, 0], y=X_test_pca[:, 1], hue=y_test, palette='Set2', alpha=0.7, edgecolor='k')
plt.title("Phân phối dữ liệu tập test sau PCA - theo nhãn thực")
plt.xlabel("Thành phần chính 1")
plt.ylabel("Thành phần chính 2")
plt.legend(title='Nhãn')
plt.grid(True)
plt.tight_layout()
plt.show()
# Bước 7: Giảm chiều dữ liệu để trực quan hóa (nếu cần)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test)
# Vẽ biểu đồ phân tán
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test_pca[:, 0], y=X_test_pca[:, 1], hue=y_pred, palette='Set2', alpha=0.7, edgecolor='k')
plt.title("Phân phối dữ liệu tập test sau PCA - theo nhãn dự đoán")
plt.xlabel("Thành phần chính 1")
plt.ylabel("Thành phần chính 2")
plt.legend(title='Nhãn Dự đoán')
plt.grid(True)
plt.tight_layout()
plt.show()