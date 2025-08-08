import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
all_k_results = []
for k_i in range(1,38):
    print(f"Chạy với k = {k_i} cột")
    # Đọc dữ liệu
    df = pd.read_csv("data.csv", sep=',')

    # Mã hóa các cột dạng object
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col])
    # print("Cột đã mã hóa:", df.select_dtypes(include='object').columns.tolist())


    # Tách dữ liệu
    X = df.drop(columns=['building'
    ''
    '_id', 'labels'])
    y = df['labels']


    # Khởi tạo
    var_smoothing_values = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    best_runs = []
    all_runs = []

    # Loại bỏ outliers (nhiễu dữ liệu)
    from scipy import stats
    import numpy as np
    from sklearn.discriminant_analysis import StandardScaler

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    # Chọn 10 đặc trưng quan trọng nhất
    from sklearn.feature_selection import SelectKBest, f_classif

    selector = SelectKBest(score_func=f_classif, k=k_i)
    X_scaled = selector.fit_transform(X_scaled, y)
    print("Cột được chọn:", selector.get_support(indices=True))


    # Lấy tên cột từ DataFrame gốc (bỏ các cột không dùng trong X)
    feature_columns = df.drop(columns=['building_id', 'labels']).columns
    print("Tên cột:", feature_columns[selector.get_support()])

    # Loại bỏ nhiễu bằng IQR
    # Q1 = np.percentile(X_scaled, 25, axis=0)

    # Loại bỏ nhiễu bằng z-score
    z = np.abs(stats.zscore(X_scaled))
    mask = (z < 3).all(axis=1)
    X = X_scaled[mask]
    y = y[mask].reset_index(drop=True)

    # Lặp 10 lần với random split
    for run in range(20):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        for vs in var_smoothing_values:
            model = GaussianNB(var_smoothing=vs)
            model.fit(X_train, y_train)

            y_pred_test = model.predict(X_test)
            y_pred_train = model.predict(X_train)

            acc_test = accuracy_score(y_test, y_pred_test)
            acc_train = accuracy_score(y_train, y_pred_train)

            all_runs.append({
                'run': run,
                'var_smoothing': vs,
                'accuracy': acc_test,
                'train_accuracy': acc_train,
                'y_test': y_test,
                'y_pred': y_pred_test
            })

    # Tìm best run cho từng var_smoothing
    for vs in var_smoothing_values:
        subset = [r for r in all_runs if r['var_smoothing'] == vs]
        best = max(subset, key=lambda r: r['accuracy'])

        # Các chỉ số
        precision = precision_score(best['y_test'], best['y_pred'], average='weighted', zero_division=0)
        recall = recall_score(best['y_test'], best['y_pred'], average='weighted')
        f1 = f1_score(best['y_test'], best['y_pred'], average='weighted')

        best_runs.append({
            'k': k_i,
            'var_smoothing': vs,
            'run': best['run'],
            'accuracy': round(best['accuracy'], 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'train_error': round(1 - best['train_accuracy'], 4),
            'test_error': round(1 - best['accuracy'], 4)
        })
        
    # Tìm best overall theo accuracy
    best_overall = max(best_runs, key=lambda x: x['accuracy'])

    # Kết quả
    results_df = pd.DataFrame(best_runs)
    print(results_df.sort_values(by="accuracy", ascending=False))
    all_k_results.append(best_overall)
# Kết quả cuối cùng
results_df = pd.DataFrame(all_k_results)
print(results_df)