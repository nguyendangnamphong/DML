import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from econml.dml import DML
from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Đường dẫn đến file dữ liệu
file_path = 'E:\\Khóa luận\\Data\\final_data.csv'

# Đọc dữ liệu từ file CSV
df = pd.read_csv(file_path)

# Xác định biến điều trị, biến kết quả, và biến nhiễu
treatment = 'Ni'  # Biến điều trị
outcome = 'Electrical_Conductivity_IACS'  # Biến kết quả
confounders = ['Cu', 'Al', 'Cr', 'Mg', 'Si']  # Biến nhiễu

# Kiểm tra xem các cột có tồn tại không
missing_columns = [col for col in confounders + [treatment, outcome] if col not in df.columns]
if missing_columns:
    print(f"Không tìm thấy các cột: {', '.join(missing_columns)}")
    exit()

# Chuẩn bị dữ liệu
X = df[confounders].values  # Ma trận biến nhiễu
T = df[treatment].values  # Biến điều trị
Y = df[outcome].values  # Biến kết quả

# Kiểm tra dữ liệu NaN
if np.any(np.isnan(X)) or np.any(np.isnan(T)) or np.any(np.isnan(Y)):
    print("Dữ liệu chứa giá trị NaN. Đang xử lý bằng cách điền giá trị trung bình...")
    X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
    T = np.where(np.isnan(T), np.nanmean(T), T)
    Y = np.where(np.isnan(Y), np.nanmean(Y), Y)

# Kiểm tra dữ liệu không phải số
try:
    X = X.astype(float)
    T = T.astype(float)
    Y = Y.astype(float)
except ValueError as e:
    print(f"Lỗi: Dữ liệu chứa giá trị không phải số: {e}")
    exit()

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
    X, T, Y, test_size=0.2, random_state=42
)

# Khởi tạo Random Forest làm nuisance model
rf_model_Y = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)  # Dự đoán Y
rf_model_T = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)  # Dự đoán T

# Khởi tạo mô hình DML
dml = DML(
    model_y=rf_model_Y,          # Nuisance model cho Y
    model_t=rf_model_T,          # Nuisance model cho T
    model_final=StatsModelsLinearRegression(fit_intercept=False),  # Mô hình cuối hỗ trợ sai số chuẩn
    discrete_treatment=False,    # Ni là biến liên tục
    cv=5,                        # Số fold cho cross-fitting
    random_state=42
)

# Huấn luyện mô hình DML
dml.fit(Y_train, T_train, X=X_train)

# Ước lượng hiệu ứng nhân quả
effect = dml.effect(X_test)

# In kết quả
print("Hiệu ứng nhân quả trung bình của Ni lên độ dẫn điện:", dml.const_marginal_effect(X_test).mean())
try:
    conf_int = dml.const_marginal_effect_interval(X_test, alpha=0.05)
    print("Khoảng tin cậy 95%:", conf_int)
except Exception as e:
    print(f"Không thể tính khoảng tin cậy: {e}")

# Vẽ biểu đồ hiệu ứng nhân quả
plt.figure(figsize=(10, 6))
plt.plot(effect, label='Hiệu ứng nhân quả của Ni')
plt.axhline(0, color='gray', linestyle='--')
plt.title('Hiệu ứng nhân quả của Ni lên độ dẫn điện')
plt.xlabel('Mẫu kiểm tra')
plt.ylabel('Hiệu ứng (%IACS)')
plt.legend()
plt.savefig('E:/Khóa luận/Data/causal_effect_plot.png')  # Lưu biểu đồ tại đường dẫn mới
plt.show()