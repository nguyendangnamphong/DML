import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from econml.dml import DML
from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Đường dẫn đến file dữ liệu
file_path = 'D:/Khóa luận/data/filtered_data.csv'

# Đọc dữ liệu
try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Không tìm thấy file tại {file_path}. Vui lòng kiểm tra đường dẫn.")
    exit()

# In các cột trong dữ liệu để kiểm tra
print("Các cột trong dữ liệu:", data.columns.tolist())

# Xác định biến điều trị (T), biến kết quả (Y), và biến nhiễu (X)
treatment = 'Ni'  # Biến điều trị
outcome = 'Electrical_Conductivity_IACS'  # Biến kết quả

# Kiểm tra xem các cột cần thiết có tồn tại không
if treatment not in data.columns or outcome not in data.columns:
    print(f"Không tìm thấy cột '{treatment}' hoặc '{outcome}' trong dữ liệu.")
    exit()

# Lọc các cột số và loại bỏ Alloy_Name, Alloy_Type
numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
confounders = [col for col in numeric_columns if col not in [treatment, outcome, 'Alloy_Name', 'Alloy_Type']]
print("Các cột số được sử dụng làm biến nhiễu:", confounders)

# Kiểm tra xem có cột số nào để làm biến nhiễu không
if not confounders:
    print("Không tìm thấy cột số nào để làm biến nhiễu. Vui lòng kiểm tra dữ liệu.")
    exit()

# Chuẩn bị dữ liệu
X = data[confounders].values  # Ma trận biến nhiễu
T = data[treatment].values  # Biến điều trị
Y = data[outcome].values  # Biến kết quả

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

# Kiểm tra kích thước dữ liệu và biến thiên
print("Số lượng mẫu:", len(X))
print("Giá trị duy nhất trong Ni:", len(np.unique(T)))
print("Thống kê Ni:", pd.Series(T).describe())
print("Thống kê Electrical_Conductivity_IACS:", pd.Series(Y).describe())
if len(X) < 50:
    print("Cảnh báo: Dữ liệu quá nhỏ (", len(X), "mẫu). DML cần dữ liệu lớn hơn để hoạt động tốt.")
if len(np.unique(T)) < 5:
    print("Cảnh báo: Biến điều trị 'Ni' có quá ít giá trị khác nhau (", len(np.unique(T)), "). Cần đủ biến thiên để ước lượng hiệu ứng nhân quả.")

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
    model_final=StatsModelsLinearRegression(fit_intercept=False),  # Mô hình cuối, không có intercept
    discrete_treatment=False,    # Ni là biến liên tục
    cv=5,                        # Số fold cho cross-fitting
    random_state=42
)

# Huấn luyện mô hình DML
try:
    dml.fit(Y_train, T_train, X=X_train)
except Exception as e:
    print(f"Lỗi khi huấn luyện mô hình DML: {e}")
    exit()

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
plt.show()