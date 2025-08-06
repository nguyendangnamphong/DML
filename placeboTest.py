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
data = pd.read_csv(file_path)

# Chuẩn bị dữ liệu cho placebo test
placebo_treatment = 'Aging_Time_h'  # Biến điều trị giả
outcome = 'Electrical_Conductivity_IACS'  # Biến kết quả

# Lọc các cột số và loại bỏ Alloy_Name, Alloy_Type, biến điều trị giả
numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
confounders = [col for col in numeric_columns if col not in [placebo_treatment, outcome, 'Alloy_Name', 'Alloy_Type']]

# Chuẩn bị dữ liệu
X = data[confounders].values
T = data[placebo_treatment].values
Y = data[outcome].values

# Chia dữ liệu
X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(X, T, Y, test_size=0.2, random_state=42)

# Khởi tạo mô hình
rf_model_Y = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model_T = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

dml = DML(
    model_y=rf_model_Y,
    model_t=rf_model_T,
    model_final=StatsModelsLinearRegression(fit_intercept=False),
    discrete_treatment=False,
    cv=5,
    random_state=42
)

# Huấn luyện và dự đoán
dml.fit(Y_train, T_train, X=X_train)
effect = dml.effect(X_test)
print(f"Hiệu ứng nhân quả trung bình của {placebo_treatment} lên {outcome}:", dml.const_marginal_effect(X_test).mean())
try:
    conf_int = dml.const_marginal_effect_interval(X_test, alpha=0.05)
    print(f"Khoảng tin cậy 95% cho {placebo_treatment}:", conf_int)
except Exception as e:
    print(f"Không thể tính khoảng tin cậy: {e}")

# Vẽ biểu đồ
plt.figure(figsize=(10, 6))
plt.plot(effect, label=f'Hiệu ứng nhân quả của {placebo_treatment}')
plt.axhline(0, color='gray', linestyle='--')
plt.title(f'Hiệu ứng nhân quả của {placebo_treatment} lên {outcome}')
plt.xlabel('Mẫu kiểm tra')
plt.ylabel('Hiệu ứng (%IACS)')
plt.legend()
plt.show()