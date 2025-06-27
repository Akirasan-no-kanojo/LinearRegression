import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Đọc dữ liệu
df = pd.read_csv('sales_data.csv')
X_raw = df['Advertising_Budget'].values
y = df['Sales'].values

# 2. Chuẩn hóa X để tránh tràn số
X_min = X_raw.min()
X_max = X_raw.max()
X = (X_raw - X_min) / (X_max - X_min)

# 3. Khởi tạo tham số
w = 0
b = 0
learning_rate = 0.01
epochs = 1000
n = len(X)

# 4. Huấn luyện bằng Gradient Descent
for epoch in range(epochs):
    y_pred = w * X + b
    error = y - y_pred

    dw = (-2/n) * np.dot(X, error) # Đạo hàm của w
    db = (-2/n) * np.sum(error)    #Đạo hàm của b

    w -= learning_rate * dw
    b -= learning_rate * db

    if epoch % 100 == 0:
        loss = np.mean(error**2)
        print(f"Epoch {epoch}: Loss = {loss:.4f}, w = {w:.4f}, b = {b:.4f}")

# 5. In phương trình hồi quy (theo dữ liệu đã scale)
print(f"\nPhương trình (theo dữ liệu chuẩn hóa): Sales = {w:.2f} * scaled_Budget + {b:.2f}")

# 6. Nhập dữ liệu từ người dùng
try:
    user_input = float(input("Nhập ngân sách quảng cáo (nghìn đô): "))
    user_input_scaled = (user_input - X_min) / (X_max - X_min)
    predicted_sales = w * user_input_scaled + b
    print(f"Dự đoán doanh số: {predicted_sales:.2f} (nghìn sản phẩm)")

    # 7. Vẽ đồ thị
    plt.figure(figsize=(8, 6))
    plt.scatter(X_raw, y, color='skyblue', label='Dữ liệu thực tế')
    plt.plot(X_raw, w * X + b, color='red', label='Hồi quy tuyến tính (GD)')

    # Vẽ điểm dự đoán
    plt.scatter(user_input, predicted_sales, color='green', s=100, label='Dự đoán của bạn', zorder=5)
    plt.text(user_input + 0.5, predicted_sales, f"({user_input:.1f}, {predicted_sales:.2f})", 
             fontsize=10, color='green')

    # Trang trí biểu đồ
    plt.xlabel("Ngân sách quảng cáo (nghìn $)")
    plt.ylabel("Doanh số bán hàng (nghìn sản phẩm)")
    plt.title("Mô hình dự đoán doanh số bán hàng")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

except:
    print("Giá trị nhập vào không hợp lệ!")
