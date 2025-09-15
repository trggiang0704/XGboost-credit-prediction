# 💳 Credit Default Prediction với Flask & XGBoost
## 📌 Giới thiệu

Đây là một ứng dụng web dự đoán khả năng khách hàng có nguy cơ vỡ nợ (default) hay không dựa trên mô hình XGBoost.
Người dùng nhập vào các thông tin tài chính và cá nhân (ví dụ: giới tính, tuổi, nghề nghiệp, học vấn, tình trạng hôn nhân, thu nhập, số thẻ tín dụng, số lần trả chậm, dư nợ, v.v...) → hệ thống sẽ đưa ra dự đoán Default (vỡ nợ) hoặc Non-default (không vỡ nợ) ngay lập tức.

## 🛠️ Công nghệ và công cụ sử dụng
| Thành phần        | Công nghệ sử dụng         |
|-------------------|---------------------------|
| Ngôn ngữ lập trình | Python 3                 |
| Web framework     | Flask                     |
| Machine Learning  | XGBoost Classifier        |
| Tối ưu tham số    | Optuna                    |
| Xử lý dữ liệu     | pandas, scikit-learn      |
| Lưu mô hình       | joblib                    |
| Trực quan hóa     | matplotlib, seaborn       |
| Frontend          | HTML, CSS, bootstrap 5    |

## 🧠 Logic & hoạt động
Tiền xử lý dữ liệu

Làm sạch dữ liệu khách hàng.

Chuẩn hóa, mã hóa nhãn (Label Encoding / One-Hot Encoding).

Tách dữ liệu thành tập huấn luyện và kiểm thử.

Huấn luyện mô hình

Sử dụng XGBoost Classifier để dự đoán khả năng vỡ nợ.

Dùng Optuna để tối ưu siêu tham số (hyperparameters).

Lưu mô hình đã huấn luyện bằng joblib.

Triển khai Flask app

Giao diện web có form nhập thông tin khách hàng.

Khi submit → Flask backend load mô hình đã huấn luyện và dự đoán.

Kết quả hiển thị:

❌ Default (Khách hàng có nguy cơ vỡ nợ)

✅ Non-default (Khách hàng an toàn)

## 🎨 Giao diện

🔹 Form nhập dữ liệu khách hàng:
(Ảnh minh họa có thể chụp màn hình app của bạn và chèn vào đây)

🔹 Kết quả dự đoán:
(Ảnh kết quả dự đoán hiển thị “Default / Non-default”)

## 📂 Cấu trúc dự án
```bash
credit_prediction/
│── train.py           # Huấn luyện mô hình XGBoost
│── app.py             # Flask app để chạy web dự đoán
│── data.csv           # Dữ liệu khách hàng (input huấn luyện)
│── model.pkl          # File mô hình đã huấn luyện
│── requirements.txt   # Các thư viện cần cài đặt
```

## 🚀 Khởi chạy ứng dụng
```bash
1. Tạo môi trường ảo
python -m venv .venv

2. Kích hoạt môi trường ảo
Windows:
.venv\Scripts\activate
macOS/Linux:
source .venv/bin/activate

3. Cài đặt thư viện
pip install -r requirements.txt

4. Huấn luyện mô hình
python train.py

5. Chạy ứng dụng Flask
python app.py

6. Mở trình duyệt
http://127.0.0.1:5000
```
