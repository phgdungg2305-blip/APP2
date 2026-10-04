# 📈 Hệ Thống Kiểm Định Chiến Lược Giao Dịch SMA + OBV & Phân Bổ Danh Mục MPT (HOSE)

Ứng dụng Web tương tác xây dựng trên nền tảng **Streamlit** phục vụ việc nghiên cứu, kiểm định định lượng (Backtesting), tối ưu hóa tham số chỉ báo kỹ thuật và so sánh hiệu quả phân bổ danh mục đầu tư giữa **Lý thuyết Danh mục Hiện đại (Modern Portfolio Theory - MPT / Markowitz)** và **Chiến lược Phân bổ Đều (Equal Weight - 1/N)** trên dữ liệu thực tế thị trường chứng khoán Việt Nam (HOSE).

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Mục Tiêu Dự Án & Phương Pháp Luận Nghiên Cứu

### 1. Tín hiệu Chỉ báo Kỹ thuật
* **SMA (Simple Moving Average):**
  - **Tín hiệu MUA (BUY = 1):** Đường SMA ngắn cắt lên trên đường SMA dài (*Golden Cross*).
  - **Tín hiệu BÁN (SELL = -1):** Đường SMA ngắn cắt xuống dưới đường SMA dài (*Death Cross*).
* **OBV (On-Balance Volume):**
  - Khối lượng cân bằng kết hợp với đường trung bình động của chính nó (OBV-MA).
  - **Tín hiệu MUA (BUY = 1):** OBV cắt lên trên đường OBV-MA (dòng tiền lớn gom hàng).
  - **Tín hiệu BÁN (SELL = -1):** OBV cắt xuống dưới đường OBV-MA (dòng tiền rút lui).
* **Cơ Chế Kết Hợp Tín Hiệu:**
  - **Chế độ AND:** Chỉ kích hoạt khi cả SMA và OBV cùng đồng thuận phát tín hiệu.
  - **Chế độ OR:** Kích hoạt khi có ít nhất một chỉ báo phát tín hiệu (tự động loại bỏ và đưa về trạng thái trung lập nếu hai chỉ báo mâu thuẫn).

### 2. Nguyên Tắc Backtesting Chặt Chẽ
* **Triệt tiêu Look-ahead Bias:** Vị thế giao dịch thực thi được **dịch chuyển 1 phiên** (`shift(1)`), đảm bảo tín hiệu chốt ngày $t$ chỉ được hành động ở ngày $t+1$.
* **Phân chia Thời gian Nghiêm ngặt (Walk-forward principle):**
  - **In-Sample (Tập Train: 2020-01-01 → 2021-12-31):** Dùng để huấn luyện, tối ưu hóa bộ tham số SMA/OBV bằng **Hyperopt (TPE Bayesian Optimization)** và ước lượng trọng số danh mục tối ưu theo MPT.
  - **Out-of-Sample (Tập Test: 2022-01-01 → 2022-12-31):** Giữ nguyên toàn bộ tham số và trọng số đã tìm được từ Train để kiểm định ngoài mẫu thực tế. Tuyệt đối không dùng Test để tối ưu.

### 3. Phân Bổ Danh Mục Đầu Tư
* **Equal Weight (1/N):** Mỗi cổ phiếu trong danh mục được phân bổ tỷ trọng bằng nhau $w_i = \frac{1}{N}$.
* **MPT (Markowitz):** Tối đa hóa Sharpe Ratio của danh mục trên tập Train với điều kiện biên Long-Only ($0 \le w_i \le 1$, $\sum w_i = 1$).

---

## 📂 Cấu Trúc Thư Mục Dự Án

```plaintext
├── app.py                      # Mã nguồn chính của ứng dụng Streamlit
├── requirements.txt            # Danh sách thư viện Python phụ thuộc
├── README.md                   # Tài liệu hướng dẫn sử dụng và triển khai
├── HOSE_2020_2023_in (1).csv   # File dữ liệu mẫu lịch sử giá HOSE 2020-2023
└── HOSE_SMA_OBV_EqualWeight_MPT (1).ipynb  # Jupyter Notebook nghiên cứu gốc
```

---

## 💻 Hướng Dẫn Cài Đặt & Chạy Trên Máy Cá Nhân (Local)

### Bước 1: Sao chép repository hoặc tải mã nguồn
```bash
git clone https://github.com/<tai-khoan-cua-ban>/<ten-repository>.git
cd <ten-repository>
```

### Bước 2: Tạo môi trường ảo Python (khuyến nghị)
```bash
# Trên Windows
python -m venv venv
.\venv\Scripts\activate

# Trên macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Bước 3: Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

### Bước 4: Khởi chạy ứng dụng Streamlit
```bash
streamlit run app.py
```
Trình duyệt web sẽ tự động mở tại địa chỉ: `http://localhost:8501`.

---

## 🚀 Hướng Dẫn Tải Lên GitHub & Triển Khai Miễn Phí Trên Streamlit Cloud

### Phần 1: Tải mã nguồn lên GitHub

1. Truy cập [GitHub](https://github.com) và đăng nhập vào tài khoản của bạn.
2. Tạo một Repository mới:
   - Nhấn vào dấu **`+`** ở góc trên cùng bên phải → chọn **New repository**.
   - Đặt tên repository (ví dụ: `hose-sma-obv-mpt-dashboard`).
   - Chọn chế độ **Public**.
   - Bỏ chọn phần *Add a README file* (vì chúng ta đã tạo sẵn file `README.md`).
   - Nhấn **Create repository**.
3. Đẩy mã nguồn từ máy tính lên GitHub bằng Git:
   ```bash
   git init
   git add .
   git commit -m "Khoi tao du an Streamlit kiem dinh SMA + OBV + MPT"
   git branch -M main
   git remote add origin https://github.com/<tai-khoan-cua-ban>/<ten-repository>.git
   git push -u origin main
   ```
   *(Hoặc bạn có thể dùng giao diện web của GitHub: chọn nút **"uploading an existing file"** rồi kéo thả các file `app.py`, `requirements.txt`, `README.md`, `HOSE_2020_2023_in (1).csv` lên và nhấn **Commit changes**).*

---

### Phần 2: Triển khai (Deploy) trên Streamlit Community Cloud

1. Truy cập vào [Streamlit Community Cloud](https://share.streamlit.io/) và đăng nhập bằng tài khoản GitHub của bạn.
2. Nhấn vào nút **"Create app"** (hoặc **"New app"**).
3. Điền thông tin cấu hình triển khai:
   - **Repository:** Chọn repository bạn vừa tạo ở Phần 1 (ví dụ: `<tai-khoan-cua-ban>/hose-sma-obv-mpt-dashboard`).
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** Bạn có thể tùy chỉnh tên miền con miễn phí theo ý muốn (ví dụ: `hose-trading-strategy.streamlit.app`).
4. Nhấn nút **"Deploy!"**.
5. Chờ hệ thống Streamlit tự động cài đặt các thư viện từ `requirements.txt` trong khoảng 1-2 phút. Sau khi hoàn tất, trang web của bạn sẽ hoạt động trực tuyến trên toàn cầu!

---

## 🌟 Các Tính Năng Nổi Bật Trên Giao Diện Web App

1. **Tự Động Nạp Dữ Liệu & Hỗ Trợ Tải File Tùy Chỉnh:**
   - Tự động nhận diện file CSV dữ liệu HOSE có sẵn trong thư mục.
   - Cho phép người dùng tải lên file CSV bất kỳ có định dạng chuẩn OHLCV.
2. **Tùy Biến Cổ Phiếu & Thời Gian Linh Hoạt:**
   - Lựa chọn danh sách cổ phiếu đa ngành (mặc định ACB, FPT, HPG hoặc tùy chọn bất kỳ).
   - Bộ chọn ngày trực quan cho hai giai đoạn In-Sample (Train) và Out-of-Sample (Test).
3. **Hai Chế Độ Tối Ưu Hóa Tham Số:**
   - **Hyperopt TPE:** Thuật toán tối ưu hóa Bayes tự động tìm ra bộ tham số có tỷ suất Sharpe cao nhất trên tập Train.
   - **Cấu hình thủ công (Manual):** Cho phép người dùng tự điều chỉnh chu kỳ SMA và OBV để kiểm chứng trực tiếp các giả thuyết đầu tư.
4. **Biểu Đồ Tương Tác Chuyên Sâu (Plotly):**
   - Đồ thị nến/đường kèm đường SMA và vị trí các điểm Buy/Sell rõ ràng.
   - Đồ thị dao động của chỉ báo OBV và đường trung bình động OBV-MA.
   - Đồ thị tăng trưởng tài sản (Portfolio Equity Curve) và mức sụt giảm tối đa (Drawdown Underwater Chart).
5. **So Sánh Toàn Diện Các Chỉ Số Hiệu Suất:**
   - Thống kê chi tiết: *Total Return, CAGR, Annual Volatility, Sharpe Ratio, Max Drawdown*.
   - Đối chiếu trực quan giữa In-Sample (Train) và Out-of-Sample (Test) nhằm phát hiện rủi ro Overfitting.
6. **Xuất Báo Cáo:**
   - Tải về kết quả kiểm định danh mục và bảng tham số tối ưu dưới định dạng file CSV chỉ với 1 click.

---

## 📜 Giấy Phép & Tác Quyền
Dự án được phân phối dưới giấy phép **MIT License**. Bạn được tự do sử dụng, chỉnh sửa và phát triển cho mục đích học tập và nghiên cứu.
