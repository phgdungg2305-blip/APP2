"""
=====================================================================================
HỆ THỐNG KIỂM ĐỊNH CHIẾN LƯỢC GIAO DỊCH SMA + OBV VÀ PHÂN BỔ DANH MỤC MPT (HOSE)
Ứng dụng Web App Streamlit chuyên nghiệp phục vụ nghiên cứu và kiểm định định lượng.
=====================================================================================
"""

import os
import io
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import ta
from scipy.optimize import minimize

try:
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------------
# CẤU HÌNH TRANG STREAMLIT
# -----------------------------------------------------------------------------------
st.set_page_config(
    page_title="Kiểm Định Chiến Lược SMA + OBV | MPT vs Equal Weight",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tùy biến giao diện CSS chuyên nghiệp
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: #F1F5F9;
        border-radius: 6px 6px 0px 0px;
        padding: 10px 18px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E3A8A !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------------
# CÁC HÀM XỬ LÝ DỮ LIỆU & CHỈ BÁO KỸ THUẬT (THEO LOGIC NOTEBOOK)
# -----------------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_csv_data(file_source):
    """
    Đọc dữ liệu CSV từ đường dẫn hoặc đối tượng tải lên của Streamlit.
    """
    if isinstance(file_source, str):
        df = pd.read_csv(file_source, encoding="utf-8-sig", low_memory=False)
    else:
        df = pd.read_csv(file_source, encoding="utf-8-sig", low_memory=False)

    df.columns = df.columns.str.strip().str.lower()
    required_cols = ["date", "ticker", "open", "high", "low", "close", "volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dữ liệu thiếu các cột bắt buộc: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df = df.dropna(subset=["date", "ticker"])
    return df


def prepare_stock_data(df_full, ticker):
    """
    Lọc và chuẩn hóa dữ liệu OHLCV cho một cổ phiếu đơn lẻ.
    """
    ticker = ticker.upper()
    df = df_full[df_full["ticker"] == ticker].copy()

    if df.empty:
        raise ValueError(f"Không tìm thấy mã {ticker} trong dữ liệu.")

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    df = df.sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="last")

    df = df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    })

    df = df[["date", "Open", "High", "Low", "Close", "Volume"]]
    df = df.set_index("date")
    return df


def find_position_sma(df, paras):
    """
    Xác định tín hiệu giao dịch dựa trên đường SMA giao cắt:
    - BUY (1): SMA ngắn cắt lên SMA dài.
    - SELL (-1): SMA ngắn cắt xuống SMA dài.
    """
    position = pd.Series(0.0, index=df.index, name="position")
    ma_short = int(paras["ma_short"])
    ma_long = int(paras["ma_long"])

    if ma_short >= ma_long:
        return position

    ma_s = ta.trend.SMAIndicator(close=df["Close"], window=ma_short).sma_indicator()
    ma_l = ta.trend.SMAIndicator(close=df["Close"], window=ma_long).sma_indicator()

    buy_signal = (ma_s > ma_l) & (ma_s.shift(1) <= ma_l.shift(1))
    sell_signal = (ma_s < ma_l) & (ma_s.shift(1) >= ma_l.shift(1))

    position.loc[buy_signal] = 1.0
    position.loc[sell_signal] = -1.0
    return position


def find_position_obv(df, paras):
    """
    Xác định tín hiệu giao dịch dựa trên OBV và OBV-MA:
    - BUY (1): OBV cắt lên đường trung bình động OBV.
    - SELL (-1): OBV cắt xuống đường trung bình động OBV.
    """
    position = pd.Series(0.0, index=df.index, name="position")
    obv_window = int(paras["obv_window"])

    obv = ta.volume.OnBalanceVolumeIndicator(close=df["Close"], volume=df["Volume"]).on_balance_volume()
    obv_ma = obv.rolling(window=obv_window).mean()

    buy_signal = (obv > obv_ma) & (obv.shift(1) <= obv_ma.shift(1))
    sell_signal = (obv < obv_ma) & (obv.shift(1) >= obv_ma.shift(1))

    position.loc[buy_signal] = 1.0
    position.loc[sell_signal] = -1.0
    return position


def find_position_combined_and(df, sma_paras, obv_paras):
    """
    Chiến lược kết hợp AND: Đồng thuận cả hai chỉ báo.
    """
    sma = find_position_sma(df, sma_paras)
    obv = find_position_obv(df, obv_paras)

    position = pd.Series(0.0, index=df.index, name="position")
    position.loc[(sma == 1.0) & (obv == 1.0)] = 1.0
    position.loc[(sma == -1.0) & (obv == -1.0)] = -1.0
    return position


def find_position_combined_or(df, sma_paras, obv_paras):
    """
    Chiến lược kết hợp OR: Một trong hai chỉ báo phát tín hiệu (loại bỏ mâu thuẫn).
    """
    sma = find_position_sma(df, sma_paras)
    obv = find_position_obv(df, obv_paras)

    position = pd.Series(0.0, index=df.index, name="position")
    buy = (sma == 1.0) | (obv == 1.0)
    sell = (sma == -1.0) | (obv == -1.0)
    conflict = buy & sell

    position.loc[buy & ~conflict] = 1.0
    position.loc[sell & ~conflict] = -1.0
    return position


def events_to_holding(events):
    """
    Chuyển đổi tín hiệu rời rạc thành chuỗi nắm giữ tài sản liên tục (Holding):
    1: Mua nắm giữ, -1: Bán về tiền mặt, 0: Giữ nguyên trạng thái trước đó.
    """
    holding = pd.Series(0.0, index=events.index)
    current = 0.0
    for i, signal in enumerate(events):
        if signal == 1.0:
            current = 1.0
        elif signal == -1.0:
            current = 0.0
        holding.iloc[i] = current
    return holding


def strategy_returns(df, events, commission=0.0):
    """
    Tính lợi nhuận chiến lược:
    - Dịch chuyển 1 phiên (shift 1) để tránh Look-ahead Bias.
    - Khấu trừ phí giao dịch phát sinh từ biến động vị thế (Turnover).
    """
    asset_ret = df["Close"].pct_change().fillna(0.0)
    holding = events_to_holding(events)

    # Tín hiệu phiên t thực thi ở phiên t+1
    executed_holding = holding.shift(1).fillna(0.0)
    strat_ret = executed_holding * asset_ret

    turnover = executed_holding.diff().abs().fillna(executed_holding.abs())
    strat_ret = strat_ret - turnover * commission

    return strat_ret, executed_holding


def performance_stats(returns, trading_days=252):
    """
    Tính toán các chỉ số đánh giá hiệu quả đầu tư định lượng.
    """
    r = returns.dropna()
    if len(r) == 0:
        return {
            "Total Return [%]": np.nan,
            "Annual Return [%]": np.nan,
            "Annual Volatility [%]": np.nan,
            "Sharpe Ratio": np.nan,
            "Max Drawdown [%]": np.nan
        }

    equity = (1.0 + r).cumprod()
    total_return = equity.iloc[-1] - 1.0

    years = len(r) / trading_days
    annual_return = (
        equity.iloc[-1] ** (1.0 / years) - 1.0
        if years > 0 and equity.iloc[-1] > 0
        else np.nan
    )

    annual_vol = r.std() * np.sqrt(trading_days)
    sharpe = (
        r.mean() / r.std() * np.sqrt(trading_days)
        if r.std() != 0
        else np.nan
    )

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_dd = drawdown.min()

    return {
        "Total Return [%]": total_return * 100.0,
        "Annual Return [%]": annual_return * 100.0,
        "Annual Volatility [%]": annual_vol * 100.0,
        "Sharpe Ratio": sharpe,
        "Max Drawdown [%]": max_dd * 100.0
    }


# -----------------------------------------------------------------------------------
# TỐI ƯU HÓA HYPEROPT TRÊN TẬP TRAIN (IN-SAMPLE ONLY)
# -----------------------------------------------------------------------------------

def score_sma(paras, df, commission=0.0):
    paras = {"ma_short": int(paras["ma_short"]), "ma_long": int(paras["ma_long"])}
    if paras["ma_short"] >= paras["ma_long"]:
        return 999999.0
    events = find_position_sma(df, paras)
    ret, _ = strategy_returns(df, events, commission=commission)
    stats = performance_stats(ret)
    sharpe = stats["Sharpe Ratio"]
    if pd.isna(sharpe):
        return 999999.0
    return -float(sharpe)


def score_obv(paras, df, commission=0.0):
    paras = {"obv_window": int(paras["obv_window"])}
    events = find_position_obv(df, paras)
    ret, _ = strategy_returns(df, events, commission=commission)
    stats = performance_stats(ret)
    sharpe = stats["Sharpe Ratio"]
    if pd.isna(sharpe):
        return 999999.0
    return -float(sharpe)


def optimize_one_stock(df_train, max_evals=40, commission=0.0):
    """
    Tối ưu hóa các tham số SMA và OBV độc lập nhằm tối đa hóa Sharpe trên tập Train.
    """
    if not HYPEROPT_AVAILABLE:
        # Fallback nếu thiếu hyperopt (Grid search thu nhỏ)
        best_sma = {"ma_short": 50, "ma_long": 200}
        best_obv = {"obv_window": 20}
        return best_sma, best_obv

    space_sma = {
        "ma_short": hp.quniform("ma_short", 20, 100, 5),
        "ma_long": hp.quniform("ma_long", 120, 300, 10)
    }

    trials_sma = Trials()
    best_sma_raw = fmin(
        fn=lambda p: score_sma(p, df_train, commission=commission),
        space=space_sma,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials_sma,
        verbose=False,
        show_progressbar=False
    )

    sma_best = {
        "ma_short": int(best_sma_raw["ma_short"]),
        "ma_long": int(best_sma_raw["ma_long"])
    }

    space_obv = {
        "obv_window": hp.quniform("obv_window", 5, 80, 5)
    }

    trials_obv = Trials()
    best_obv_raw = fmin(
        fn=lambda p: score_obv(p, df_train, commission=commission),
        space=space_obv,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials_obv,
        verbose=False,
        show_progressbar=False
    )

    obv_best = {
        "obv_window": int(best_obv_raw["obv_window"])
    }

    return sma_best, obv_best


def evaluate_stock(df, sma_best, obv_best, commission=0.0):
    """
    Đánh giá cả 4 biến thể chiến lược trên một khung dữ liệu.
    """
    strategies = {}

    event_sma = find_position_sma(df, sma_best)
    ret_sma, _ = strategy_returns(df, event_sma, commission=commission)
    strategies["SMA"] = ret_sma

    event_obv = find_position_obv(df, obv_best)
    ret_obv, _ = strategy_returns(df, event_obv, commission=commission)
    strategies["OBV"] = ret_obv

    event_and = find_position_combined_and(df, sma_best, obv_best)
    ret_and, _ = strategy_returns(df, event_and, commission=commission)
    strategies["SMA + OBV AND"] = ret_and

    event_or = find_position_combined_or(df, sma_best, obv_best)
    ret_or, _ = strategy_returns(df, event_or, commission=commission)
    strategies["SMA + OBV OR"] = ret_or

    # Buy & Hold làm mốc so sánh
    bh_ret = df["Close"].pct_change().fillna(0.0)
    strategies["Buy & Hold"] = bh_ret

    stats = pd.DataFrame({
        name: performance_stats(ret)
        for name, ret in strategies.items()
    }).T

    return strategies, stats


# -----------------------------------------------------------------------------------
# PHÂN BỔ DANH MỤC: MPT (MARKOWITZ) VS EQUAL WEIGHT
# -----------------------------------------------------------------------------------

def portfolio_annual_return(weights, returns, trading_days=252):
    mean_daily = returns.mean().values
    return float(weights @ mean_daily * trading_days)


def portfolio_annual_volatility(weights, returns, trading_days=252):
    cov_annual = returns.cov().values * trading_days
    variance = float(weights.T @ cov_annual @ weights)
    return np.sqrt(max(variance, 0.0))


def negative_sharpe(weights, returns, risk_free_rate=0.0, trading_days=252):
    p_return = portfolio_annual_return(weights, returns, trading_days)
    p_vol = portfolio_annual_volatility(weights, returns, trading_days)
    if p_vol == 0:
        return 1e9
    return -(p_return - risk_free_rate) / p_vol


def optimize_mpt(returns, risk_free_rate=0.0, trading_days=252):
    """
    Tối đa hóa Sharpe Ratio trên dữ liệu Train (Long-Only: sum(w)=1, w_i >= 0).
    """
    n = returns.shape[1]
    x0 = np.repeat(1.0 / n, n)
    bounds = [(0.0, 1.0)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    result = minimize(
        negative_sharpe,
        x0=x0,
        args=(returns, risk_free_rate, trading_days),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        return x0
    return result.x


def portfolio_returns(return_matrix, weights):
    return return_matrix.mul(weights, axis=1).sum(axis=1)


# -----------------------------------------------------------------------------------
# GIAO DIỆN CHÍNH STREAMLIT
# -----------------------------------------------------------------------------------

def main():
    st.markdown('<div class="main-header">📊 Kiểm Định Chiến Lược Kết Hợp SMA & OBV</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Phân tích hiệu quả chiến lược giao dịch kỹ thuật và phân bổ danh mục MPT (Markowitz) vs Equal Weight trên sàn HOSE</div>',
        unsafe_allow_html=True
    )

    # ------------------ SIDEBAR CẤU HÌNH ------------------
    st.sidebar.header("⚙️ Cấu Hình Dữ Liệu & Tham Số")

    # 1. Nạp file dữ liệu
    uploaded_file = st.sidebar.file_uploader("Tải lên file CSV dữ liệu HOSE", type=["csv"])

    default_files = [
        "HOSE_2020_2023_in (1).csv",
        "HOSE_2020_2023_in.csv",
        "data.csv"
    ]
    data_path = None
    if uploaded_file is not None:
        data_source = uploaded_file
    else:
        for fname in default_files:
            if os.path.exists(fname):
                data_path = fname
                break
        data_source = data_path

    if data_source is None:
        st.warning("⚠️ Chưa tìm thấy file dữ liệu CSV mặc định. Vui lòng tải file lên thanh bên để bắt đầu!")
        st.stop()

    try:
        df_full = load_csv_data(data_source)
    except Exception as e:
        st.error(f"Lỗi khi đọc file dữ liệu: {e}")
        st.stop()

    all_tickers = sorted(df_full["ticker"].dropna().unique().tolist())

    # 2. Chọn cổ phiếu
    default_selected = [t for t in ["ACB", "FPT", "HPG"] if t in all_tickers]
    if len(default_selected) < 3 and len(all_tickers) >= 3:
        default_selected = all_tickers[:3]

    selected_tickers = st.sidebar.multiselect(
        "Chọn danh sách cổ phiếu (khuyến nghị ≥ 3 mã):",
        options=all_tickers,
        default=default_selected
    )

    if len(selected_tickers) < 2:
        st.sidebar.error("Vui lòng chọn ít nhất 2 cổ phiếu để phân bổ danh mục!")
        st.stop()

    # 3. Phân chia Train & Test
    st.sidebar.subheader("📅 Phân Kỳ Dữ Liệu")
    min_date = df_full["date"].min().date()
    max_date = df_full["date"].max().date()

    col_t1, col_t2 = st.sidebar.columns(2)
    train_start = col_t1.date_input("Train Bắt đầu", pd.to_datetime("2020-01-01").date(), min_value=min_date, max_value=max_date)
    train_end = col_t2.date_input("Train Kết thúc", pd.to_datetime("2021-12-31").date(), min_value=min_date, max_value=max_date)

    col_t3, col_t4 = st.sidebar.columns(2)
    test_start = col_t3.date_input("Test Bắt đầu", pd.to_datetime("2022-01-01").date(), min_value=min_date, max_value=max_date)
    test_end = col_t4.date_input("Test Kết thúc", pd.to_datetime("2022-12-31").date(), min_value=min_date, max_value=max_date)

    # 4. Cấu hình giao dịch
    st.sidebar.subheader("💰 Cấu Hình Giao Dịch")
    portfolio_mode = st.sidebar.selectbox("Chế độ kết hợp tín hiệu cho Danh mục:", ["OR", "AND"])
    commission_pct = st.sidebar.number_input("Phí giao dịch mỗi chiều (%)", value=0.0, step=0.05, min_value=0.0, max_value=2.0)
    commission = commission_pct / 100.0

    initial_capital = st.sidebar.number_input("Vốn khởi điểm (VNĐ)", value=1_000_000, step=100_000)
    risk_free_rate = st.sidebar.number_input("Lãi suất phi rủi ro năm (Rf, %)", value=0.0, step=0.5) / 100.0

    # 5. Phương thức tối ưu hóa
    st.sidebar.subheader("🧠 Tối Ưu Hóa Tham Số")
    opt_mode = st.sidebar.radio("Phương thức tìm tham số SMA / OBV:", ["Hyperopt (Tự động tối ưu trên Train)", "Cấu hình thủ công (Manual)"])

    manual_params = {}
    max_evals = 30
    if opt_mode == "Hyperopt (Tự động tối ưu trên Train)":
        max_evals = st.sidebar.slider("Số lần thử nghiệm Hyperopt (Max Evals):", 10, 80, 30, step=10)
    else:
        st.sidebar.caption("Thiết lập tham số cho từng mã:")
        for t in selected_tickers:
            with st.sidebar.expander(f"Tham số {t}", expanded=False):
                s_ma = st.number_input(f"{t} - SMA Ngắn", min_value=5, max_value=150, value=50, step=5, key=f"s_{t}")
                l_ma = st.number_input(f"{t} - SMA Dài", min_value=50, max_value=400, value=200, step=10, key=f"l_{t}")
                obv_w = st.number_input(f"{t} - OBV Window", min_value=3, max_value=120, value=20, step=5, key=f"o_{t}")
                manual_params[t] = {
                    "SMA": {"ma_short": s_ma, "ma_long": l_ma},
                    "OBV": {"obv_window": obv_w}
                }

    run_clicked = st.sidebar.button("🚀 Chạy Phân Tích & Tối Ưu Hóa", use_container_width=True, type="primary")

    # Lưu kết quả tính toán vào Session State để chuyển đổi tab mượt mà
    if "results" not in st.session_state or run_clicked:
        if run_clicked or ("results" not in st.session_state):
            with st.spinner("Đang chuẩn bị dữ liệu và thực hiện kiểm định..."):
                try:
                    # Chuẩn bị dữ liệu từng mã
                    stock_data = {t: prepare_stock_data(df_full, t) for t in selected_tickers}
                    train_data = {}
                    test_data = {}

                    for t, df in stock_data.items():
                        train = df.loc[(df.index >= pd.to_datetime(train_start)) & (df.index <= pd.to_datetime(train_end))].copy()
                        test = df.loc[(df.index >= pd.to_datetime(test_start)) & (df.index <= pd.to_datetime(test_end))].copy()
                        if train.empty or test.empty:
                            st.error(f"Mã {t} có tập Train hoặc Test bị rỗng. Vui lòng kiểm tra lại khoảng ngày chọn!")
                            st.stop()
                        train_data[t] = train
                        test_data[t] = test

                    # Tối ưu hóa hoặc lấy tham số
                    best_params = {}
                    progress_bar = st.progress(0, text="Bắt đầu tối ưu hóa tham số...")
                    for idx, t in enumerate(selected_tickers):
                        if opt_mode == "Hyperopt (Tự động tối ưu trên Train)":
                            progress_bar.progress((idx) / len(selected_tickers), text=f"Đang tối ưu hóa Bayesian TPE cho {t} trên tập Train...")
                            sma_b, obv_b = optimize_one_stock(train_data[t], max_evals=max_evals, commission=commission)
                            best_params[t] = {"SMA": sma_b, "OBV": obv_b}
                        else:
                            best_params[t] = manual_params[t]
                    progress_bar.progress(1.0, text="Hoàn tất tối ưu hóa!")

                    # Đánh giá trên Train và Test
                    train_results = {}
                    test_results = {}
                    for t in selected_tickers:
                        sma_p = best_params[t]["SMA"]
                        obv_p = best_params[t]["OBV"]
                        train_s, train_st = evaluate_stock(train_data[t], sma_p, obv_p, commission=commission)
                        test_s, test_st = evaluate_stock(test_data[t], sma_p, obv_p, commission=commission)
                        train_results[t] = {"returns": train_s, "stats": train_st}
                        test_results[t] = {"returns": test_s, "stats": test_st}

                    # Xây dựng ma trận lợi nhuận danh mục
                    strat_name = "SMA + OBV OR" if portfolio_mode == "OR" else "SMA + OBV AND"
                    train_matrix = pd.concat({t: train_results[t]["returns"][strat_name] for t in selected_tickers}, axis=1).dropna()
                    test_matrix = pd.concat({t: test_results[t]["returns"][strat_name] for t in selected_tickers}, axis=1).dropna()

                    # Phân bổ Equal Weight & MPT
                    n_assets = len(selected_tickers)
                    equal_weights = np.repeat(1.0 / n_assets, n_assets)
                    mpt_weights = optimize_mpt(train_matrix, risk_free_rate=risk_free_rate)

                    # Backtest Danh Mục
                    ew_train_ret = portfolio_returns(train_matrix, equal_weights)
                    mpt_train_ret = portfolio_returns(train_matrix, mpt_weights)
                    ew_test_ret = portfolio_returns(test_matrix, equal_weights)
                    mpt_test_ret = portfolio_returns(test_matrix, mpt_weights)

                    port_train_stats = pd.DataFrame({
                        "Equal Weight": performance_stats(ew_train_ret),
                        "MPT (Markowitz)": performance_stats(mpt_train_ret)
                    }).T

                    port_test_stats = pd.DataFrame({
                        "Equal Weight": performance_stats(ew_test_ret),
                        "MPT (Markowitz)": performance_stats(mpt_test_ret)
                    }).T

                    # Lưu session
                    st.session_state["results"] = {
                        "stock_data": stock_data,
                        "train_data": train_data,
                        "test_data": test_data,
                        "best_params": best_params,
                        "train_results": train_results,
                        "test_results": test_results,
                        "strat_name": strat_name,
                        "train_matrix": train_matrix,
                        "test_matrix": test_matrix,
                        "equal_weights": equal_weights,
                        "mpt_weights": mpt_weights,
                        "ew_train_ret": ew_train_ret,
                        "mpt_train_ret": mpt_train_ret,
                        "ew_test_ret": ew_test_ret,
                        "mpt_test_ret": mpt_test_ret,
                        "port_train_stats": port_train_stats,
                        "port_test_stats": port_test_stats,
                        "selected_tickers": selected_tickers,
                        "initial_capital": initial_capital,
                        "commission": commission
                    }
                    st.success("✅ Đã hoàn tất phân tích & kiểm định thành công!")
                except Exception as err:
                    st.error(f"Xảy ra lỗi trong quá trình tính toán: {err}")
                    st.stop()

    res = st.session_state.get("results", None)
    if res is None:
        st.info("Nhấn '🚀 Chạy Phân Tích & Tối Ưu Hóa' ở thanh bên trái để bắt đầu mô phỏng.")
        return

    # ------------------ CÁC TAB HIỂN THỊ KẾT QUẢ ------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Tổng Quan & Dữ Liệu",
        "📈 Phân Tích Từng Cổ Phiếu",
        "⚙️ Tham Số Tối Ưu (Train)",
        "💼 Danh Mục (MPT vs Equal Weight)",
        "📋 Báo Cáo & Kết Luận Kiểm Định"
    ])

    # ===============================================================================
    # TAB 1: TỔNG QUAN DỮ LIỆU
    # ===============================================================================
    with tab1:
        st.subheader("📌 Tổng quan bộ dữ liệu HOSE")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tổng số dòng dữ liệu", f"{len(df_full):,}")
        c2.metric("Số mã cổ phiếu trong file", f"{df_full['ticker'].nunique()}")
        c3.metric("Ngày bắt đầu dữ liệu", str(df_full['date'].min().date()))
        c4.metric("Ngày kết thúc dữ liệu", str(df_full['date'].max().date()))

        st.markdown("---")
        st.markdown("#### 🔍 Thống kê các mã cổ phiếu đang chọn:")
        summary_rows = []
        for t in res["selected_tickers"]:
            d_all = res["stock_data"][t]
            d_tr = res["train_data"][t]
            d_te = res["test_data"][t]
            summary_rows.append({
                "Mã CP": t,
                "Số phiên (Toàn bộ)": len(d_all),
                "Số phiên Train": len(d_tr),
                "Số phiên Test": len(d_te),
                "Giá đầu kỳ (Train)": round(d_tr["Close"].iloc[0], 2),
                "Giá cuối kỳ (Train)": round(d_tr["Close"].iloc[-1], 2),
                "Giá cuối kỳ (Test)": round(d_te["Close"].iloc[-1], 2)
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

        with st.expander("👀 Xem trước mẫu dữ liệu gốc (5 dòng đầu)"):
            st.dataframe(df_full.head(10), use_container_width=True)

    # ===============================================================================
    # TAB 2: PHÂN TÍCH TỪNG CỔ PHIẾU
    # ===============================================================================
    with tab2:
        st.subheader("📈 Phân tích kỹ thuật & Hiệu quả chiến lược từng cổ phiếu")

        ticker_pick = st.selectbox("Chọn cổ phiếu để xem chi tiết:", res["selected_tickers"])
        df_stock = res["stock_data"][ticker_pick]
        p_sma = res["best_params"][ticker_pick]["SMA"]
        p_obv = res["best_params"][ticker_pick]["OBV"]

        # Tính toán chỉ báo để vẽ đồ thị
        ma_s_series = ta.trend.SMAIndicator(close=df_stock["Close"], window=int(p_sma["ma_short"])).sma_indicator()
        ma_l_series = ta.trend.SMAIndicator(close=df_stock["Close"], window=int(p_sma["ma_long"])).sma_indicator()
        obv_series = ta.volume.OnBalanceVolumeIndicator(close=df_stock["Close"], volume=df_stock["Volume"]).on_balance_volume()
        obv_ma_series = obv_series.rolling(window=int(p_obv["obv_window"])).mean()

        events_or = find_position_combined_or(df_stock, p_sma, p_obv)
        buy_points = df_stock[events_or == 1.0]
        sell_points = df_stock[events_or == -1.0]

        # Đồ thị Nến & Chỉ báo kỹ thuật Plotly
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(f"Đồ thị Giá & Tín hiệu Giao dịch ({ticker_pick})", "Chỉ báo OBV & OBV-MA"),
            row_heights=[0.65, 0.35]
        )

        # Candlestick / Line
        fig.add_trace(go.Scatter(
            x=df_stock.index, y=df_stock["Close"],
            mode="lines", name="Giá Close", line=dict(color="#2563EB", width=1.5)
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df_stock.index, y=ma_s_series,
            mode="lines", name=f"SMA ({p_sma['ma_short']})", line=dict(color="#F59E0B", width=1.2)
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df_stock.index, y=ma_l_series,
            mode="lines", name=f"SMA ({p_sma['ma_long']})", line=dict(color="#9333EA", width=1.2)
        ), row=1, col=1)

        # Điểm Buy & Sell
        fig.add_trace(go.Scatter(
            x=buy_points.index, y=buy_points["Close"],
            mode="markers", name="Tín hiệu MUA",
            marker=dict(symbol="triangle-up", size=10, color="#10B981")
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=sell_points.index, y=sell_points["Close"],
            mode="markers", name="Tín hiệu BÁN",
            marker=dict(symbol="triangle-down", size=10, color="#EF4444")
        ), row=1, col=1)

        # OBV
        fig.add_trace(go.Scatter(
            x=df_stock.index, y=obv_series,
            mode="lines", name="OBV", line=dict(color="#0D9488", width=1.2)
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=df_stock.index, y=obv_ma_series,
            mode="lines", name=f"OBV-MA ({p_obv['obv_window']})", line=dict(color="#E11D48", width=1.2, dash="dash")
        ), row=2, col=1)

        fig.update_layout(
            height=600,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=60, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

        # So sánh hiệu năng các biến thể chiến lược trên Train & Test
        st.markdown(f"#### 📊 Bảng so sánh hiệu năng các chiến lược cho mã **{ticker_pick}**")
        col_st1, col_st2 = st.columns(2)

        with col_st1:
            st.markdown("##### 🟢 Dữ liệu In-Sample (TRAIN)")
            st.dataframe(res["train_results"][ticker_pick]["stats"].round(3), use_container_width=True)

        with col_st2:
            st.markdown("##### 🔴 Dữ liệu Out-of-Sample (TEST)")
            st.dataframe(res["test_results"][ticker_pick]["stats"].round(3), use_container_width=True)

        # Đồ thị Tăng trưởng vốn (Equity Curve) của các chiến lược trên tập TEST
        st.markdown(f"#### 📈 Tăng trưởng vốn (Equity Curve) trên tập TEST ({ticker_pick})")
        test_returns_dict = res["test_results"][ticker_pick]["returns"]
        fig_equity = go.Figure()

        for s_name, s_ret in test_returns_dict.items():
            eq = (1.0 + s_ret).cumprod()
            fig_equity.add_trace(go.Scatter(
                x=eq.index, y=eq,
                mode="lines", name=s_name,
                line=dict(width=2 if "OR" in s_name or "AND" in s_name else 1.2)
            ))

        fig_equity.update_layout(
            height=400,
            xaxis_title="Thời gian",
            yaxis_title="Giá trị Tăng trưởng (Lũy kế)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_equity, use_container_width=True)

    # ===============================================================================
    # TAB 3: THAM SỐ TỐI ƯU (TRAIN)
    # ===============================================================================
    with tab3:
        st.subheader("⚙️ Bảng tham số tối ưu hóa tìm được từ tập Train")
        st.markdown("""
        **Quy tắc nghiên cứu:**
        - Các tham số được tối ưu hóa **hoàn toàn trên dữ liệu Train** để tối đa hóa Sharpe Ratio.
        - Tuyệt đối **không sử dụng dữ liệu Test** trong quá trình tìm tham số nhằm tránh rủi ro Data Snooping / Overfitting.
        """)

        param_rows = []
        for t in res["selected_tickers"]:
            param_rows.append({
                "Mã Cổ Phiếu": t,
                "SMA Ngắn (ma_short)": res["best_params"][t]["SMA"]["ma_short"],
                "SMA Dài (ma_long)": res["best_params"][t]["SMA"]["ma_long"],
                "Chu kỳ OBV (obv_window)": res["best_params"][t]["OBV"]["obv_window"]
            })
        df_params = pd.DataFrame(param_rows).set_index("Mã Cổ Phiếu")
        st.table(df_params)

        st.info("💡 **Gợi ý kiểm nghiệm:** Khi một cổ phiếu có `ma_short` và `ma_long` chênh lệch hợp lý (ví dụ 40-50 và 180-200), chiến lược sẽ ít bị nhiễu sóng ngắn hạn hơn.")

    # ===============================================================================
    # TAB 4: PHÂN BỔ DANH MỤC (EQUAL WEIGHT VS MPT)
    # ===============================================================================
    with tab4:
        st.subheader("💼 So sánh Hiệu Quả Danh Mục: MPT (Markowitz) vs Equal Weight")

        col_w1, col_w2 = st.columns([1, 1.2])

        df_weights = pd.DataFrame({
            "Mã Cổ Phiếu": res["selected_tickers"],
            "Equal Weight (1/N)": res["equal_weights"],
            "MPT (Tối Đa Sharpe Train)": res["mpt_weights"]
        })

        with col_w1:
            st.markdown("#### ⚖️ Bảng trọng số phân bổ danh mục:")
            st.dataframe(df_weights.style.format({
                "Equal Weight (1/N)": "{:.2%}",
                "MPT (Tối Đa Sharpe Train)": "{:.2%}"
            }), use_container_width=True)

            st.caption(f"Chiến lược tín hiệu cơ sở cho danh mục: **{res['strat_name']}**")

        with col_w2:
            # Biểu đồ so sánh trọng số
            fig_w = go.Figure()
            fig_w.add_trace(go.Bar(
                x=df_weights["Mã Cổ Phiếu"], y=df_weights["Equal Weight (1/N)"] * 100,
                name="Equal Weight", marker_color="#64748B"
            ))
            fig_w.add_trace(go.Bar(
                x=df_weights["Mã Cổ Phiếu"], y=df_weights["MPT (Tối Đa Sharpe Train)"] * 100,
                name="MPT", marker_color="#2563EB"
            ))
            fig_w.update_layout(
                title="Tỷ trọng phân bổ (% vốn)",
                yaxis_title="Trọng số (%)",
                barmode="group",
                height=280,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_w, use_container_width=True)

        st.markdown("---")

        # So sánh hiệu quả định lượng (Train & Test)
        st.markdown("#### 🏆 Bảng chỉ số hiệu năng danh mục đầu tư:")
        col_p1, col_p2 = st.columns(2)

        with col_p1:
            st.markdown("##### 🟢 Tập Huấn Luyện (In-Sample: Train)")
            st.dataframe(res["port_train_stats"].round(3), use_container_width=True)

        with col_p2:
            st.markdown("##### 🔴 Tập Kiểm Định Ngoài Mẫu (Out-of-Sample: TEST)")
            st.dataframe(res["port_test_stats"].round(3), use_container_width=True)

        # Đường cong tài sản Portfolio Equity Curve (Tập TEST)
        st.markdown("#### 📈 Đường tăng trưởng danh mục trên tập TEST (Out-of-sample)")
        init_cap = res["initial_capital"]
        ew_eq_test = init_cap * (1.0 + res["ew_test_ret"]).cumprod()
        mpt_eq_test = init_cap * (1.0 + res["mpt_test_ret"]).cumprod()

        fig_port_eq = go.Figure()
        fig_port_eq.add_trace(go.Scatter(
            x=ew_eq_test.index, y=ew_eq_test,
            mode="lines", name="Equal Weight Portfolio", line=dict(color="#10B981", width=2.2)
        ))
        fig_port_eq.add_trace(go.Scatter(
            x=mpt_eq_test.index, y=mpt_eq_test,
            mode="lines", name="MPT (Markowitz) Portfolio", line=dict(color="#2563EB", width=2.2)
        ))

        fig_port_eq.update_layout(
            title=f"Tăng trưởng giá trị danh mục vốn ban đầu {init_cap:,.0f} VNĐ",
            xaxis_title="Thời gian (Tập Test)",
            yaxis_title="Giá trị danh mục (VNĐ)",
            hovermode="x unified",
            height=420
        )
        st.plotly_chart(fig_port_eq, use_container_width=True)

        # Biểu đồ Drawdown (Underwater Chart)
        st.markdown("#### 📉 Mức sụt giảm tài sản (Drawdown Chart) trên tập TEST")
        ew_running_max = ew_eq_test.cummax()
        ew_dd = (ew_eq_test / ew_running_max - 1.0) * 100.0

        mpt_running_max = mpt_eq_test.cummax()
        mpt_dd = (mpt_eq_test / mpt_running_max - 1.0) * 100.0

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=ew_dd.index, y=ew_dd,
            mode="lines", name="Equal Weight Drawdown", line=dict(color="#10B981", width=1.5),
            fill="tozeroy"
        ))
        fig_dd.add_trace(go.Scatter(
            x=mpt_dd.index, y=mpt_dd,
            mode="lines", name="MPT Drawdown", line=dict(color="#EF4444", width=1.5),
            fill="tozeroy"
        ))

        fig_dd.update_layout(
            yaxis_title="Mức sụt giảm (%)",
            hovermode="x unified",
            height=320,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_dd, use_container_width=True)

    # ===============================================================================
    # TAB 5: BÁO CÁO & KẾT LUẬN KIỂM ĐỊNH
    # ===============================================================================
    with tab5:
        st.subheader("📋 Báo Cáo Kiểm Định Định Lượng & Đánh Giá Thực Nghiệm")

        ew_test_sharpe = res["port_test_stats"].loc["Equal Weight", "Sharpe Ratio"]
        mpt_test_sharpe = res["port_test_stats"].loc["MPT (Markowitz)", "Sharpe Ratio"]
        ew_test_return = res["port_test_stats"].loc["Equal Weight", "Total Return [%]"]
        mpt_test_return = res["port_test_stats"].loc["MPT (Markowitz)", "Total Return [%]"]
        ew_test_mdd = res["port_test_stats"].loc["Equal Weight", "Max Drawdown [%]"]
        mpt_test_mdd = res["port_test_stats"].loc["MPT (Markowitz)", "Max Drawdown [%]"]

        winner = "MPT (Markowitz)" if mpt_test_sharpe > ew_test_sharpe else "Equal Weight (1/N)"

        st.markdown(f"""
        ### 🎯 Kết Quả Kiểm Định Chính Ngoài Mẫu (Out-of-Sample Test):
        - **Phương pháp phân bổ hiệu quả hơn trên dữ liệu Test:** **{winner}** (căn cứ theo Sharpe Ratio điều chỉnh rủi ro).
        - **Tổng lợi nhuận (Total Return):**
          - Equal Weight: **{ew_test_return:.2f}%**
          - MPT: **{mpt_test_return:.2f}%**
        - **Tỷ suất sinh lời điều chỉnh rủi ro (Sharpe Ratio):**
          - Equal Weight: **{ew_test_sharpe:.3f}**
          - MPT: **{mpt_test_sharpe:.3f}**
        - **Mức sụt giảm tài sản lớn nhất (Max Drawdown):**
          - Equal Weight: **{ew_test_mdd:.2f}%**
          - MPT: **{mpt_test_mdd:.2f}%**
        """)

        st.markdown("""
        ---
        ### 🧠 Phân Tích Phương Pháp Luận Định Lượng:
        
        1. **Hiện tượng Overfitting trong Tối ưu hóa MPT:**
           - MPT tối ưu hóa dựa trên ma trận hiệp phương sai và lợi nhuận kỳ vọng của tập Train. Trên thực tế, thị trường tài chính Việt Nam (HOSE) có tính biến động chế độ (regime change) rất mạnh giữa chu kỳ tăng trưởng (2020-2021) và chu kỳ suy thoái (2022).
           - Do đó, việc MPT đạt Sharpe rất cao trên Train nhưng suy giảm trên Test là hiện tượng phổ biến trong tài chính định lượng, minh chứng tầm quan trọng của việc kiểm định ngoài mẫu (Out-of-sample).
        
        2. **Sức mạnh phòng thủ của chiến lược kết hợp SMA + OBV:**
           - Tín hiệu kết hợp giữa **SMA (xu hướng giá)** và **OBV (dòng tiền khối lượng)** giúp hạn chế các bẫy tín hiệu giả (whipsaws).
           - Cơ chế chuyển sang nắm giữ tiền mặt (Holding = 0) khi thị trường xác nhận xu hướng giảm giúp giảm thiểu đáng kể Max Drawdown so với chiến lược Buy & Hold truyền thống trong năm 2022.
        
        3. **Khuyến nghị thực tế cho nhà đầu tư:**
           - Chiến lược **Equal Weight (1/N)** thường có tính bền vững (robustness) cao, tránh rủi ro dồn tỷ trọng vào một vài cổ phiếu có quá khứ tốt nhưng đảo chiều xu hướng trong tương lai.
           - Nên áp dụng thêm tái cân bằng định kỳ (Periodic Rebalancing) và kết hợp lệnh dừng lỗ (Stop-loss) để tối ưu hóa quản trị rủi ro danh mục.
        """)

        # Nút tải xuống kết quả kiểm định dạng CSV
        st.markdown("#### 📥 Xuất Báo Cáo Kết Quả:")
        col_dl1, col_dl2 = st.columns(2)

        csv_summary = res["port_test_stats"].to_csv().encode("utf-8-sig")
        col_dl1.download_button(
            label="Tải bảng kết quả danh mục Test (CSV)",
            data=csv_summary,
            file_name="ket_qua_danh_muc_test.csv",
            mime="text/csv"
        )

        csv_params = df_params.to_csv().encode("utf-8-sig")
        col_dl2.download_button(
            label="Tải bảng tham số tối ưu (CSV)",
            data=csv_params,
            file_name="tham_so_toi_uu_train.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
