import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- 1. ページ設定 ---
st.set_page_config(layout="wide", page_title="FX Cockpit (ATR Filter)")
st.title("⚡ FX Synthetic Cockpit (ATR Filter)")

# --- 2. CSS ---
st.markdown("""
    <style>
    .block-container {padding-top: 0.5rem; padding-bottom: 5rem;}
    [data-testid="stSidebar"] { min-width: 300px; }
    .js-plotly-plot .plotly .main-svg { margin-top: 0px; margin-bottom: 0px; }
    /* ステータスバッジのデザイン */
    .status-badge {
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
        color: white;
        text-align: center;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. 分析対象設定 ---
MAIN_TICKER = "USDJPY=X"
SUB_TICKERS = [
    "EURUSD=X", "GBPUSD=X", "AUDUSD=X", "USDCHF=X", "USDCAD=X", "NZDUSD=X",
    "EURJPY=X", "GBPJPY=X", "AUDJPY=X", "EURGBP=X", "AUDNZD=X"
]
NAME_MAP = {
    "USDJPY=X": "USD/JPY", "EURUSD=X": "EUR/USD", "GBPUSD=X": "GBP/USD",
    "AUDUSD=X": "AUD/USD", "USDCHF=X": "USD/CHF", "USDCAD=X": "USD/CAD",
    "NZDUSD=X": "NZD/USD", "EURJPY=X": "EUR/JPY", "GBPJPY=X": "GBP/JPY",
    "AUDJPY=X": "AUD/JPY", "EURGBP=X": "EUR/GBP", "AUDNZD=X": "AUD/NZD"
}

# --- 4. サイドバー ---
st.sidebar.header("Control Panel")
c_a1, c_a2 = st.sidebar.columns([1,2])
auto_refresh = c_a1.toggle("Live", value=True)
refresh_rate = c_a2.slider("Interval (sec)", 10, 60, 15)

st.sidebar.divider()
show_shadows = st.sidebar.checkbox("Shadows (Correlations)", value=True)
show_forecast = st.sidebar.checkbox("Forecast Line", value=True)

# --- 5. データ取得 ---

def clean_df(df):
    if df is None or df.empty: return df
    if isinstance(df.columns, pd.MultiIndex):
        try:
            if 'Ticker' in df.columns.names: df.columns = df.columns.droplevel('Ticker')
            elif df.columns.nlevels > 1: df.columns = df.columns.droplevel(1)
        except: pass
    if df.index.tz is not None: df.index = df.index.tz_localize(None)
    df = df.loc[:, ~df.columns.duplicated()]
    return df.dropna()

@st.cache_data(ttl=15, show_spinner=False)
def fetch_main_ticker():
    try:
        # ATR計算のために少し長めに取る
        df = yf.download(MAIN_TICKER, period="5d", interval="1m", progress=False)
        return clean_df(df)
    except: return None

@st.cache_data(ttl=30, show_spinner=False)
def fetch_sub_tickers_1m():
    try:
        data = yf.download(SUB_TICKERS, period="1d", interval="1m", group_by='ticker', progress=False, threads=True)
        return data
    except: return None

@st.cache_data(ttl=300, show_spinner=False)
def fetch_sub_tickers_1h():
    try:
        all_tickers = [MAIN_TICKER] + SUB_TICKERS
        data = yf.download(all_tickers, period="5d", interval="1h", group_by='ticker', progress=False, threads=True)
        return data
    except: return None

def extract_from_bulk(bulk_data, ticker):
    try:
        if bulk_data is None or bulk_data.empty: return pd.DataFrame()
        if isinstance(bulk_data.columns, pd.MultiIndex):
            if ticker in bulk_data.columns.levels[0]:
                df = bulk_data[ticker].copy()
                return clean_df(df)
        return pd.DataFrame()
    except: return pd.DataFrame()

# --- 6. 分析ロジック (ATR追加) ---

def calculate_atr(df, period=14):
    """ATR (Average True Range) を計算"""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(period).mean()
    return atr

def get_market_status(df):
    """ATRに基づいて市場の状態を判定"""
    if len(df) < 100: return "UNKNOWN", "#888888", 0
    
    atr = calculate_atr(df)
    current_atr = atr.iloc[-1]
    
    # 過去24時間の平均ATRと比較
    # 1分足なので 60*24 = 1440本だが、直近数時間の平均と比較する
    recent_mean_atr = atr.iloc[-300:].mean() # 直近5時間の平均
    
    if np.isnan(current_atr) or np.isnan(recent_mean_atr): return "UNKNOWN", "#888888", 0
    
    ratio = current_atr / recent_mean_atr
    
    status = "ACTIVE"
    color = "#2ECC40" # Green
    
    if ratio < 0.6: # 平均の60%未満しかない
        status = "DEAD (No Volatility)"
        color = "#FF4136" # Red
    elif ratio > 2.0: # 平均の2倍以上暴れている
        status = "VOLATILE (Caution)"
        color = "#FF851B" # Orange
        
    return status, color, ratio

def get_correlations(base_df, bulk_1m):
    corrs = {}
    base_ret = base_df['Close'].pct_change()
    for ticker in SUB_TICKERS:
        df = extract_from_bulk(bulk_1m, ticker)
        if df.empty: continue
        aligned = df['Close'].reindex(base_df.index, method='ffill').fillna(method='bfill')
        if len(aligned) < 30: continue
        corr = base_ret.corr(aligned.pct_change())
        if not np.isnan(corr): corrs[ticker] = corr
    pos = sorted([(k, v) for k, v in corrs.items() if v > 0], key=lambda x: x[1], reverse=True)[:2]
    neg = sorted([(k, v) for k, v in corrs.items() if v < 0], key=lambda x: x[1])[:2]
    return pos, neg

def analyze_trend(df):
    if len(df) < 30: return 0, ""
    score = 0
    reasons = []
    def v(s): return float(s.iloc[-1].item()) if isinstance(s.iloc[-1], (pd.Series, np.ndarray)) else float(s.iloc[-1])
    c, m5, m13, m25 = v(df['Close']), v(df['Close'].rolling(5).mean()), v(df['Close'].rolling(13).mean()), v(df['Close'].rolling(25).mean())
    if m5 > m13 > m25: score += 2; reasons.append("MA↑")
    elif m5 < m13 < m25: score -= 2; reasons.append("MA↓")
    if c > m5: score += 1
    elif c < m5: score -= 1
    return score, ",".join(reasons)

def calc_power(bulk_1h):
    scores = {"USD": 0, "JPY": 0, "EUR": 0, "GBP": 0, "AUD": 0}
    def chg(tk):
        df = extract_from_bulk(bulk_1h, tk)
        if df.empty: return 0
        try: return (float(df['Close'].iloc[-1]) - float(df['Open'].iloc[-1])) / float(df['Open'].iloc[-1]) * 100
        except: return 0
    uj, eu, gu, au = chg("USDJPY=X"), chg("EURUSD=X"), chg("GBPUSD=X"), chg("AUDUSD=X")
    scores["USD"] += uj - eu - gu - au
    scores["JPY"] -= uj
    scores["EUR"] += eu
    scores["GBP"] += gu
    scores["AUD"] += au
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def normalize(target, base, invert=False):
    aligned = target.reindex(base.index, method='ffill').fillna(method='bfill')
    b_mean, b_std = base.mean(), base.std()
    t_mean, t_std = aligned.mean(), aligned.std()
    if t_std == 0: return base
    norm = (aligned - t_mean) / t_std * b_std + b_mean
    if invert: norm = b_mean - (norm - b_mean)
    return norm

# --- 7. メイン処理 ---

@st.fragment(run_every=refresh_rate if auto_refresh else None)
def render_dashboard():
    st.caption(f"Last Update: {datetime.now().strftime('%H:%M:%S')}")

    main_df = fetch_main_ticker()
    if main_df is None or main_df.empty:
        st.error("Waiting for Data...")
        return

    # ★ ATR判定
    status_text, status_color, atr_ratio = get_market_status(main_df)

    # チャート範囲計算 (直近60本)
    df_jst = main_df.copy()
    df_jst.index = df_jst.index + timedelta(hours=9)
    visible_df = df_jst.iloc[-60:]
    y_min, y_max = visible_df['Low'].min(), visible_df['High'].max()
    y_pad = (y_max - y_min) * 0.2 if (y_max - y_min) > 0 else 0.05
    
    # チャート
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_jst.index, open=df_jst['Open'], high=df_jst['High'], low=df_jst['Low'], close=df_jst['Close'],
        name='USD/JPY', increasing_line_color='#FF4136', decreasing_line_color='#0074D9'
    ))

    # 分析
    bulk_1m, bulk_1h = fetch_sub_tickers_1m(), fetch_sub_tickers_1h()
    analysis_ready = False
    pos_top2, neg_top2 = [], []
    power_ranking = []
    total_pressure = 0
    analysis_log = []

    if bulk_1m is not None and not bulk_1m.empty and bulk_1h is not None and not bulk_1h.empty:
        analysis_ready = True
        pos_top2, neg_top2 = get_correlations(main_df, bulk_1m)
        power_ranking = calc_power(bulk_1h)
        for tk, corr in pos_top2:
            df = extract_from_bulk(bulk_1h, tk)
            if not df.empty:
                s, r = analyze_trend(df)
                total_pressure += s * corr
                analysis_log.append(f"📈 {NAME_MAP.get(tk,tk)}({corr:.2f}): {r}")
        for tk, corr in neg_top2:
            df = extract_from_bulk(bulk_1h, tk)
            if not df.empty:
                s, r = analyze_trend(df)
                total_pressure += s * corr
                analysis_log.append(f"📉 {NAME_MAP.get(tk,tk)}({corr:.2f}): {r}")

    # 相関と予測
    if analysis_ready and show_shadows:
        cols = ['#FFA500', '#FFD700']
        for i, (tk, corr) in enumerate(pos_top2):
            df = extract_from_bulk(bulk_1m, tk)
            if not df.empty:
                shad = normalize(df['Close'], main_df['Close'], False)
                shad.index += timedelta(hours=9)
                fig.add_trace(go.Scatter(x=shad.index, y=shad, mode='lines', name=NAME_MAP.get(tk,tk), line=dict(color=cols[i], width=1), opacity=0.6))
        cols = ['#00FFFF', '#1E90FF']
        for i, (tk, corr) in enumerate(neg_top2):
            df = extract_from_bulk(bulk_1m, tk)
            if not df.empty:
                shad = normalize(df['Close'], main_df['Close'], True)
                shad.index += timedelta(hours=9)
                fig.add_trace(go.Scatter(x=shad.index, y=shad, mode='lines', name=NAME_MAP.get(tk,tk)+"(Inv)", line=dict(color=cols[i], width=1, dash='dot'), opacity=0.6))

    if analysis_ready and show_forecast:
        # ★ ATRが低すぎる(DEAD)ときは、予測線を消すか、灰色にする処理
        forecast_color = "#32CD32" if total_pressure > 0 else "#FF00FF"
        if status_text.startswith("DEAD"):
            forecast_color = "#888888" # 無効色
            total_pressure = 0 # 圧力なしとみなす

        last_t = df_jst.index[-1]
        last_p = float(df_jst['Close'].iloc[-1])
        fut_t = last_t + timedelta(minutes=30)
        vol = float(df_jst['Close'].diff().std()) * 15
        if np.isnan(vol): vol = 0.05
        fut_p = last_p + (total_pressure * vol * 0.3)
        
        fig.add_trace(go.Scatter(x=[last_t, fut_t], y=[last_p, fut_p], mode='lines+markers', marker=dict(symbol='arrow-right', size=10), name='Forecast', line=dict(color=forecast_color, width=4, dash='dash')))

    fig.update_layout(
        height=700, template="plotly_dark",
        xaxis=dict(range=[df_jst.index[-60], df_jst.index[-1] + timedelta(minutes=35)], showgrid=False),
        yaxis=dict(side="right", showgrid=True, gridcolor='rgba(128,128,128,0.2)', range=[y_min-y_pad, y_max+y_pad]),
        margin=dict(l=0, r=0, t=10, b=0), showlegend=True, legend=dict(x=0, y=1, bgcolor='rgba(0,0,0,0.5)')
    )
    
    c_chart, c_info = st.columns([3, 1])
    with c_chart:
        # ★ Market Status Badge
        st.markdown(f"<div class='status-badge' style='background-color:{status_color};'>MARKET STATUS: {status_text} (ATR Ratio: {atr_ratio:.2f})</div>", unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        
    with c_info:
        if analysis_ready:
            st.subheader("🤖 AI Decision")
            if status_text.startswith("DEAD"):
                st.warning("NO TRADE (Low Volatility)") # トレード禁止表示
            else:
                if total_pressure > 3: st.success(f"STRONG BUY ({total_pressure:.1f})")
                elif total_pressure > 0: st.info(f"BUY ({total_pressure:.1f})")
                elif total_pressure < -3: st.error(f"STRONG SELL ({total_pressure:.1f})")
                else: st.warning(f"NEUTRAL ({total_pressure:.1f})")
            
            st.divider()
            st.subheader("💪 Power")
            max_s = max([abs(x[1]) for x in power_ranking]) if power_ranking else 1
            for ccy, sc in power_ranking:
                norm = sc / max_s if max_s != 0 else 0
                bg = "#FF4136" if sc > 0 else "#0074D9"
                st.write(f"**{ccy}**")
                st.markdown(f"<div style='background:{bg};width:{abs(norm)*100}%;height:6px;border-radius:3px;'></div>", unsafe_allow_html=True)
            st.divider()
            st.caption("Drivers")
            for l in analysis_log: st.text(l)
        else:
            st.info("Initializing...")

render_dashboard()
