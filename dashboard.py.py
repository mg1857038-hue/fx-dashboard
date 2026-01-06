import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- 1. ページ設定 ---
st.set_page_config(layout="wide", page_title="FX Cockpit (yfinance Ver.)")
st.title("⚡ FX Cockpit (Evolution Ver.)")

# --- 2. Session State ---
if 'forecast_history' not in st.session_state:
    st.session_state.forecast_history = [] 

# --- 3. CSS ---
st.markdown("""
    <style>
    .block-container {padding-top: 0.5rem; padding-bottom: 5rem;}
    [data-testid="stSidebar"] { min-width: 300px; }
    .status-badge {
        padding: 5px 10px; border-radius: 5px; font-weight: bold; color: white; text-align: center; margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 4. 設定 ---
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

st.sidebar.header("Control Panel")
c_a1, c_a2 = st.sidebar.columns([1,2])
auto_refresh = c_a1.toggle("Live", value=True)
refresh_rate = c_a2.slider("Interval (sec)", 10, 60, 15)
st.sidebar.divider()
show_shadows = st.sidebar.checkbox("Shadows", value=True)
show_forecast = st.sidebar.checkbox("Forecast Arrow", value=True)
show_trails = st.sidebar.checkbox("Trails (History)", value=True)

# --- 5. データ取得 (yfinance) ---

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

# --- 6. 計算ロジック (ATR / ブレーキ / 軌跡) ---

def calculate_technical_filters(df):
    if len(df) < 20: return 1.0, []
    close = df['Close']
    reasons = []
    brake_factor = 1.0
    
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]
    
    if current_rsi > 70:
        reasons.append(f"RSI Over({current_rsi:.0f})")
        brake_factor *= 0.5
    elif current_rsi < 30:
        reasons.append(f"RSI Under({current_rsi:.0f})")
        brake_factor *= 0.5

    # BB
    sma = close.rolling(20).mean()
    std = close.rolling(20).std()
    upper = sma + (std * 2)
    lower = sma - (std * 2)
    c = close.iloc[-1]
    
    if c > upper.iloc[-1]:
        reasons.append("BB High")
        brake_factor *= 0.3
    elif c < lower.iloc[-1]:
        reasons.append("BB Low")
        brake_factor *= 0.3
        
    return brake_factor, reasons

def calculate_atr_status(df):
    if len(df) < 100: return "UNKNOWN", "#888888", 0
    tr = np.maximum(df['High'] - df['Low'], 
           np.maximum(np.abs(df['High'] - df['Close'].shift()), 
                      np.abs(df['Low'] - df['Close'].shift())))
    atr = tr.rolling(14).mean()
    curr, mean = atr.iloc[-1], atr.iloc[-300:].mean()
    if np.isnan(curr) or np.isnan(mean) or mean == 0: return "UNKNOWN", "#888888", 0
    
    ratio = curr / mean
    if ratio < 0.6: return "DEAD (Low Vol)", "#FF4136", ratio
    elif ratio > 2.0: return "VOLATILE", "#FF851B", ratio
    return "ACTIVE", "#2ECC40", ratio

def get_correlations_and_trends(main_df, bulk_1m, bulk_1h):
    corrs = {}
    base_ret = main_df['Close'].pct_change()
    for tk in SUB_TICKERS:
        df = extract_from_bulk(bulk_1m, tk)
        if df.empty: continue
        aligned = df['Close'].reindex(main_df.index, method='ffill').fillna(method='bfill')
        if len(aligned) < 30: continue
        c = base_ret.corr(aligned.pct_change())
        if not np.isnan(c): corrs[tk] = c
            
    pos = sorted([(k, v) for k, v in corrs.items() if v > 0], key=lambda x: x[1], reverse=True)[:2]
    neg = sorted([(k, v) for k, v in corrs.items() if v < 0], key=lambda x: x[1])[:2]
    
    pressure = 0
    logs = []
    
    def get_trend(df):
        c, m5, m13 = df['Close'].iloc[-1], df['Close'].rolling(5).mean().iloc[-1], df['Close'].rolling(13).mean().iloc[-1]
        score = 0
        if m5 > m13: score = 1
        elif m5 < m13: score = -1
        if c > m5: score += 1
        elif c < m5: score -= 1
        return score

    for tk, corr in pos:
        df = extract_from_bulk(bulk_1h, tk)
        if not df.empty:
            s = get_trend(df)
            pressure += s * corr
            logs.append(f"📈 {NAME_MAP.get(tk,tk)}({corr:.2f})")
            
    for tk, corr in neg:
        df = extract_from_bulk(bulk_1h, tk)
        if not df.empty:
            s = get_trend(df)
            pressure += s * corr
            logs.append(f"📉 {NAME_MAP.get(tk,tk)}({corr:.2f})")
            
    return pressure, logs, pos, neg

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
    st.caption(f"Last Update: {datetime.now().strftime('%H:%M:%S')} (Source: yfinance)")
    
    main_df = fetch_main_ticker()
    if main_df is None or main_df.empty:
        st.error("Waiting for Data...")
        return

    status, status_col, atr_ratio = calculate_atr_status(main_df)
    brake_factor, brake_reasons = calculate_technical_filters(main_df)
    
    # 日本時間調整 & 範囲
    df_jst = main_df.copy()
    df_jst.index = df_jst.index + timedelta(hours=9)
    visible_df = df_jst.iloc[-60:]
    y_min, y_max = visible_df['Low'].min(), visible_df['High'].max()
    y_pad = (y_max - y_min) * 0.2 if (y_max - y_min) > 0 else 0.05

    bulk_1m, bulk_1h = fetch_sub_tickers_1m(), fetch_sub_tickers_1h()
    
    pressure = 0
    logs = []
    pos_top2, neg_top2 = [], []
    
    if bulk_1m is not None and bulk_1h is not None:
        pressure, logs, pos_top2, neg_top2 = get_correlations_and_trends(main_df, bulk_1m, bulk_1h)
        
    final_pressure = pressure * brake_factor
    if status.startswith("DEAD"): final_pressure = 0
        
    # Trails
    last_t = df_jst.index[-1]
    last_p = float(df_jst['Close'].iloc[-1])
    vol = float(df_jst['Close'].diff().std()) * 15
    if np.isnan(vol): vol = 0.05
    fut_t = last_t + timedelta(minutes=30)
    fut_p = last_p + (final_pressure * vol * 0.3)
    
    history = st.session_state.forecast_history
    if not history or history[-1][0] != fut_t:
        history.append((fut_t, fut_p))
        if len(history) > 30: history.pop(0)
        st.session_state.forecast_history = history

    # Draw
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_jst.index, open=df_jst['Open'], high=df_jst['High'], low=df_jst['Low'], close=df_jst['Close'],
        name='USD/JPY', increasing_line_color='#FF4136', decreasing_line_color='#0074D9'
    ))
    
    if show_shadows and bulk_1m is not None:
        cols = ['#FFA500', '#FFD700']
        for i, (tk, c) in enumerate(pos_top2):
            df = extract_from_bulk(bulk_1m, tk)
            if not df.empty:
                s = normalize(df['Close'], main_df['Close'], False)
                s.index += timedelta(hours=9)
                fig.add_trace(go.Scatter(x=s.index, y=s, mode='lines', name=NAME_MAP.get(tk,tk), line=dict(color=cols[i], width=1), opacity=0.6))
        cols = ['#00FFFF', '#1E90FF']
        for i, (tk, c) in enumerate(neg_top2):
            df = extract_from_bulk(bulk_1m, tk)
            if not df.empty:
                s = normalize(df['Close'], main_df['Close'], True)
                s.index += timedelta(hours=9)
                fig.add_trace(go.Scatter(x=s.index, y=s, mode='lines', name=NAME_MAP.get(tk,tk)+"(Inv)", line=dict(color=cols[i], width=1, dash='dot'), opacity=0.6))

    if show_trails and history:
        trail_x = [h[0] for h in history]
        trail_y = [h[1] for h in history]
        fig.add_trace(go.Scatter(
            x=trail_x, y=trail_y, mode='markers', name='Trails', 
            marker=dict(color='yellow', size=8, symbol='x', line=dict(width=2, color='white'))
        ))

    if show_forecast:
        col = "#32CD32" if final_pressure > 0 else "#FF00FF"
        if final_pressure == 0: col = "#888888"
        fig.add_trace(go.Scatter(
            x=[last_t, fut_t], y=[last_p, fut_p], 
            mode='lines+markers', marker=dict(symbol='arrow-right', size=10),
            name='Forecast', line=dict(color=col, width=4, dash='dash')
        ))

    fig.update_layout(
        height=700, template="plotly_dark",
        xaxis=dict(range=[df_jst.index[-60], df_jst.index[-1] + timedelta(minutes=35)], showgrid=False),
        yaxis=dict(side="right", showgrid=True, gridcolor='rgba(128,128,128,0.2)', range=[y_min-y_pad, y_max+y_pad]),
        margin=dict(l=0, r=0, t=10, b=0), showlegend=True, legend=dict(x=0, y=1, bgcolor='rgba(0,0,0,0.5)')
    )
    
    c_chart, c_info = st.columns([3, 1])
    with c_chart:
        st.markdown(f"<div class='status-badge' style='background-color:{status_col};'>MARKET: {status} (ATR Ratio: {atr_ratio:.2f})</div>", unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        
    with c_info:
        st.subheader("🤖 AI Decision")
        p_text = f"{final_pressure:.1f}"
        if final_pressure == 0:
            if status.startswith("DEAD"): st.warning("NO TRADE (Low Vol)")
            elif brake_factor < 1.0: st.warning(f"BLOCKED BY TECH\n({', '.join(brake_reasons)})")
            else: st.warning("NEUTRAL")
        elif final_pressure > 3: st.success(f"STRONG BUY ({p_text})")
        elif final_pressure > 0: st.info(f"BUY ({p_text})")
        elif final_pressure < -3: st.error(f"STRONG SELL ({p_text})")
        else: st.warning(f"SELL ({p_text})")
            
        if brake_factor < 1.0: st.caption(f"⚠️ {', '.join(brake_reasons)}")
        st.divider()
        st.caption("Drivers")
        for l in logs: st.text(l)

render_dashboard()
