import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# --- 1. ページ設定 ---
st.set_page_config(layout="wide", page_title="Realtime Speed FX")
st.title("⚡ Speed Synthetic Forecaster (Fixed Scale)")

# --- 2. CSS ---
st.markdown("""
    <style>
    .block-container {padding-top: 0.5rem; padding-bottom: 5rem;}
    [data-testid="stSidebar"] { min-width: 320px; }
    .js-plotly-plot .plotly .main-svg { margin-top: 0px; margin-bottom: 0px; }
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
show_shadows = st.sidebar.checkbox("相関ペアの影を表示", value=True)
show_forecast = st.sidebar.checkbox("未来予測線を表示", value=True)

# --- 5. データ取得 (分割ロード方式) ---

def clean_df(df):
    if df is None or df.empty: return df
    if isinstance(df.columns, pd.MultiIndex):
        try:
            if 'Ticker' in df.columns.names:
                df.columns = df.columns.droplevel('Ticker')
            elif df.columns.nlevels > 1:
                df.columns = df.columns.droplevel(1)
        except: pass
        
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    df = df.loc[:, ~df.columns.duplicated()]
    return df.dropna()

@st.cache_data(ttl=15, show_spinner=False)
def fetch_main_ticker():
    try:
        df = yf.download(MAIN_TICKER, period="1d", interval="1m", progress=False)
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

# --- 6. 分析ロジック ---

def get_correlations(base_df, bulk_1m):
    corrs = {}
    base_ret = base_df['Close'].pct_change()
    
    for ticker in SUB_TICKERS:
        df = extract_from_bulk(bulk_1m, ticker)
        if df.empty: continue
        aligned = df['Close'].reindex(base_df.index, method='ffill').fillna(method='bfill')
        if len(aligned) < 30: continue
        corr = base_ret.corr(aligned.pct_change())
        if not np.isnan(corr):
            corrs[ticker] = corr
            
    pos = sorted([(k, v) for k, v in corrs.items() if v > 0], key=lambda x: x[1], reverse=True)[:2]
    neg = sorted([(k, v) for k, v in corrs.items() if v < 0], key=lambda x: x[1])[:2]
    return pos, neg

def analyze_trend(df):
    if len(df) < 30: return 0, ""
    score = 0
    reasons = []
    
    def v(s): return float(s.iloc[-1].item()) if isinstance(s.iloc[-1], (pd.Series, np.ndarray)) else float(s.iloc[-1])
    
    c = v(df['Close'])
    ma5 = v(df['Close'].rolling(5).mean())
    ma13 = v(df['Close'].rolling(13).mean())
    ma25 = v(df['Close'].rolling(25).mean())
    
    if ma5 > ma13 > ma25:
        score += 2
        reasons.append("MA↑")
    elif ma5 < ma13 < ma25:
        score -= 2
        reasons.append("MA↓")
        
    if c > ma5: score += 1
    elif c < ma5: score -= 1
    
    return score, ",".join(reasons)

def calc_power(bulk_1h):
    scores = {"USD": 0, "JPY": 0, "EUR": 0, "GBP": 0, "AUD": 0}
    
    def chg(tk):
        df = extract_from_bulk(bulk_1h, tk)
        if df.empty: return 0
        try:
            c = float(df['Close'].iloc[-1])
            o = float(df['Open'].iloc[-1])
            return (c - o) / o * 100
        except: return 0
        
    uj = chg("USDJPY=X")
    eu = chg("EURUSD=X")
    gu = chg("GBPUSD=X")
    au = chg("AUDUSD=X")
    
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
        st.error("Failed to load USD/JPY. Retrying...")
        return

    # ★修正ポイント：表示用データのY軸範囲を計算（直近60分）
    df_jst = main_df.copy()
    df_jst.index = df_jst.index + timedelta(hours=9)
    
    # 直近60本のデータを取り出す
    visible_df = df_jst.iloc[-60:]
    
    # 最小値・最大値を計算し、少し余白(20%)を持たせる
    y_min = visible_df['Low'].min()
    y_max = visible_df['High'].max()
    y_padding = (y_max - y_min) * 0.2
    
    # もし値動きがなさすぎてmin=maxの場合は強制的に幅を持たせる
    if y_padding == 0: y_padding = 0.05
    
    y_range = [y_min - y_padding, y_max + y_padding]

    # チャート作成
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=df_jst.index, open=df_jst['Open'], high=df_jst['High'], low=df_jst['Low'], close=df_jst['Close'],
        name='USD/JPY', increasing_line_color='#FF4136', decreasing_line_color='#0074D9'
    ))

    # 相関データの取得
    bulk_1m = fetch_sub_tickers_1m()
    bulk_1h = fetch_sub_tickers_1h()
    
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
                analysis_log.append(f"📈 {NAME_MAP.get(tk, tk)}({corr:.2f}): {r}")
                
        for tk, corr in neg_top2:
            df = extract_from_bulk(bulk_1h, tk)
            if not df.empty:
                s, r = analyze_trend(df)
                total_pressure += s * corr
                analysis_log.append(f"📉 {NAME_MAP.get(tk, tk)}({corr:.2f}): {r}")

    # 影と予測線
    if analysis_ready and show_shadows:
        cols_pos = ['#FFA500', '#FFD700']
        for i, (tk, corr) in enumerate(pos_top2):
            df = extract_from_bulk(bulk_1m, tk)
            if not df.empty:
                shad = normalize(df['Close'], main_df['Close'], False)
                shad.index += timedelta(hours=9)
                fig.add_trace(go.Scatter(x=shad.index, y=shad, mode='lines', 
                    name=f"{NAME_MAP.get(tk,tk)}", line=dict(color=cols_pos[i], width=1), opacity=0.6))
                    
        cols_neg = ['#00FFFF', '#1E90FF']
        for i, (tk, corr) in enumerate(neg_top2):
            df = extract_from_bulk(bulk_1m, tk)
            if not df.empty:
                shad = normalize(df['Close'], main_df['Close'], True)
                shad.index += timedelta(hours=9)
                fig.add_trace(go.Scatter(x=shad.index, y=shad, mode='lines', 
                    name=f"{NAME_MAP.get(tk,tk)} (Inv)", line=dict(color=cols_neg[i], width=1, dash='dot'), opacity=0.6))

    if analysis_ready and show_forecast:
        last_t = df_jst.index[-1]
        last_p = float(df_jst['Close'].iloc[-1])
        fut_t = last_t + timedelta(minutes=30)
        vol = float(df_jst['Close'].diff().std()) * 15
        if np.isnan(vol): vol = 0.05
        fut_p = last_p + (total_pressure * vol * 0.3)
        col = "#32CD32" if total_pressure > 0 else "#FF00FF"
        
        fig.add_trace(go.Scatter(x=[last_t, fut_t], y=[last_p, fut_p],
            mode='lines+markers', marker=dict(symbol='arrow-right', size=10),
            name='Forecast', line=dict(color=col, width=4, dash='dash')))

    # レイアウト設定
    fig.update_layout(
        height=700, template="plotly_dark",
        # X軸: 直近60分 + 未来35分
        xaxis=dict(range=[df_jst.index[-60], df_jst.index[-1] + timedelta(minutes=35)], showgrid=False),
        # ★Y軸: ドル円の動きに合わせて強制固定（影に影響されない）
        yaxis=dict(
            side="right", showgrid=True, gridcolor='rgba(128,128,128,0.2)',
            range=y_range  # ここで計算した範囲を適用
        ),
        margin=dict(l=0, r=0, t=10, b=0), showlegend=True, 
        legend=dict(x=0, y=1, bgcolor='rgba(0,0,0,0.5)')
    )
    
    # 画面描画
    c_chart, c_info = st.columns([3, 1])
    with c_chart:
        st.plotly_chart(fig, use_container_width=True)
        
    with c_info:
        if analysis_ready:
            st.subheader("🤖 AI Decision")
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
            st.info("Analyzing... (Background)")

render_dashboard()
