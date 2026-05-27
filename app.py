"""
시계열 데이터를 활용한 주식·코인 매수 타점 예측 프로그램
"""

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# ==========================================================
# 기본 설정
# ==========================================================

st.set_page_config(
    page_title="시계열 매수 타점",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================
# UI 스타일
# ==========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@400;500;600;700;800;900&display=swap');
:root{--ink:#0f172a;--muted:#64748b;--line:rgba(148,163,184,.28);--green:#10b981;--red:#ef4444;--blue:#2563eb;--card:rgba(255,255,255,.88)}
html,body,[class*="css"]{font-family:'Pretendard',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}.stApp{background:radial-gradient(circle at 12% 8%,rgba(37,99,235,.14),transparent 28%),radial-gradient(circle at 85% 0%,rgba(14,165,233,.14),transparent 30%),linear-gradient(180deg,#f8fbff 0%,#f4f7fb 45%,#eef3f8 100%);color:var(--ink)}.block-container{padding-top:1.05rem;padding-bottom:2.4rem;max-width:1540px}#MainMenu,footer,header{visibility:hidden}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0f172a 0%,#111827 52%,#0b1120 100%);border-right:1px solid rgba(255,255,255,.10);box-shadow:18px 0 44px rgba(15,23,42,.16)}[data-testid="stSidebar"] *{color:rgba(255,255,255,.92)!important}[data-testid="stSidebar"] label,[data-testid="stSidebar"] span,[data-testid="stSidebar"] p{color:rgba(255,255,255,.82)!important}[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{color:#fff!important;letter-spacing:-.02em}[data-testid="stSidebar"] hr{border-color:rgba(255,255,255,.10)}
.stButton>button{border:0!important;border-radius:14px!important;font-weight:800!important;background:linear-gradient(135deg,#2563eb,#06b6d4)!important;color:#fff!important;box-shadow:0 14px 26px rgba(37,99,235,.23)!important;transition:all .18s ease-in-out!important}.stButton>button:hover{transform:translateY(-1px);box-shadow:0 18px 34px rgba(37,99,235,.32)!important;filter:brightness(1.03)}
[data-baseweb="input"],[data-baseweb="select"]>div,textarea{border-radius:14px!important;border-color:rgba(148,163,184,.36)!important;background:rgba(255,255,255,.88)!important}[data-testid="stSidebar"] [data-baseweb="input"],[data-testid="stSidebar"] [data-baseweb="select"]>div,[data-testid="stSidebar"] textarea{background:rgba(255,255,255,.08)!important;border-color:rgba(255,255,255,.16)!important}
[data-testid="stMetric"]{background:rgba(255,255,255,.88);border:1px solid var(--line);border-radius:20px;padding:15px 16px;box-shadow:0 14px 34px rgba(15,23,42,.06)}[data-testid="stDataFrame"]{border-radius:18px;overflow:hidden;border:1px solid var(--line);box-shadow:0 12px 28px rgba(15,23,42,.05)}.stPlotlyChart{background:rgba(255,255,255,.82);border:1px solid var(--line);border-radius:22px;padding:10px;box-shadow:0 18px 46px rgba(15,23,42,.07)}div[role="radiogroup"]{gap:10px}div[role="radiogroup"] label{background:rgba(255,255,255,.78);border:1px solid var(--line);border-radius:999px;padding:8px 14px;box-shadow:0 10px 22px rgba(15,23,42,.05)}
.hero-card{position:relative;overflow:hidden;border-radius:28px;padding:26px 28px;margin-bottom:18px;background:linear-gradient(135deg,rgba(15,23,42,.96) 0%,rgba(30,41,59,.94) 42%,rgba(37,99,235,.88) 100%);border:1px solid rgba(255,255,255,.16);box-shadow:0 26px 60px rgba(15,23,42,.23);color:#fff}.hero-card:before{content:'';position:absolute;width:360px;height:360px;border-radius:999px;right:-120px;top:-170px;background:rgba(6,182,212,.28);filter:blur(4px)}.hero-content{position:relative;z-index:1}.hero-kicker{font-size:13px;font-weight:800;letter-spacing:.10em;color:#67e8f9;text-transform:uppercase;margin-bottom:8px}.hero-title{font-size:34px;line-height:1.14;font-weight:900;letter-spacing:-.055em;margin:0}.hero-subtitle{font-size:15px;color:rgba(255,255,255,.74);margin:10px 0 0 0;max-width:860px}.hero-stats{display:flex;gap:10px;flex-wrap:wrap;margin-top:18px}.hero-pill{padding:8px 12px;border-radius:999px;background:rgba(255,255,255,.10);border:1px solid rgba(255,255,255,.14);font-size:12px;font-weight:800;color:#fff}
.section-card{background:var(--card);border:1px solid var(--line);border-radius:24px;padding:18px 20px;margin:14px 0;box-shadow:0 18px 42px rgba(15,23,42,.07);backdrop-filter:blur(16px)}.asset-header{background:rgba(255,255,255,.88);border:1px solid var(--line);border-radius:26px;padding:20px 22px;margin:16px 0 14px 0;box-shadow:0 18px 42px rgba(15,23,42,.07)}.asset-symbol{font-size:14px;color:var(--muted);font-weight:800;margin-left:8px}.asset-price{font-size:32px;font-weight:900;letter-spacing:-.045em;color:var(--ink);margin-top:4px}.price-up{color:var(--green);font-weight:900;font-size:15px;margin-left:10px}.price-down{color:var(--red);font-weight:900;font-size:15px;margin-left:10px}.small-muted{font-size:12px;color:var(--muted)}
.main-title{display:none}.metric-card,.info-card{background:rgba(255,255,255,.88)!important;border:1px solid var(--line)!important;border-radius:20px!important;box-shadow:0 14px 34px rgba(15,23,42,.06)!important}.good{color:var(--green);font-weight:900}.bad{color:var(--red);font-weight:900}.warn{color:#f59e0b;font-weight:900}hr{border-color:var(--line)}

.compact-hero{padding:22px 26px;margin-bottom:16px;}
.compact-hero .hero-title{font-size:30px;margin:0;line-height:1.15;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
:root{--tv-bg:#0b1120;--tv-panel:#111827;--tv-panel2:#172033;--tv-line:rgba(148,163,184,.22);--tv-text:#e5e7eb;--tv-muted:#94a3b8;--card-blue:#3182f6;--card-green:#12b886;--card-red:#fa5252}
.block-container{max-width:1680px;padding-left:1.35rem;padding-right:1.35rem}.stApp{background:linear-gradient(180deg,#f5f8fc 0%,#eef3f9 100%)}
.hero-card{border-radius:30px;background:radial-gradient(circle at 80% -10%,rgba(20,184,166,.25),transparent 28%),radial-gradient(circle at 10% 10%,rgba(49,130,246,.25),transparent 32%),linear-gradient(135deg,#07111f 0%,#101827 52%,#14284b 100%);box-shadow:0 28px 68px rgba(2,6,23,.28)}.hero-title{font-size:38px}.hero-pill{background:rgba(255,255,255,.08);backdrop-filter:blur(8px)}
.section-card,.asset-header,.trade-panel,.signal-terminal,.chart-shell{border-radius:26px!important;background:rgba(255,255,255,.82)!important;border:1px solid rgba(148,163,184,.24)!important;box-shadow:0 22px 54px rgba(15,23,42,.08)!important;backdrop-filter:blur(18px)}
.terminal-card{background:linear-gradient(180deg,#0f172a,#111827)!important;color:#e5e7eb!important;border:1px solid rgba(148,163,184,.22)!important;border-radius:24px!important;padding:16px 18px!important;box-shadow:0 22px 56px rgba(15,23,42,.22)!important}.terminal-card .muted{color:#94a3b8;font-size:12px}
.badge{display:inline-flex;align-items:center;gap:6px;padding:6px 10px;border-radius:999px;font-size:12px;font-weight:900;letter-spacing:-.01em}.badge-green{background:rgba(18,184,134,.13);color:#12b886;border:1px solid rgba(18,184,134,.22)}.badge-red{background:rgba(250,82,82,.13);color:#fa5252;border:1px solid rgba(250,82,82,.22)}.badge-blue{background:rgba(49,130,246,.13);color:#3182f6;border:1px solid rgba(49,130,246,.22)}.badge-gray{background:rgba(148,163,184,.13);color:#64748b;border:1px solid rgba(148,163,184,.22)}
.kpi-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin:14px 0}.kpi-card{background:rgba(255,255,255,.86);border:1px solid rgba(148,163,184,.24);border-radius:20px;padding:14px 15px;box-shadow:0 14px 34px rgba(15,23,42,.06)}.kpi-label{font-size:12px;color:#64748b;font-weight:800}.kpi-value{font-size:21px;font-weight:950;letter-spacing:-.04em;color:#0f172a;margin-top:4px}.kpi-sub{font-size:11px;color:#94a3b8;margin-top:4px}
.progress-row{margin:10px 0 12px}.progress-head{display:flex;justify-content:space-between;font-size:12px;font-weight:800;color:#cbd5e1;margin-bottom:5px}.progress-track{height:8px;border-radius:999px;background:rgba(148,163,184,.20);overflow:hidden}.progress-bar{height:8px;border-radius:999px;background:linear-gradient(90deg,#3182f6,#12b886)}
.plan-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin-top:10px}.plan-item{background:rgba(248,250,252,.78);border:1px solid rgba(148,163,184,.20);border-radius:14px;padding:10px}.plan-label{font-size:11px;color:#64748b;font-weight:800}.plan-value{font-size:15px;color:#0f172a;font-weight:900;margin-top:2px}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#07111f 0%,#0b1120 45%,#0f172a 100%)}[data-testid="stSidebar"] .stSlider [data-testid="stTickBar"]{opacity:.35}
@media(max-width:1100px){.kpi-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.plan-grid{grid-template-columns:1fr}.hero-title{font-size:30px}}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* 사이드바 가독성 보정: 계정명 코드 박스 제거 + 숫자 입력칸 글자색 수정 */
.sidebar-user-line{
    display:flex;
    align-items:center;
    gap:8px;
    margin:4px 0 12px 0;
    font-size:14px;
    color:rgba(255,255,255,.82) !important;
}
.sidebar-user-line span{
    color:rgba(255,255,255,.78) !important;
    font-weight:700;
}
.sidebar-user-line strong{
    color:#ffffff !important;
    font-weight:900;
    background:transparent !important;
    border:none !important;
    box-shadow:none !important;
    padding:0 !important;
}
[data-testid="stSidebar"] code{
    background:transparent !important;
    color:#ffffff !important;
    border:none !important;
    padding:0 !important;
    box-shadow:none !important;
}
/* number_input / text_input 값이 흰 배경에 흰색으로 보이지 않도록 고정 */
[data-testid="stSidebar"] [data-testid="stNumberInput"] label,
[data-testid="stSidebar"] [data-testid="stTextInput"] label,
[data-testid="stSidebar"] [data-testid="stSelectbox"] label,
[data-testid="stSidebar"] [data-testid="stSlider"] label{
    color:rgba(255,255,255,.88) !important;
    font-weight:800 !important;
}
[data-testid="stSidebar"] [data-testid="stNumberInput"] input,
[data-testid="stSidebar"] [data-testid="stTextInput"] input{
    background:#ffffff !important;
    color:#0f172a !important;
    -webkit-text-fill-color:#0f172a !important;
    caret-color:#0f172a !important;
    border-radius:12px !important;
}
[data-testid="stSidebar"] [data-testid="stNumberInput"] input::placeholder,
[data-testid="stSidebar"] [data-testid="stTextInput"] input::placeholder{
    color:#64748b !important;
    -webkit-text-fill-color:#64748b !important;
}
/* number_input의 +/- 버튼 아이콘도 보이게 */
[data-testid="stSidebar"] [data-testid="stNumberInput"] button,
[data-testid="stSidebar"] [data-testid="stNumberInput"] button *{
    color:#0f172a !important;
}
/* selectbox 선택값은 어두운 배경 위에서 보이도록 */
[data-testid="stSidebar"] [data-baseweb="select"] *{
    color:#ffffff !important;
}
</style>
""", unsafe_allow_html=True)


C = {
    "bg": "#f8fbff", "surface": "#ffffff", "border": "#e2e8f0", "text": "#0f172a", "subtext": "#64748b",
    "buy": "#10b981", "sell": "#ef4444", "warn": "#f59e0b", "blue": "#2563eb",
    "ma5": "#f59e0b", "ma20": "#06b6d4", "ma60": "#f97316", "ma120": "#7c3aed",
}

DB_FILE = "users_db.json"

PREDEFINED_STOCK_SEARCH = {
    "삼성전자": "005930.KS", "삼성": "005930.KS",
    "sk하이닉스": "000660.KS", "하이닉스": "000660.KS", "sk": "000660.KS",
    "naver": "035420.KS", "네이버": "035420.KS",
    "카카오": "035720.KS", "kakao": "035720.KS",
    "현대차": "005380.KS", "현대자동차": "005380.KS",
    "셀트리온": "068270.KS", "nh투자증권": "005940.KS", "nh": "005940.KS",
    "애플": "AAPL", "apple": "AAPL",
    "엔비디아": "NVDA", "nvidia": "NVDA",
    "테슬라": "TSLA", "tesla": "TSLA",
    "마이크로소프트": "MSFT", "msft": "MSFT", "마소": "MSFT",
    "아마존": "AMZN", "amazon": "AMZN",
    "구글": "GOOGL", "google": "GOOGL",
    "메타": "META", "meta": "META", "페이스북": "META",
}

PREDEFINED_COIN_SEARCH = {
    "비트코인": "BTC-USD", "비트": "BTC-USD", "btc": "BTC-USD",
    "이더리움": "ETH-USD", "이더": "ETH-USD", "eth": "ETH-USD",
    "리플": "XRP-USD", "xrp": "XRP-USD",
    "도지코인": "DOGE-USD", "도지": "DOGE-USD", "doge": "DOGE-USD",
    "솔라나": "SOL-USD", "sol": "SOL-USD",
    "바이낸스코인": "BNB-USD", "bnb": "BNB-USD",
    "에이다": "ADA-USD", "ada": "ADA-USD",
    "수이": "SUI-USD", "sui": "SUI-USD",
    "아발란체": "AVAX-USD", "avax": "AVAX-USD",
}

TICKER_NAME_MAP = {
    "005930.KS": "삼성전자", "000660.KS": "SK하이닉스", "035420.KS": "NAVER",
    "035720.KS": "카카오", "005380.KS": "현대차", "068270.KS": "셀트리온",
    "005940.KS": "NH투자증권", "AAPL": "애플", "NVDA": "엔비디아",
    "TSLA": "테슬라", "MSFT": "마이크로소프트", "AMZN": "아마존",
    "GOOGL": "구글", "META": "메타", "BTC-USD": "비트코인",
    "ETH-USD": "이더리움", "XRP-USD": "리플", "DOGE-USD": "도지코인",
    "SOL-USD": "솔라나", "BNB-USD": "바이낸스코인", "ADA-USD": "에이다",
    "SUI-USD": "수이", "AVAX-USD": "아발란체",
}

STOCK_SCAN_LIST = ["AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "GOOGL", "META", "005930.KS", "000660.KS", "035420.KS", "035720.KS", "005380.KS"]
COIN_SCAN_LIST = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "BNB-USD", "ADA-USD", "AVAX-USD"]

TIMEFRAME_MAP = {
    "일봉 (1d)": ("1d", "5y"),
    "1시간봉 (1h)": ("1h", "2y"),
    "30분봉 (30m)": ("30m", "60d"),
    "15분봉 (15m)": ("15m", "60d"),
}

# ==========================================================
# 로컬 세션 초기화
# ==========================================================

LOCAL_STATE_FILE = "local_portfolio.json"

def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def init_user_session(user_data=None):
    user_data = user_data or {}
    st.session_state.stock_balance = user_data.get("stock_balance", 10000.0)
    st.session_state.coin_balance = user_data.get("coin_balance", 10000.0)
    st.session_state.stock_shares = user_data.get("stock_shares", {})
    st.session_state.coin_shares = user_data.get("coin_shares", {})
    st.session_state.stock_trade_log = user_data.get("stock_trade_log", [])
    st.session_state.coin_trade_log = user_data.get("coin_trade_log", [])


def save_current_user_data():
    data = {
        "stock_balance": st.session_state.get("stock_balance", 10000.0),
        "coin_balance": st.session_state.get("coin_balance", 10000.0),
        "stock_shares": st.session_state.get("stock_shares", {}),
        "coin_shares": st.session_state.get("coin_shares", {}),
        "stock_trade_log": st.session_state.get("stock_trade_log", []),
        "coin_trade_log": st.session_state.get("coin_trade_log", []),
    }
    save_json(LOCAL_STATE_FILE, data)


def reset_local_portfolio():
    init_user_session({})
    if os.path.exists(LOCAL_STATE_FILE):
        try:
            os.remove(LOCAL_STATE_FILE)
        except Exception:
            pass


if "session_ready" not in st.session_state:
    init_user_session(load_json(LOCAL_STATE_FILE, {}))
    st.session_state.session_ready = True

# ==========================================================
# 유틸 / 데이터 수집
# ==========================================================

def is_korean_ticker(symbol):
    return symbol.endswith(".KS") or symbol.endswith(".KQ")


def search_ticker_by_name(query, is_coin=False):
    query = query.strip()
    if not query:
        return None
    clean = query.lower().replace(" ", "")
    if is_coin:
        if clean in PREDEFINED_COIN_SEARCH:
            return PREDEFINED_COIN_SEARCH[clean]
        if "-" in query:
            return query.upper()
        return f"{query.upper()}-USD"
    if query.upper().endswith((".KS", ".KQ")):
        return query.upper()
    if clean in PREDEFINED_STOCK_SEARCH:
        return PREDEFINED_STOCK_SEARCH[clean]
    url = "https://query1.finance.yahoo.com/v1/finance/search"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, params={"q": query, "quotesCount": 5, "newsCount": 0}, headers=headers, timeout=10)
        if r.status_code == 200:
            js = r.json()
            quotes = js.get("quotes", [])
            if quotes:
                return quotes[0].get("symbol", query).upper()
    except Exception:
        pass
    return query.upper()


@st.cache_data(ttl=300)
def get_realtime_exchange_rate():
    url = "https://query1.finance.yahoo.com/v8/finance/chart/USDKRW=X?range=1d&interval=1m"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            return float(r.json()["chart"]["result"][0]["meta"]["regularMarketPrice"])
    except Exception:
        pass
    return 1380.0


EXCHANGE_RATE = get_realtime_exchange_rate()


def fmt_curr(v):
    if v is None or pd.isna(v):
        return "N/A"
    sign = "-" if float(v) < 0 else ""
    v = abs(float(v))
    if st.session_state.get("currency") == "KRW":
        return f"{sign}₩{v * EXCHANGE_RATE:,.0f}"
    return f"{sign}${v:,.2f}"


@st.cache_data(ttl=300)
def fetch_ohlcv(symbol, interval, data_range):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={data_range}&interval={interval}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=12)
        if r.status_code != 200:
            return None, symbol
        raw = r.json()
        result = raw.get("chart", {}).get("result")
        if not result:
            return None, symbol
        result = result[0]
        if "timestamp" not in result:
            return None, symbol
        q = result["indicators"]["quote"][0]
    except Exception:
        return None, symbol

    date_format = "%Y-%m-%d %H:%M" if interval != "1d" else "%Y-%m-%d"
    df = pd.DataFrame({
        "date": [datetime.fromtimestamp(ts).strftime(date_format) for ts in result["timestamp"]],
        "open": q.get("open"), "high": q.get("high"), "low": q.get("low"),
        "close": q.get("close"), "volume": q.get("volume"),
    }).dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    if df.empty:
        return None, symbol
    df["volume"] = df["volume"].fillna(0)
    df = df[df["close"] > 0].reset_index(drop=True)

    meta = result.get("meta", {})
    api_currency = str(meta.get("currency", "USD")).upper()
    api_name = TICKER_NAME_MAP.get(symbol) or meta.get("shortName") or symbol
    if is_korean_ticker(symbol) or api_currency == "KRW":
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col] / EXCHANGE_RATE
    return df, api_name


def calculate_indicators(df):
    df = df.copy()
    ret = df["close"].pct_change()
    for w in [5, 10, 14, 20, 60, 120, 200]:
        df[f"ma{w}"] = df["close"].rolling(w).mean()
        df[f"ret{w}"] = df["close"].pct_change(w)
        df[f"volatility{w}"] = ret.rolling(w).std()
    df["volMa20"] = df["volume"].rolling(20).mean()
    df["volRatio"] = df["volume"] / df["volMa20"].replace(0, np.nan)

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr14"] / df["close"].replace(0, np.nan)

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df.loc[loss == 0, "rsi"] = 100

    df["candle_body"] = (df["close"] - df["open"]) / df["open"].replace(0, np.nan)
    df["range_pct"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["dollar_volume"] = df["close"] * df["volume"]
    return df


@st.cache_data(ttl=300)
def get_market_regime(symbol, interval, data_range):
    if symbol.endswith("-USD"):
        proxy = "BTC-USD"
        proxy_name = "비트코인 시장"
    elif is_korean_ticker(symbol):
        proxy = "^KS11"
        proxy_name = "코스피"
    else:
        proxy = "^GSPC"
        proxy_name = "S&P 500"
    df, _ = fetch_ohlcv(proxy, interval, data_range)
    if df is None or len(df) < 130:
        return {"proxy": proxy, "name": proxy_name, "ok": True, "score": 0.5, "text": "시장필터 데이터 부족"}
    df = calculate_indicators(df)
    i = len(df) - 1
    close = df.loc[i, "close"]
    ma20 = df.loc[i, "ma20"]
    ma120 = df.loc[i, "ma120"]
    ma20_prev = df.loc[max(0, i - 20), "ma20"]
    score = 0.0
    if pd.notna(ma120) and close > ma120:
        score += 0.45
    if pd.notna(ma20) and pd.notna(ma20_prev) and ma20 > ma20_prev:
        score += 0.35
    if pd.notna(ma20) and close > ma20:
        score += 0.20
    ok = score >= 0.55
    return {
        "proxy": proxy, "name": proxy_name, "ok": ok, "score": float(score),
        "text": f"{proxy_name} 시장필터 {'통과' if ok else '주의'} ({score*100:.0f}점)",
    }

# ==========================================================
# 시계열 특징 / 모델
# ==========================================================

@dataclass
class CostConfig:
    fee_pct: float
    slippage_pct: float
    sell_tax_pct: float

@dataclass
class RiskConfig:
    start_capital: float
    risk_per_trade: float
    max_position_pct: float
    fixed_stop_pct: float
    take_profit_pct: float
    atr_stop_mult: float
    trailing_stop_pct: float
    max_hold_bars: int
    max_risk_score: float
    min_dollar_volume: float


def sigmoid(x):
    x = np.clip(x, -40, 40)
    return 1 / (1 + np.exp(-x))


def nan_to_zero(x):
    if x is None or pd.isna(x) or np.isinf(x):
        return 0.0
    return float(x)


def risk_score_at(df, idx):
    risk = 0.0
    rsi = df.loc[idx, "rsi"] if "rsi" in df else np.nan
    vol20 = df.loc[idx, "volatility20"] if "volatility20" in df else np.nan
    vol60 = df.loc[idx, "volatility60"] if "volatility60" in df else np.nan
    close = df.loc[idx, "close"]
    ma20 = df.loc[idx, "ma20"] if "ma20" in df else np.nan
    ma60 = df.loc[idx, "ma60"] if "ma60" in df else np.nan
    ma120 = df.loc[idx, "ma120"] if "ma120" in df else np.nan
    atr_pct = df.loc[idx, "atr_pct"] if "atr_pct" in df else np.nan

    if pd.notna(rsi):
        if rsi > 75:
            risk += min((rsi - 75) / 25, 1) * 0.25
        elif rsi < 25:
            risk += min((25 - rsi) / 25, 1) * 0.10
    if pd.notna(vol20) and pd.notna(vol60) and vol60 > 0:
        risk += min(max((vol20 / vol60) - 1, 0), 2) / 2 * 0.25
    if pd.notna(atr_pct):
        risk += min(max((atr_pct - 0.03) / 0.10, 0), 1) * 0.20
    if all(pd.notna(x) for x in [close, ma20, ma60, ma120]):
        if close < ma120:
            risk += 0.15
        if ma20 < ma60 < ma120:
            risk += 0.15
    return float(np.clip(risk, 0, 1))


def compute_feature_at(df, idx, lookback, market_score=0.5):
    if idx < max(lookback, 200):
        return None
    window = df.iloc[idx - lookback + 1: idx + 1]
    close = float(df.loc[idx, "close"])
    returns = window["close"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    feats = []

    # 최근 수익률과 모멘텀
    for w in [1, 2, 3, 5, 10, 20, 60, 120]:
        if idx - w >= 0 and df.loc[idx - w, "close"] != 0:
            feats.append(close / df.loc[idx - w, "close"] - 1)
        else:
            feats.append(0.0)

    # 변동성, 왜도 비슷한 방향성 정보
    for w in [5, 10, 20, min(lookback, 60), min(lookback, 120)]:
        s = returns.tail(w)
        feats.append(nan_to_zero(s.std()))
        feats.append(nan_to_zero(s.mean()))

    # 이동평균선 위치와 기울기
    for w in [5, 10, 20, 60, 120, 200]:
        ma = df.loc[idx, f"ma{w}"]
        feats.append(close / ma - 1 if pd.notna(ma) and ma != 0 else 0.0)
        past = df.loc[max(0, idx - min(w, 20)), f"ma{w}"]
        feats.append(ma / past - 1 if pd.notna(ma) and pd.notna(past) and past != 0 else 0.0)

    # RSI, 거래량, ATR
    feats.append((nan_to_zero(df.loc[idx, "rsi"]) - 50) / 50)
    feats.append(nan_to_zero(df.loc[idx, "volRatio"]) - 1)
    vol_short = window["volume"].tail(5).mean()
    vol_long = window["volume"].tail(min(lookback, 60)).mean()
    feats.append(vol_short / vol_long - 1 if vol_long > 0 else 0.0)
    feats.append(nan_to_zero(df.loc[idx, "atr_pct"]))
    feats.append(nan_to_zero(df.loc[idx, "candle_body"]))
    feats.append(nan_to_zero(df.loc[idx, "range_pct"]))

    # 박스권 위치
    box_h = window["high"].max()
    box_l = window["low"].min()
    if box_h > box_l:
        feats.append((close - box_l) / (box_h - box_l))
        feats.append((box_h - box_l) / close)
    else:
        feats.extend([0.5, 0.0])

    # 리스크/시장 필터 정보
    feats.append(risk_score_at(df, idx))
    feats.append(float(market_score))

    arr = np.array(feats, dtype=float)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def build_dataset(df, lookback, horizon, market_score=0.5):
    X, idxs, y = [], [], []
    start = max(lookback, 200)
    for idx in range(start, len(df)):
        feat = compute_feature_at(df, idx, lookback, market_score)
        if feat is None:
            continue
        idxs.append(idx)
        X.append(feat)
        if idx + horizon < len(df):
            y.append(df.loc[idx + horizon, "close"] / df.loc[idx, "close"] - 1)
        else:
            y.append(np.nan)
    if not X:
        return None, None, None
    return np.vstack(X), np.array(idxs), np.array(y, dtype=float)


def fit_predict_ensemble(X_train, y_train, x_one, model_mode="앙상블"):
    mask = np.isfinite(y_train)
    X_train = X_train[mask]
    y_train = y_train[mask]
    if len(y_train) < 60:
        return 0.0, 0.5, 0.02

    scale = float(np.nanstd(y_train))
    if not np.isfinite(scale) or scale < 1e-5:
        scale = 0.02

    if SKLEARN_AVAILABLE:
        preds = []
        probs = []
        if model_mode in ["앙상블", "Ridge"]:
            ridge = make_pipeline(StandardScaler(), Ridge(alpha=2.0))
            ridge.fit(X_train, y_train)
            p = float(ridge.predict(x_one.reshape(1, -1))[0])
            preds.append(p)
            probs.append(float(sigmoid(p / scale)))
        if model_mode in ["앙상블", "GradientBoosting"]:
            gb = GradientBoostingRegressor(
                n_estimators=80, learning_rate=0.04, max_depth=2,
                subsample=0.8, random_state=42
            )
            gb.fit(X_train, y_train)
            p = float(gb.predict(x_one.reshape(1, -1))[0])
            preds.append(p)
            probs.append(float(sigmoid(p / scale)))
        if model_mode in ["앙상블", "RandomForest"]:
            y_cls = (y_train > 0).astype(int)
            if len(np.unique(y_cls)) >= 2:
                rf = RandomForestClassifier(
                    n_estimators=120, max_depth=4, min_samples_leaf=8,
                    random_state=42, class_weight="balanced_subsample"
                )
                rf.fit(X_train, y_cls)
                probs.append(float(rf.predict_proba(x_one.reshape(1, -1))[0][1]))
        if model_mode in ["앙상블", "Logistic"]:
            y_cls = (y_train > 0).astype(int)
            if len(np.unique(y_cls)) >= 2:
                logi = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced"))
                logi.fit(X_train, y_cls)
                probs.append(float(logi.predict_proba(x_one.reshape(1, -1))[0][1]))
        pred = float(np.mean(preds)) if preds else 0.0
        prob = float(np.mean(probs)) if probs else float(sigmoid(pred / scale))
        return pred, prob, scale

    # fallback: 직접 구현한 Ridge
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    Xs = (X_train - mean) / std
    xo = (x_one - mean) / std
    X_aug = np.column_stack([np.ones(len(Xs)), Xs])
    xo_aug = np.concatenate([[1.0], xo])
    alpha = 2.0
    reg = np.eye(X_aug.shape[1]) * alpha
    reg[0, 0] = 0.0
    try:
        beta = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ y_train)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(X_aug.T @ X_aug + reg) @ X_aug.T @ y_train
    pred = float(xo_aug @ beta)
    return pred, float(sigmoid(pred / scale)), scale



def clamp01(v):
    try:
        if pd.isna(v) or np.isinf(v):
            return 0.5
        return float(np.clip(v, 0, 1))
    except Exception:
        return 0.5


def component_scores_at(df, idx, market_score=0.5):
    """현재 봉의 기술적 상태를 0~1 점수로 변환한다."""
    close = float(df.loc[idx, "close"])
    ma20 = df.loc[idx, "ma20"] if "ma20" in df else np.nan
    ma60 = df.loc[idx, "ma60"] if "ma60" in df else np.nan
    ma120 = df.loc[idx, "ma120"] if "ma120" in df else np.nan
    ma20_prev = df.loc[max(0, idx - 20), "ma20"] if "ma20" in df else np.nan
    ma60_prev = df.loc[max(0, idx - 20), "ma60"] if "ma60" in df else np.nan
    rsi = df.loc[idx, "rsi"] if "rsi" in df else np.nan
    vol_ratio = df.loc[idx, "volRatio"] if "volRatio" in df else np.nan
    atr_pct = df.loc[idx, "atr_pct"] if "atr_pct" in df else np.nan
    ret5 = df.loc[idx, "ret5"] if "ret5" in df else np.nan
    ret20 = df.loc[idx, "ret20"] if "ret20" in df else np.nan

    trend = 0.5
    if pd.notna(ma20) and pd.notna(ma60) and pd.notna(ma120):
        trend = 0.20
        trend += 0.25 if close > ma20 else 0.0
        trend += 0.25 if ma20 > ma60 else 0.0
        trend += 0.20 if ma60 > ma120 else 0.0
        if pd.notna(ma20_prev) and ma20_prev != 0:
            trend += 0.10 if ma20 > ma20_prev else -0.05
        if pd.notna(ma60_prev) and ma60_prev != 0:
            trend += 0.10 if ma60 > ma60_prev else -0.05
    trend = clamp01(trend)

    momentum = 0.5
    if pd.notna(ret5):
        momentum += np.tanh(ret5 * 10) * 0.20
    if pd.notna(ret20):
        momentum += np.tanh(ret20 * 5) * 0.25
    momentum = clamp01(momentum)

    # RSI는 45~60 회복 구간을 높게 보고, 과열/과매도는 낮춘다.
    rsi_score = 0.5
    if pd.notna(rsi):
        if 45 <= rsi <= 62:
            rsi_score = 0.85
        elif 35 <= rsi < 45:
            rsi_score = 0.65
        elif 62 < rsi <= 72:
            rsi_score = 0.55
        elif rsi > 72:
            rsi_score = 0.25
        else:
            rsi_score = 0.35

    volume = 0.5
    if pd.notna(vol_ratio):
        # 너무 적은 거래량은 감점, 평균 대비 1.2~2.5배는 가점, 지나친 폭증은 약간 감점
        if vol_ratio < 0.7:
            volume = 0.25
        elif vol_ratio <= 2.5:
            volume = clamp01(0.45 + (vol_ratio - 0.7) / 1.8 * 0.45)
        else:
            volume = 0.70

    volatility = 0.5
    if pd.notna(atr_pct):
        # ATR이 낮을수록 안정적. 너무 낮으면 움직임 부족이므로 0.8 정도까지.
        volatility = 1.0 - min(max((atr_pct - 0.015) / 0.10, 0), 1) * 0.75
        volatility = clamp01(volatility)

    pullback = 0.5
    if pd.notna(ma20) and close > 0 and ma20 > 0:
        dist = close / ma20 - 1
        # 20일선 부근 눌림목을 선호. 너무 멀리 위/아래면 감점.
        pullback = 1.0 - min(abs(dist) / 0.12, 1) * 0.70
        if -0.05 <= dist <= 0.04:
            pullback += 0.15
        pullback = clamp01(pullback)

    return {
        "trend": float(trend),
        "momentum": float(momentum),
        "rsi": float(rsi_score),
        "volume": float(volume),
        "volatility": float(volatility),
        "pullback": float(pullback),
        "market": clamp01(market_score),
        "risk_inverse": 1.0 - risk_score_at(df, idx),
    }


def weighted_technical_score_at(df, idx, weights, market_score=0.5):
    comps = component_scores_at(df, idx, market_score)
    total_w = sum(max(0.0, float(v)) for v in weights.values())
    if total_w <= 0:
        total_w = 1.0
    score = 0.0
    for k, w in weights.items():
        score += comps.get(k, 0.5) * max(0.0, float(w))
    return float(np.clip(score / total_w, 0, 1)), comps


def signal_grade(prob, expected_net, risk, market_ok, final_score=0.5):
    if not market_ok:
        return "금지", "시장 추세 불리"
    if risk >= 0.78:
        return "금지", "리스크 과다"
    if final_score >= 0.78 and prob >= 0.66 and expected_net >= 0.025 and risk <= 0.40:
        return "S", "강한 매수 후보"
    if final_score >= 0.68 and prob >= 0.61 and expected_net >= 0.015 and risk <= 0.55:
        return "A", "매수 후보"
    if final_score >= 0.58 and prob >= 0.56 and expected_net >= 0.005 and risk <= 0.68:
        return "B", "관심 후보"
    return "관망", "조건 부족"


def walk_forward_predictions(
    df, lookback, horizon, train_window, model_mode, risk_strength, cost_cfg,
    market_score=0.5, step=1, weights=None, ml_weight=0.65
):
    X, idxs, y = build_dataset(df, lookback, horizon, market_score)
    if X is None:
        return pd.DataFrame()
    weights = weights or {"trend": 1, "momentum": 1, "rsi": 1, "volume": 1, "volatility": 1, "pullback": 1, "market": 1, "risk_inverse": 1}
    rows = []
    round_trip_cost = cost_cfg.fee_pct * 2 + cost_cfg.slippage_pct * 2 + cost_cfg.sell_tax_pct
    for row_i in range(0, len(idxs), step):
        idx = int(idxs[row_i])
        train_end = row_i
        train_start = max(0, train_end - train_window)
        X_train = X[train_start:train_end]
        y_train = y[train_start:train_end]
        pred_raw, prob_raw, scale = fit_predict_ensemble(X_train, y_train, X[row_i], model_mode)
        risk = risk_score_at(df, idx)
        pred_adj = pred_raw - risk_strength * risk * scale
        prob_adj = float(np.clip((prob_raw * 0.65) + (sigmoid(pred_adj / scale) * 0.35), 0, 1))
        expected_net = pred_adj - round_trip_cost
        tech_score, comps = weighted_technical_score_at(df, idx, weights, market_score)
        ml_score = float(np.clip((prob_adj * 0.60) + (sigmoid(expected_net / max(scale, 1e-6)) * 0.40), 0, 1))
        final_score = float(np.clip(ml_score * ml_weight + tech_score * (1 - ml_weight), 0, 1))
        rows.append({
            "idx": idx,
            "date": df.loc[idx, "date"],
            "pred_return_raw": pred_raw,
            "pred_return": pred_adj,
            "expected_net": expected_net,
            "prob_up": prob_adj,
            "risk_score": risk,
            "tech_score": tech_score,
            "ml_score": ml_score,
            "final_score": final_score,
            "scale": scale,
            "actual_future_return": y[row_i] if np.isfinite(y[row_i]) else np.nan,
            **{f"comp_{k}": v for k, v in comps.items()},
        })
    return pd.DataFrame(rows)


def latest_prediction(
    df, lookback, horizon, train_window, model_mode, risk_strength, cost_cfg,
    market_score=0.5, market_ok=True, weights=None, ml_weight=0.65
):
    X, idxs, y = build_dataset(df, lookback, horizon, market_score)
    if X is None or len(X) < 70:
        return None
    weights = weights or {"trend": 1, "momentum": 1, "rsi": 1, "volume": 1, "volatility": 1, "pullback": 1, "market": 1, "risk_inverse": 1}
    latest_idx = len(df) - 1
    x = compute_feature_at(df, latest_idx, lookback, market_score)
    if x is None:
        return None
    valid = np.isfinite(y)
    X_train = X[valid][-train_window:]
    y_train = y[valid][-train_window:]
    pred_raw, prob_raw, scale = fit_predict_ensemble(X_train, y_train, x, model_mode)
    risk = risk_score_at(df, latest_idx)
    round_trip_cost = cost_cfg.fee_pct * 2 + cost_cfg.slippage_pct * 2 + cost_cfg.sell_tax_pct
    pred_adj = pred_raw - risk_strength * risk * scale
    prob_adj = float(np.clip((prob_raw * 0.65) + (sigmoid(pred_adj / scale) * 0.35), 0, 1))
    expected_net = pred_adj - round_trip_cost
    tech_score, comps = weighted_technical_score_at(df, latest_idx, weights, market_score)
    ml_score = float(np.clip((prob_adj * 0.60) + (sigmoid(expected_net / max(scale, 1e-6)) * 0.40), 0, 1))
    final_score = float(np.clip(ml_score * ml_weight + tech_score * (1 - ml_weight), 0, 1))
    grade, reason = signal_grade(prob_adj, expected_net, risk, market_ok, final_score)
    return {
        "idx": latest_idx,
        "date": df.loc[latest_idx, "date"],
        "pred_return_raw": pred_raw,
        "pred_return": pred_adj,
        "expected_net": expected_net,
        "prob_up": prob_adj,
        "risk_score": risk,
        "tech_score": tech_score,
        "ml_score": ml_score,
        "final_score": final_score,
        "scale": scale,
        "grade": grade,
        "reason": reason,
        "components": comps,
    }

# ==========================================================
# 백테스트 / 성능 지표
# ==========================================================

def calc_position_size(capital, entry_price, stop_price, risk_cfg):
    if entry_price <= 0 or stop_price <= 0 or entry_price <= stop_price:
        return 0.0, 0.0
    risk_amount = capital * risk_cfg.risk_per_trade
    per_share_risk = entry_price - stop_price
    qty_by_risk = risk_amount / per_share_risk
    qty_by_cap = (capital * risk_cfg.max_position_pct) / entry_price
    qty = max(0.0, min(qty_by_risk, qty_by_cap))
    used_capital = qty * entry_price
    return qty, used_capital


def run_backtest(df, pred_df, min_prob, min_expected_net, min_final_score, cost_cfg, risk_cfg, market_ok=True, use_market_filter=True):
    if pred_df.empty:
        return [], pd.DataFrame(columns=["date", "equity"]), [], [], None

    signal_map = {int(r["idx"]): r for _, r in pred_df.iterrows()}
    capital = float(risk_cfg.start_capital)
    cash = capital
    position = None
    trades, buys, sells, equity_rows = [], [], [], []

    for i in range(len(df)):
        close = float(df.loc[i, "close"])
        high = float(df.loc[i, "high"])
        low = float(df.loc[i, "low"])
        date = df.loc[i, "date"]

        if position is not None:
            position["max_high"] = max(position["max_high"], high)
            trail_stop = position["max_high"] * (1 - risk_cfg.trailing_stop_pct) if risk_cfg.trailing_stop_pct > 0 else 0
            active_stop = max(position["stop_price"], trail_stop)
            target = position["target_price"]
            reason = None
            exit_ref_price = close

            if low <= active_stop:
                reason = "손절/트레일링"
                exit_ref_price = active_stop
            elif high >= target:
                reason = "익절"
                exit_ref_price = target
            elif (i - position["entry_idx"]) >= risk_cfg.max_hold_bars:
                reason = "시간청산"
                exit_ref_price = close

            if reason:
                exit_price = exit_ref_price * (1 - cost_cfg.slippage_pct)
                gross_value = position["qty"] * exit_price
                fee = gross_value * cost_cfg.fee_pct
                tax = gross_value * cost_cfg.sell_tax_pct
                net_value = gross_value - fee - tax
                cash += net_value
                ret = (net_value - position["entry_total_cost"]) / position["entry_total_cost"] * 100 if position["entry_total_cost"] else 0
                trades.append({
                    "entryDate": position["entry_date"], "entryPrice": position["entry_price"],
                    "exitDate": date, "exitPrice": exit_price, "qty": position["qty"],
                    "ret": ret, "win": ret > 0, "exitReason": reason,
                    "prob_up": position["prob_up"], "pred_return": position["pred_return"] * 100,
                    "expected_net": position["expected_net"] * 100, "risk_score": position["risk_score"], "final_score": position.get("final_score", 0.0),
                    "capital_after": cash,
                })
                sells.append({"idx": i, "price": exit_price, "date": date, "reason": reason})
                position = None

        if position is None:
            row = signal_map.get(i)
            if row is not None:
                market_pass = (market_ok or not use_market_filter)
                risk_pass = float(row["risk_score"]) <= risk_cfg.max_risk_score
                liquidity_pass = float(df.loc[i, "dollar_volume"]) >= risk_cfg.min_dollar_volume
                signal_pass = (
                    market_pass and risk_pass and liquidity_pass and
                    float(row["prob_up"]) >= min_prob and float(row["expected_net"]) >= min_expected_net and float(row.get("final_score", 0.0)) >= min_final_score
                )
                if signal_pass and cash > 0:
                    atr_pct = nan_to_zero(df.loc[i, "atr_pct"])
                    stop_pct = max(risk_cfg.fixed_stop_pct, risk_cfg.atr_stop_mult * atr_pct)
                    stop_pct = min(max(stop_pct, 0.005), 0.35)
                    entry_price = close * (1 + cost_cfg.slippage_pct)
                    stop_price = entry_price * (1 - stop_pct)
                    target_price = entry_price * (1 + risk_cfg.take_profit_pct)
                    qty, used_cap = calc_position_size(cash, entry_price, stop_price, risk_cfg)
                    if qty > 0 and used_cap > 0:
                        entry_fee = used_cap * cost_cfg.fee_pct
                        total_cost = used_cap + entry_fee
                        if total_cost <= cash:
                            cash -= total_cost
                            position = {
                                "entry_idx": i, "entry_date": date, "entry_price": entry_price,
                                "qty": qty, "entry_total_cost": total_cost,
                                "stop_price": stop_price, "target_price": target_price,
                                "max_high": high, "prob_up": float(row["prob_up"]),
                                "pred_return": float(row["pred_return"]), "expected_net": float(row["expected_net"]),
                                "risk_score": float(row["risk_score"]),
                                "final_score": float(row.get("final_score", 0.0)),
                            }
                            buys.append({"idx": i, "price": entry_price, "date": date})

        # 일별 평가금액 기록
        if position is not None:
            mark_value = position["qty"] * close * (1 - cost_cfg.slippage_pct) * (1 - cost_cfg.fee_pct - cost_cfg.sell_tax_pct)
            equity = cash + mark_value
        else:
            equity = cash
        equity_rows.append({"idx": i, "date": date, "equity": equity})

    active = position
    eq_df = pd.DataFrame(equity_rows)
    return trades, eq_df, buys, sells, active


def max_drawdown(equity):
    if len(equity) == 0:
        return 0.0
    arr = np.array(equity, dtype=float)
    peak = np.maximum.accumulate(arr)
    dd = arr / np.where(peak == 0, np.nan, peak) - 1
    return float(np.nanmin(dd) * 100)


def calc_metrics(trades, equity_df, start_capital):
    if equity_df.empty:
        final_cap = start_capital
    else:
        final_cap = float(equity_df["equity"].iloc[-1])
    total_return = (final_cap / start_capital - 1) * 100 if start_capital else 0
    n = len(trades)
    wins = [t["ret"] for t in trades if t["ret"] > 0]
    losses = [t["ret"] for t in trades if t["ret"] <= 0]
    win_rate = len(wins) / n * 100 if n else 0
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    profit_factor = (sum(wins) / abs(sum(losses))) if losses and abs(sum(losses)) > 0 else (999.0 if wins else 0.0)
    payoff = (avg_win / abs(avg_loss)) if avg_loss < 0 else 0.0
    expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)
    mdd = max_drawdown(equity_df["equity"].values if not equity_df.empty else [])
    sharpe = 0.0
    if not equity_df.empty and len(equity_df) > 5:
        rets = equity_df["equity"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        if len(rets) > 2 and rets.std() > 0:
            sharpe = float((rets.mean() / rets.std()) * math.sqrt(252))
    return {
        "거래횟수": n,
        "총수익률": total_return,
        "승률": win_rate,
        "평균수익": avg_win,
        "평균손실": avg_loss,
        "손익비": payoff,
        "Profit Factor": profit_factor,
        "기대값": expectancy,
        "MDD": mdd,
        "Sharpe": sharpe,
        "최종자산": final_cap,
    }


def prediction_accuracy(pred_df):
    if pred_df.empty or "actual_future_return" not in pred_df:
        return 0.0, 0
    d = pred_df.dropna(subset=["actual_future_return"])
    if d.empty:
        return 0.0, 0
    pred_dir = d["pred_return"] > 0
    actual_dir = d["actual_future_return"] > 0
    return float((pred_dir == actual_dir).mean() * 100), len(d)

# ==========================================================
# 차트
# ==========================================================

def format_unit(v):
    if v is None or pd.isna(v):
        return "0"
    v = abs(float(v))
    if v >= 100000000:
        return f"{v/100000000:.1f}억"
    if v >= 10000:
        return f"{v/10000:.1f}만"
    return f"{v:,.0f}"


def make_price_chart(df, pred_df=None, buys=None, sells=None, min_prob=0.6, min_expected_net=0.01):
    buys = buys or []
    sells = sells or []
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_width=[0.22, 0.78])
    mult = EXCHANGE_RATE if st.session_state.get("currency") == "KRW" else 1.0
    x = df.index
    fig.add_trace(go.Candlestick(
        x=x, open=df["open"]*mult, high=df["high"]*mult, low=df["low"]*mult, close=df["close"]*mult,
        name="시세", text=df["date"], increasing_line_color=C["buy"], increasing_fillcolor=C["buy"],
        decreasing_line_color=C["sell"], decreasing_fillcolor=C["sell"]), row=1, col=1)
    for ma, col in [("ma5", "ma5"), ("ma20", "ma20"), ("ma60", "ma60"), ("ma120", "ma120")]:
        fig.add_trace(go.Scatter(x=x, y=df[ma]*mult, line=dict(color=C[col], width=1.2), name=ma.upper()), row=1, col=1)
    if pred_df is not None and not pred_df.empty:
        cand = pred_df[(pred_df["prob_up"] >= min_prob) & (pred_df["expected_net"] >= min_expected_net) & (pred_df.get("final_score", 0) >= min_final_score)]
        if not cand.empty:
            fig.add_trace(go.Scatter(
                x=cand["idx"], y=df.loc[cand["idx"], "low"]*0.965*mult,
                mode="markers", marker=dict(symbol="circle", size=8, color=C["blue"], opacity=0.55),
                name="예측 후보"), row=1, col=1)
    if buys:
        fig.add_trace(go.Scatter(
            x=[b["idx"] for b in buys], y=[df.loc[b["idx"], "low"]*0.94*mult for b in buys],
            mode="markers+text", text=["매수"]*len(buys), textposition="bottom center",
            marker=dict(symbol="triangle-up", size=15, color=C["buy"], line=dict(width=1, color="black")), name="매수"), row=1, col=1)
    if sells:
        fig.add_trace(go.Scatter(
            x=[s["idx"] for s in sells], y=[df.loc[s["idx"], "high"]*1.055*mult for s in sells],
            mode="markers+text", text=[s.get("reason", "매도") for s in sells], textposition="top center",
            marker=dict(symbol="triangle-down", size=15, color=C["sell"], line=dict(width=1, color="black")), name="매도"), row=1, col=1)
    vol_colors = [C["buy"] if df.loc[i, "close"] >= df.loc[i, "open"] else C["sell"] for i in range(len(df))]
    fig.add_trace(go.Bar(x=x, y=df["volume"], marker=dict(color=vol_colors), opacity=0.55, name="거래량"), row=2, col=1)
    step = max(1, len(df)//8)
    fig.update_xaxes(tickmode="array", tickvals=list(df.index[::step]), ticktext=list(df["date"][::step]), row=2, col=1)
    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.72)", height=660, margin=dict(l=10, r=10, t=10, b=8), xaxis_rangeslider_visible=False, hovermode="x unified", showlegend=False, font=dict(family="Pretendard, sans-serif", color=C["text"]))
    return fig


def make_equity_chart(equity_df):
    fig = go.Figure()
    if equity_df is not None and not equity_df.empty:
        fig.add_trace(go.Scatter(x=equity_df["date"], y=equity_df["equity"], mode="lines", name="자산곡선"))
    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.70)", height=310, margin=dict(l=10, r=10, t=12, b=10), hovermode="x unified", font=dict(family="Pretendard, sans-serif", color=C["text"]))
    return fig


# ==========================================================
# 차트 보조지표 및 차트
# ==========================================================

def calculate_indicators(df):
    """기존 지표와 차트용 보조지표를 함께 계산한다."""
    df = df.copy()
    ret = df["close"].pct_change()
    for w in [5, 10, 14, 20, 60, 120, 200]:
        df[f"ma{w}"] = df["close"].rolling(w).mean()
        df[f"ret{w}"] = df["close"].pct_change(w)
        df[f"volatility{w}"] = ret.rolling(w).std()
    df["volMa20"] = df["volume"].rolling(20).mean()
    df["volRatio"] = df["volume"] / df["volMa20"].replace(0, np.nan)
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr14"] / df["close"].replace(0, np.nan)
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df.loc[loss == 0, "rsi"] = 100
    df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema12"] - df["ema26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["bb_mid"] = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std
    typical = (df["high"] + df["low"] + df["close"]) / 3
    vol_cum = df["volume"].replace(0, np.nan).cumsum()
    df["vwap"] = (typical * df["volume"]).cumsum() / vol_cum
    low14 = df["low"].rolling(14).min()
    high14 = df["high"].rolling(14).max()
    df["stoch_k"] = (df["close"] - low14) / (high14 - low14).replace(0, np.nan) * 100
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    df["support60"] = df["low"].rolling(60).min()
    df["resistance60"] = df["high"].rolling(60).max()
    df["candle_body"] = (df["close"] - df["open"]) / df["open"].replace(0, np.nan)
    df["range_pct"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["dollar_volume"] = df["close"] * df["volume"]
    return df


def _chart_theme_values():
    theme = globals().get("chart_visual_theme", "다크 차트")
    if theme == "다크 차트":
        return {"template":"plotly_dark","paper":"rgba(5,10,20,0)","plot":"#0b1120","grid":"rgba(148,163,184,.16)","text":"#e5e7eb","muted":"#94a3b8","inc":"#12b886","dec":"#fa5252"}
    if theme == "라이트 차트":
        return {"template":"plotly_white","paper":"rgba(0,0,0,0)","plot":"#ffffff","grid":"rgba(148,163,184,.22)","text":"#0f172a","muted":"#64748b","inc":"#12b886","dec":"#fa5252"}
    return {"template":"plotly_white","paper":"rgba(0,0,0,0)","plot":"#fbfdff","grid":"rgba(203,213,225,.45)","text":"#111827","muted":"#6b7280","inc":"#16a34a","dec":"#dc2626"}


def _visible_df(df):
    bars = int(globals().get("chart_bars", 260))
    if df is None or df.empty:
        return pd.DataFrame()
    return df.iloc[-bars:].copy() if bars and len(df) > bars else df.copy()


def _clean_aux_indicators():
    aux = list(globals().get("aux_indicators", ["RSI", "MACD"]))
    allowed = ["RSI", "MACD", "Stochastic", "ATR"]
    return [x for x in aux if x in allowed]


def make_price_chart(df, pred_df=None, buys=None, sells=None, min_prob=0.6, min_expected_net=0.01):
    """안정형 캔들 차트. 날짜 파싱 문제를 피하기 위해 x축은 원본 인덱스를 사용한다."""
    tv = _chart_theme_values()
    fig_empty = go.Figure()
    if df is None or df.empty:
        fig_empty.add_annotation(text="차트 데이터를 불러오지 못했습니다.", x=0.5, y=0.5, showarrow=False)
        fig_empty.update_layout(template=tv["template"], paper_bgcolor=tv["paper"], plot_bgcolor=tv["plot"], height=520)
        return fig_empty

    buys = buys or []
    sells = sells or []
    dfp = _visible_df(df)
    if dfp.empty:
        return fig_empty

    start_idx = int(dfp.index.min())
    end_idx = int(dfp.index.max())
    mult = EXCHANGE_RATE if st.session_state.get("currency") == "KRW" else 1.0
    aux = _clean_aux_indicators()
    rows = max(2, 2 + len(aux))
    raw_heights = [0.66, 0.18] + [0.14] * len(aux)
    total_h = sum(raw_heights)
    row_heights = [h / total_h for h in raw_heights]
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=row_heights,
    )

    x = dfp.index.to_list()
    fig.add_trace(
        go.Candlestick(
            x=x,
            open=dfp["open"] * mult,
            high=dfp["high"] * mult,
            low=dfp["low"] * mult,
            close=dfp["close"] * mult,
            name="가격",
            text=dfp["date"],
            increasing_line_color=tv["inc"],
            increasing_fillcolor=tv["inc"],
            decreasing_line_color=tv["dec"],
            decreasing_fillcolor=tv["dec"],
            hovertemplate="%{text}<br>시가 %{open:,.2f}<br>고가 %{high:,.2f}<br>저가 %{low:,.2f}<br>종가 %{close:,.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    if globals().get("show_bollinger", True) and {"bb_upper", "bb_lower"}.issubset(dfp.columns):
        fig.add_trace(go.Scatter(x=x, y=dfp["bb_upper"] * mult, line=dict(width=1, color="rgba(99,102,241,.45)"), name="상단 밴드", hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=dfp["bb_lower"] * mult, fill="tonexty", fillcolor="rgba(99,102,241,.08)", line=dict(width=1, color="rgba(99,102,241,.45)"), name="하단 밴드", hoverinfo="skip"), row=1, col=1)

    ma_map = {"MA5": ("ma5", C["ma5"]), "MA20": ("ma20", C["ma20"]), "MA60": ("ma60", C["ma60"]), "MA120": ("ma120", C["ma120"]), "MA200": ("ma200", "#475569")}
    for label in globals().get("ma_lines", ["MA5", "MA20", "MA60", "MA120"]):
        col, color = ma_map.get(label, (None, None))
        if col and col in dfp.columns:
            fig.add_trace(go.Scatter(x=x, y=dfp[col] * mult, line=dict(color=color, width=1.45), name=label, hovertemplate=f"{label} %{{y:,.2f}}<extra></extra>"), row=1, col=1)

    if globals().get("show_vwap", True) and "vwap" in dfp.columns:
        fig.add_trace(go.Scatter(x=x, y=dfp["vwap"] * mult, line=dict(color="#a855f7", width=1.2, dash="dot"), name="VWAP"), row=1, col=1)

    if globals().get("show_support_resistance", True):
        if "resistance60" in dfp.columns:
            fig.add_trace(go.Scatter(x=x, y=dfp["resistance60"] * mult, line=dict(color="rgba(250,82,82,.65)", width=1, dash="dash"), name="저항60", hoverinfo="skip"), row=1, col=1)
        if "support60" in dfp.columns:
            fig.add_trace(go.Scatter(x=x, y=dfp["support60"] * mult, line=dict(color="rgba(18,184,134,.65)", width=1, dash="dash"), name="지지60", hoverinfo="skip"), row=1, col=1)

    if globals().get("show_last_price_line", True) and len(dfp) > 0:
        last_price = float(dfp["close"].iloc[-1]) * mult
        fig.add_hline(y=last_price, row=1, col=1, line_width=1, line_dash="dot", line_color=tv["muted"], annotation_text=f"현재가 {last_price:,.2f}", annotation_position="top right")

    if pred_df is not None and not pred_df.empty and {"idx", "prob_up", "expected_net"}.issubset(pred_df.columns):
        fs = pred_df["final_score"] if "final_score" in pred_df.columns else pd.Series(1.0, index=pred_df.index)
        cand = pred_df[(pred_df["idx"] >= start_idx) & (pred_df["idx"] <= end_idx) & (pred_df["prob_up"] >= min_prob) & (pred_df["expected_net"] >= min_expected_net) & (fs >= globals().get("min_final_score", 0.62))]
        if not cand.empty:
            cand_idx = cand["idx"].astype(int).tolist()
            valid_idx = [i for i in cand_idx if i in df.index]
            if valid_idx:
                fig.add_trace(go.Scatter(x=valid_idx, y=(df.loc[valid_idx, "low"] * 0.975 * mult), mode="markers", marker=dict(symbol="circle", size=9, color="#3182f6", opacity=.72, line=dict(width=1, color="#fff")), name="예측 후보", hovertemplate="예측 후보<extra></extra>"), row=1, col=1)

    b = [p for p in buys if start_idx <= int(p.get("idx", -1)) <= end_idx and int(p.get("idx", -1)) in df.index]
    if b:
        b_idx = [int(p["idx"]) for p in b]
        fig.add_trace(go.Scatter(x=b_idx, y=[df.loc[i, "low"] * 0.94 * mult for i in b_idx], mode="markers+text", text=["매수"] * len(b_idx), textposition="bottom center", marker=dict(symbol="triangle-up", size=16, color=tv["inc"], line=dict(width=1, color="#ffffff")), name="매수"), row=1, col=1)

    s = [p for p in sells if start_idx <= int(p.get("idx", -1)) <= end_idx and int(p.get("idx", -1)) in df.index]
    if s:
        s_idx = [int(p["idx"]) for p in s]
        fig.add_trace(go.Scatter(x=s_idx, y=[df.loc[i, "high"] * 1.055 * mult for i in s_idx], mode="markers+text", text=[p.get("reason", "매도") for p in s], textposition="top center", marker=dict(symbol="triangle-down", size=16, color=tv["dec"], line=dict(width=1, color="#ffffff")), name="매도"), row=1, col=1)

    vol_colors = [tv["inc"] if dfp.loc[i, "close"] >= dfp.loc[i, "open"] else tv["dec"] for i in dfp.index]
    fig.add_trace(go.Bar(x=x, y=dfp["volume"], marker=dict(color=vol_colors), opacity=0.55, name="거래량"), row=2, col=1)
    if globals().get("show_volume_ma", True) and "volMa20" in dfp.columns:
        fig.add_trace(go.Scatter(x=x, y=dfp["volMa20"], line=dict(color="#64748b", width=1.2), name="거래량 MA20"), row=2, col=1)

    row = 3
    for ind in aux:
        if row > rows:
            break
        if ind == "RSI" and "rsi" in dfp.columns:
            fig.add_trace(go.Scatter(x=x, y=dfp["rsi"], line=dict(color="#f472b6", width=1.4), name="RSI"), row=row, col=1)
            fig.add_hrect(y0=70, y1=100, fillcolor="rgba(250,82,82,.10)", line_width=0, row=row, col=1)
            fig.add_hrect(y0=0, y1=30, fillcolor="rgba(18,184,134,.10)", line_width=0, row=row, col=1)
            for yv in [30, 50, 70]:
                fig.add_hline(y=yv, row=row, col=1, line_width=.8, line_dash="dot", line_color=tv["grid"])
            fig.update_yaxes(range=[0, 100], row=row, col=1)
        elif ind == "MACD" and {"macd", "macd_signal", "macd_hist"}.issubset(dfp.columns):
            hist_colors = [tv["inc"] if v >= 0 else tv["dec"] for v in dfp["macd_hist"].fillna(0)]
            fig.add_trace(go.Bar(x=x, y=dfp["macd_hist"], marker=dict(color=hist_colors), opacity=.45, name="MACD Hist"), row=row, col=1)
            fig.add_trace(go.Scatter(x=x, y=dfp["macd"], line=dict(color="#60a5fa", width=1.3), name="MACD"), row=row, col=1)
            fig.add_trace(go.Scatter(x=x, y=dfp["macd_signal"], line=dict(color="#f59e0b", width=1.1), name="Signal"), row=row, col=1)
            fig.add_hline(y=0, row=row, col=1, line_width=.8, line_color=tv["grid"])
        elif ind == "Stochastic" and {"stoch_k", "stoch_d"}.issubset(dfp.columns):
            fig.add_trace(go.Scatter(x=x, y=dfp["stoch_k"], line=dict(color="#22c55e", width=1.2), name="%K"), row=row, col=1)
            fig.add_trace(go.Scatter(x=x, y=dfp["stoch_d"], line=dict(color="#f97316", width=1.2), name="%D"), row=row, col=1)
            fig.add_hline(y=80, row=row, col=1, line_width=.8, line_dash="dot", line_color=tv["grid"])
            fig.add_hline(y=20, row=row, col=1, line_width=.8, line_dash="dot", line_color=tv["grid"])
            fig.update_yaxes(range=[0, 100], row=row, col=1)
        elif ind == "ATR" and "atr_pct" in dfp.columns:
            fig.add_trace(go.Scatter(x=x, y=dfp["atr_pct"] * 100, line=dict(color="#a855f7", width=1.3), name="ATR %"), row=row, col=1)
        row += 1

    step = max(1, len(dfp) // 8)
    tickvals = dfp.index.to_list()[::step]
    ticktext = dfp["date"].astype(str).to_list()[::step]
    for r in range(1, rows + 1):
        fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext, showgrid=True, gridcolor=tv["grid"], zeroline=False, showspikes=True, spikemode="across", spikesnap="cursor", spikecolor=tv["muted"], spikethickness=1, row=r, col=1)
        fig.update_yaxes(showgrid=True, gridcolor=tv["grid"], zeroline=False, showspikes=True, spikecolor=tv["muted"], row=r, col=1)

    fig.update_layout(
        template=tv["template"],
        paper_bgcolor=tv["paper"],
        plot_bgcolor=tv["plot"],
        height=620 + 110 * len(aux),
        margin=dict(l=8, r=8, t=28, b=8),
        xaxis_rangeslider_visible=bool(globals().get("show_range_slider", False)),
        hovermode="x unified",
        dragmode="pan",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, font=dict(size=11)),
        font=dict(family="Pretendard, sans-serif", color=tv["text"]),
        title=dict(text="", x=0.01, y=0.99, font=dict(size=13, color=tv["muted"])),
    )
    return fig


def make_equity_chart(equity_df):
    tv = _chart_theme_values()
    fig = go.Figure()
    if equity_df is not None and not equity_df.empty:
        eq = equity_df.copy()
        x = list(range(len(eq)))
        fig.add_trace(go.Scatter(x=x, y=eq["equity"], mode="lines", fill="tozeroy", fillcolor="rgba(49,130,246,.10)", line=dict(color="#3182f6", width=2.2), name="자산곡선"))
        roll_max = eq["equity"].cummax()
        dd = (eq["equity"] / roll_max - 1) * 100
        fig.add_trace(go.Scatter(x=x, y=dd, yaxis="y2", line=dict(color="#fa5252", width=1.1, dash="dot"), name="낙폭 %"))
        step = max(1, len(eq) // 6)
        fig.update_xaxes(tickmode="array", tickvals=x[::step], ticktext=eq["date"].astype(str).to_list()[::step])
    fig.update_layout(template=tv["template"], paper_bgcolor=tv["paper"], plot_bgcolor=tv["plot"], height=340, margin=dict(l=8, r=8, t=20, b=10), hovermode="x unified", font=dict(family="Pretendard, sans-serif", color=tv["text"]), yaxis=dict(title="자산", gridcolor=tv["grid"]), yaxis2=dict(title="낙폭%", overlaying="y", side="right", showgrid=False), legend=dict(orientation="h", y=1.04, x=0))
    return fig


# ==========================================================
# UI 설정
# ==========================================================

st.sidebar.markdown(f"<h3 style='color:{C['text']}'>⚙️ 설정</h3>", unsafe_allow_html=True)
if st.sidebar.button("모의투자 초기화", use_container_width=True):
    reset_local_portfolio()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(f"<h3 style='color:{C['text']}'>⚙️ 표시 설정</h3>", unsafe_allow_html=True)
currency_mode = st.sidebar.radio("표시 통화", ["달러 ($)", "원 (₩)"], label_visibility="collapsed")
st.session_state.currency = "USD" if "달러" in currency_mode else "KRW"

st.sidebar.markdown("---")
st.sidebar.markdown(f"<h3 style='color:{C['text']}'>📊 차트 디자인</h3>", unsafe_allow_html=True)
chart_visual_theme = st.sidebar.selectbox("차트 테마", ["다크 차트", "라이트 차트", "미니멀 라이트"], index=0)
chart_bars = st.sidebar.slider("차트 표시 봉 수", 80, 1000, 260, 20, help="최근 몇 개 봉만 차트에 표시할지 정합니다. 낮추면 더 빠르고 보기 쉽습니다.")
ma_lines = st.sidebar.multiselect("이동평균선", ["MA5", "MA20", "MA60", "MA120", "MA200"], default=["MA5", "MA20", "MA60", "MA120"])
aux_indicators = st.sidebar.multiselect("보조지표 패널", ["RSI", "MACD", "Stochastic", "ATR"], default=["RSI", "MACD"])
show_bollinger = st.sidebar.checkbox("볼린저밴드", value=True)
show_vwap = st.sidebar.checkbox("VWAP", value=True)
show_support_resistance = st.sidebar.checkbox("60봉 지지/저항선", value=True)
show_volume_ma = st.sidebar.checkbox("거래량 평균선", value=True)
show_last_price_line = st.sidebar.checkbox("현재가 기준선", value=True)
show_range_slider = st.sidebar.checkbox("하단 기간 조절 바", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown(f"<h3 style='color:{C['text']}'>🧠 예측 모델</h3>", unsafe_allow_html=True)
analysis_mode = st.sidebar.radio("분석 속도", ["빠른 실전 모드", "정밀 검증 모드"], index=0, horizontal=True)
lookback_window = st.sidebar.selectbox("입력 시계열 길이", [20, 60, 120], index=1)
forecast_horizon = st.sidebar.selectbox("예측 기간", [3, 5, 10, 20], index=1)
train_window = st.sidebar.slider("Walk-forward 학습 표본 수", 120, 900, 320, 10)
model_default = 3 if analysis_mode == "빠른 실전 모드" else 0
model_mode = st.sidebar.selectbox("모델 방식", ["앙상블", "GradientBoosting", "RandomForest", "Ridge", "Logistic"], index=model_default)
pred_step = st.sidebar.slider("백테스트 계산 간격", 1, 20, 8 if analysis_mode == "빠른 실전 모드" else 3, 1, help="숫자가 클수록 빠르지만 검증 표본이 줄어듭니다.")
min_prob = st.sidebar.slider("최소 상승확률", 0.50, 0.90, 0.60, 0.01)
min_expected_net = st.sidebar.slider("최소 비용차감 기대수익률", 0.0, 10.0, 1.2, 0.1, format="%.1f%%") / 100
min_final_score = st.sidebar.slider("최소 종합점수", 0.30, 0.95, 0.62, 0.01)
risk_strength = st.sidebar.slider("리스크 감점 강도", 0.0, 2.0, 0.9, 0.05)
use_market_filter = st.sidebar.checkbox("시장 추세 필터 사용", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown(f"<h3 style='color:{C['text']}'>🎚️ 매수 점수 가중치</h3>", unsafe_allow_html=True)
weight_preset = st.sidebar.selectbox("가중치 프리셋", ["균형형", "추세추종형", "눌림목형", "단기모멘텀형", "보수방어형", "직접조정"], index=0)
PRESET_WEIGHTS = {
    "균형형": {"trend": 1.0, "momentum": 1.0, "rsi": 1.0, "volume": 0.8, "volatility": 0.8, "pullback": 1.0, "market": 1.0, "risk_inverse": 1.0},
    "추세추종형": {"trend": 1.8, "momentum": 1.2, "rsi": 0.7, "volume": 0.7, "volatility": 0.7, "pullback": 0.5, "market": 1.4, "risk_inverse": 1.0},
    "눌림목형": {"trend": 1.2, "momentum": 0.7, "rsi": 1.2, "volume": 0.8, "volatility": 0.9, "pullback": 1.8, "market": 1.0, "risk_inverse": 1.1},
    "단기모멘텀형": {"trend": 0.9, "momentum": 1.8, "rsi": 0.8, "volume": 1.3, "volatility": 0.6, "pullback": 0.7, "market": 0.8, "risk_inverse": 0.8},
    "보수방어형": {"trend": 1.3, "momentum": 0.6, "rsi": 0.9, "volume": 0.7, "volatility": 1.7, "pullback": 1.0, "market": 1.5, "risk_inverse": 1.8},
}
base_w = PRESET_WEIGHTS.get(weight_preset, PRESET_WEIGHTS["균형형"]).copy()
with st.sidebar.expander("세부 가중치 조정", expanded=(weight_preset == "직접조정")):
    w_trend = st.slider("추세 가중치", 0.0, 3.0, float(base_w["trend"]), 0.1)
    w_momentum = st.slider("모멘텀 가중치", 0.0, 3.0, float(base_w["momentum"]), 0.1)
    w_rsi = st.slider("RSI 회복 가중치", 0.0, 3.0, float(base_w["rsi"]), 0.1)
    w_volume = st.slider("거래량 가중치", 0.0, 3.0, float(base_w["volume"]), 0.1)
    w_volatility = st.slider("변동성 안정 가중치", 0.0, 3.0, float(base_w["volatility"]), 0.1)
    w_pullback = st.slider("눌림목 가중치", 0.0, 3.0, float(base_w["pullback"]), 0.1)
    w_market = st.slider("시장필터 가중치", 0.0, 3.0, float(base_w["market"]), 0.1)
    w_risk_inv = st.slider("저위험 가중치", 0.0, 3.0, float(base_w["risk_inverse"]), 0.1)
ml_weight = st.sidebar.slider("AI 예측 비중", 0.0, 1.0, 0.65, 0.05, help="높을수록 머신러닝 예측을 더 믿고, 낮을수록 사용자가 정한 기술적 가중치를 더 믿습니다.")
score_weights = {"trend": w_trend, "momentum": w_momentum, "rsi": w_rsi, "volume": w_volume, "volatility": w_volatility, "pullback": w_pullback, "market": w_market, "risk_inverse": w_risk_inv}

st.sidebar.markdown("---")
st.sidebar.markdown(f"<h3 style='color:{C['text']}'>💸 거래비용</h3>", unsafe_allow_html=True)
fee_pct = st.sidebar.slider("매수·매도 수수료", 0.0, 1.0, 0.05, 0.01, format="%.2f%%") / 100
slippage_pct = st.sidebar.slider("슬리피지", 0.0, 2.0, 0.10, 0.01, format="%.2f%%") / 100
sell_tax_pct = st.sidebar.slider("매도세/기타 비용", 0.0, 1.0, 0.00, 0.01, format="%.2f%%") / 100
cost_cfg = CostConfig(fee_pct=fee_pct, slippage_pct=slippage_pct, sell_tax_pct=sell_tax_pct)

st.sidebar.markdown("---")
st.sidebar.markdown(f"<h3 style='color:{C['text']}'>🛡️ 리스크 관리</h3>", unsafe_allow_html=True)
start_capital = st.sidebar.number_input("백테스트 시작자산($)", min_value=100.0, value=10000.0, step=100.0)
risk_per_trade = st.sidebar.slider("1회 거래 허용 손실", 0.1, 5.0, 1.0, 0.1, format="%.1f%%") / 100
max_position_pct = st.sidebar.slider("최대 투입 비중", 5.0, 100.0, 30.0, 1.0, format="%.0f%%") / 100
fixed_stop_pct = st.sidebar.slider("최소 고정 손절폭", 1.0, 30.0, 5.0, 0.5, format="%.1f%%") / 100
atr_stop_mult = st.sidebar.slider("ATR 손절 배수", 0.5, 5.0, 1.8, 0.1)
take_profit_pct = st.sidebar.slider("익절 목표", 1.0, 60.0, 12.0, 0.5, format="%.1f%%") / 100
trailing_stop_pct = st.sidebar.slider("트레일링 스탑", 0.0, 30.0, 6.0, 0.5, format="%.1f%%") / 100
max_hold_bars = st.sidebar.slider("최대 보유 봉 수", 5, 300, 80, 5)
max_risk_score = st.sidebar.slider("허용 리스크 점수", 0.20, 1.00, 0.70, 0.01)
min_dollar_volume = st.sidebar.number_input("최소 유동성($ 거래대금)", min_value=0.0, value=0.0, step=100000.0)
risk_cfg = RiskConfig(start_capital, risk_per_trade, max_position_pct, fixed_stop_pct, take_profit_pct, atr_stop_mult, trailing_stop_pct, max_hold_bars, max_risk_score, min_dollar_volume)

if not SKLEARN_AVAILABLE:
    st.sidebar.warning("scikit-learn 설치 필요: pip install scikit-learn")

# ==========================================================
# 화면 렌더링
# ==========================================================

def render_prediction_card(latest, market):
    if latest is None:
        st.warning("모델 학습에 필요한 데이터가 부족합니다.")
        return
    grade = latest["grade"]
    badge_cls = {"S":"badge-green", "A":"badge-green", "B":"badge-blue", "관망":"badge-gray", "금지":"badge-red"}.get(grade, "badge-gray")
    grade_msg = {"S":"강한 매수 후보", "A":"매수 후보", "B":"관심 후보", "관망":"관망", "금지":"매수 회피"}.get(grade, latest.get("reason", ""))
    html = f'''
    <div class="terminal-card">
      <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:10px;">
        <div><div class="muted">SIGNAL TERMINAL</div><div style="font-size:22px;font-weight:950;letter-spacing:-.04em;color:#fff;">{grade_msg}</div></div>
        <span class="badge {badge_cls}">{grade}</span>
      </div>
      <div style="font-size:13px;line-height:1.8;color:#cbd5e1;">
        예측일 <b style="color:#fff;">{latest['date']}</b><br>
        {forecast_horizon}봉 예상 수익률 <b style="color:#fff;">{latest['pred_return']*100:+.2f}%</b><br>
        비용차감 기대수익률 <b style="color:#fff;">{latest['expected_net']*100:+.2f}%</b><br>
        시장필터 <b style="color:#fff;">{market['text']}</b>
      </div>
      {_pct_bar('상승확률', latest['prob_up'], '#3182f6')}
      {_pct_bar('종합점수', latest.get('final_score', 0), '#12b886')}
      {_pct_bar('AI 점수', latest.get('ml_score', 0), '#8b5cf6')}
      {_pct_bar('기술점수', latest.get('tech_score', 0), '#f59e0b')}
      {_pct_bar('리스크 점수', latest['risk_score'], '#fa5252')}
    </div>
    '''
    st.markdown(html, unsafe_allow_html=True)
    comps = latest.get("components", {})
    if comps:
        with st.expander("가중치 점수 구성", expanded=False):
            comp_df = pd.DataFrame([{"항목": k, "점수": f"{v*100:.1f}%", "가중치": score_weights.get(k, 0)} for k, v in comps.items()])
            st.dataframe(comp_df, use_container_width=True, hide_index=True)


def render_trade_plan(df, latest):
    if latest is None or df is None or df.empty:
        return
    i = len(df) - 1
    close = float(df.loc[i, "close"])
    atr_pct = nan_to_zero(df.loc[i, "atr_pct"]) if "atr_pct" in df else 0.0
    stop_pct = max(risk_cfg.fixed_stop_pct, risk_cfg.atr_stop_mult * atr_pct)
    stop_pct = min(max(stop_pct, 0.005), 0.35)
    stop = close * (1 - stop_pct)
    target = close * (1 + risk_cfg.take_profit_pct)
    qty, used_cap = calc_position_size(risk_cfg.start_capital, close, stop, risk_cfg)
    rr = (target - close) / max(close - stop, 1e-9)
    st.markdown(f'''
    <div class="trade-panel" style="padding:14px 15px;margin-top:12px;">
      <div style="font-weight:950;color:#0f172a;margin-bottom:8px;">주문 계획 시뮬레이터</div>
      <div class="plan-grid">
        <div class="plan-item"><div class="plan-label">예상 진입가</div><div class="plan-value">{fmt_curr(close)}</div></div>
        <div class="plan-item"><div class="plan-label">손절 기준</div><div class="plan-value" style="color:#fa5252;">{fmt_curr(stop)} / -{stop_pct*100:.1f}%</div></div>
        <div class="plan-item"><div class="plan-label">목표가</div><div class="plan-value" style="color:#12b886;">{fmt_curr(target)} / +{risk_cfg.take_profit_pct*100:.1f}%</div></div>
        <div class="plan-item"><div class="plan-label">손익비</div><div class="plan-value">1 : {rr:.2f}</div></div>
        <div class="plan-item"><div class="plan-label">권장 수량</div><div class="plan-value">{qty:,.4f}</div></div>
        <div class="plan-item"><div class="plan-label">예상 투입금</div><div class="plan-value">{fmt_curr(used_cap)}</div></div>
      </div>
    </div>
    ''', unsafe_allow_html=True)


def render_metrics(metrics, acc, acc_n):
    def fmt_pf(v):
        return "∞" if v >= 999 else f"{v:.2f}"
    cards = [("총수익률", f"{metrics['총수익률']:+.2f}%", "백테스트 최종 수익"), ("MDD", f"{metrics['MDD']:.2f}%", "최대 낙폭"), ("승률", f"{metrics['승률']:.1f}%", "수익 거래 비율"), ("Profit Factor", fmt_pf(metrics['Profit Factor']), "총수익 / 총손실"), ("Sharpe", f"{metrics['Sharpe']:.2f}", "변동성 대비 수익"), ("거래횟수", f"{metrics['거래횟수']}회", "검증 거래 수"), ("평균수익", f"{metrics['평균수익']:+.2f}%", "승리 거래 평균"), ("평균손실", f"{metrics['평균손실']:+.2f}%", "패배 거래 평균"), ("기대값/거래", f"{metrics['기대값']:+.2f}%", "1회 거래 기대값"), ("방향 적중률", f"{acc:.1f}%", f"검증 표본 {acc_n}개")]
    html = '<div class="kpi-grid">'
    for label, value, sub in cards:
        color = '#12b886' if ('+' in value and label != 'MDD') else ('#fa5252' if label in ['MDD','평균손실'] and '-' in value else '#0f172a')
        html += f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value" style="color:{color};">{value}</div><div class="kpi-sub">{sub}</div></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def analyze_symbol(symbol, interval, data_range):
    df_raw, name = fetch_ohlcv(symbol, interval, data_range)
    if df_raw is None or len(df_raw) < max(lookback_window, 200) + forecast_horizon + 80:
        return None, name, None, None, None, [], pd.DataFrame(), [], [], {}, 0.0, 0, None
    df = calculate_indicators(df_raw)
    market = get_market_regime(symbol, interval, data_range)
    market_ok = market["ok"] if use_market_filter else True
    pred_df = walk_forward_predictions(df, lookback_window, forecast_horizon, train_window, model_mode, risk_strength, cost_cfg, market["score"], step=pred_step, weights=score_weights, ml_weight=ml_weight)
    latest = latest_prediction(df, lookback_window, forecast_horizon, train_window, model_mode, risk_strength, cost_cfg, market["score"], market_ok, weights=score_weights, ml_weight=ml_weight)
    trades, equity_df, buys, sells, active = run_backtest(df, pred_df, min_prob, min_expected_net, min_final_score, cost_cfg, risk_cfg, market["ok"], use_market_filter)
    metrics = calc_metrics(trades, equity_df, risk_cfg.start_capital)
    acc, acc_n = prediction_accuracy(pred_df)
    return df, name, market, pred_df, latest, trades, equity_df, buys, sells, metrics, acc, acc_n, active


def render_asset_tab(asset_type, is_coin, default_list, default_input):
    label = "코인" if is_coin else "주식"
    st.markdown(f'<div class="section-card"><h3 style="margin:0;color:{C["text"]};font-weight:900;">{"🪙" if is_coin else "📈"} {label}</h3></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([3, 4, 3])
    with c1:
        method = st.radio(f"{label} 검색 방식", ["목록에서 고르기", "직접 입력"], key=f"{asset_type}_method", label_visibility="collapsed")
    with c2:
        if method == "목록에서 고르기":
            user_input = st.selectbox(f"{label} 선택", default_list, key=f"{asset_type}_select")
        else:
            user_input = st.text_input(f"{label}명 또는 티커", default_input, key=f"{asset_type}_input").strip()
    with c3:
        time_label = st.selectbox("차트 주기", list(TIMEFRAME_MAP.keys()), index=0, key=f"{asset_type}_time")
    interval, data_range = TIMEFRAME_MAP[time_label]
    symbol = search_ticker_by_name(user_input, is_coin=is_coin)
    if not symbol:
        st.warning("종목을 입력해 주세요.")
        return

    result = analyze_symbol(symbol, interval, data_range)
    df, name, market, pred_df, latest, trades, equity_df, buys, sells, metrics, acc, acc_n, active = result
    if df is None:
        st.error("데이터가 부족하거나 불러오지 못했습니다. 일봉처럼 더 긴 기간이 있는 차트를 선택해 주세요.")
        return

    last = len(df) - 1
    current = df.loc[last, "close"]
    prev = df.loc[last - 1, "close"]
    diff = current - prev
    pct = diff / prev * 100 if prev else 0
    sign = "+" if diff > 0 else ""
    color = C["buy"] if diff >= 0 else C["sell"]
    price_class = "price-up" if diff >= 0 else "price-down"
    st.markdown(f"""
    <div class="asset-header"><div style="display:flex;align-items:center;justify-content:space-between;gap:14px;flex-wrap:wrap;"><div><h2 style="margin:0;color:{C['text']};font-size:25px;font-weight:900;letter-spacing:-0.04em;">{name}<span class="asset-symbol">{symbol}</span></h2><div class="asset-price">{fmt_curr(current)}<span class="{price_class}">{sign}{fmt_curr(diff)} ({sign}{pct:.2f}%)</span></div></div><div style="text-align:right;"><div class="small-muted">{time_label}</div><div class="small-muted" style="margin-top:5px;">{model_mode} · {forecast_horizon}봉 예측</div></div></div></div>
    """, unsafe_allow_html=True)

    left, right = st.columns([7.5, 2.5])
    with left:
        st.plotly_chart(make_price_chart(df, pred_df, buys, sells, min_prob, min_expected_net), use_container_width=True, config={"scrollZoom": True}, key=f"chart_{asset_type}")
        render_metrics(metrics, acc, acc_n)
        st.markdown("### 자산곡선")
        st.plotly_chart(make_equity_chart(equity_df), use_container_width=True, key=f"equity_{asset_type}")
    with right:
        render_prediction_card(latest, market)
        render_trade_plan(df, latest)
        st.markdown(f"""
        <div style="background:{C['surface']}; border:1px solid {C['border']}; border-radius:10px; padding:14px; margin-top:12px;">
        <b>자동 수량 계산 방식</b><br>
        <span style="font-size:13px; line-height:1.7; color:{C['subtext']};">
        1회 허용 손실금액 = 계좌 × {risk_per_trade*100:.1f}%<br>
        수량 = 허용 손실금액 ÷ 손절폭<br>
        최대 투입 비중 = {max_position_pct*100:.0f}%
        </span></div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 백테스트 거래 기록")
    if trades:
        h = pd.DataFrame(trades)
        h = h[["entryDate", "entryPrice", "exitDate", "exitPrice", "qty", "ret", "exitReason", "prob_up", "expected_net", "risk_score", "final_score", "capital_after", "win"]]
        h.columns = ["매수일", "매수가", "매도일", "매도가", "수량", "수익률", "청산이유", "상승확률", "비용차감 기대수익", "리스크", "종합점수", "청산 후 자산", "성공"]
        h["매수가"] = h["매수가"].apply(fmt_curr)
        h["매도가"] = h["매도가"].apply(fmt_curr)
        h["수익률"] = h["수익률"].map("{:+.2f}%".format)
        h["상승확률"] = h["상승확률"].map(lambda x: f"{x*100:.1f}%")
        h["비용차감 기대수익"] = h["비용차감 기대수익"].map(lambda x: f"{x*100:+.2f}%")
        h["리스크"] = h["리스크"].map(lambda x: f"{x*100:.1f}%")
        h["종합점수"] = h["종합점수"].map(lambda x: f"{x*100:.1f}%")
        h["청산 후 자산"] = h["청산 후 자산"].apply(fmt_curr)
        h["성공"] = h["성공"].map(lambda x: "✅" if x else "❌")
        st.dataframe(h.iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.info("현재 조건으로는 거래 기록이 없습니다. 기준을 완화하거나 차트 주기를 바꿔보세요.")

    st.markdown("---")
    st.markdown(f"### 수동 {label} 모의투자")
    share_key = "coin_shares" if is_coin else "stock_shares"
    balance_key = "coin_balance" if is_coin else "stock_balance"
    log_key = "coin_trade_log" if is_coin else "stock_trade_log"
    st.session_state[share_key].setdefault(symbol, 0.0)
    a1, a2, a3 = st.columns([2, 2, 3])
    a1.metric("예수금", fmt_curr(st.session_state[balance_key]))
    a2.metric("보유수량", f"{st.session_state[share_key][symbol]:.4f}")
    with a3:
        amount = st.number_input("수량", min_value=0.0001, value=1.0 if (is_korean_ticker(symbol) and not is_coin) else 0.1, step=1.0 if (is_korean_ticker(symbol) and not is_coin) else 0.01, key=f"manual_amt_{asset_type}")
    b1, b2 = st.columns(2)
    with b1:
        if st.button(f"🔴 {label} 매수", use_container_width=True, key=f"buy_{asset_type}"):
            total = amount * current
            if st.session_state[balance_key] >= total:
                st.session_state[balance_key] -= total
                st.session_state[share_key][symbol] += amount
                st.session_state[log_key].append({"시간": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "종목": name, "종류": "매수", "수량": amount, "가격": current, "총금액": total})
                save_current_user_data(); st.rerun()
            else:
                st.error("잔액이 부족합니다.")
    with b2:
        if st.button(f"🔵 {label} 매도", use_container_width=True, key=f"sell_{asset_type}"):
            if st.session_state[share_key][symbol] >= amount:
                total = amount * current
                st.session_state[balance_key] += total
                st.session_state[share_key][symbol] -= amount
                st.session_state[log_key].append({"시간": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "종목": name, "종류": "매도", "수량": amount, "가격": current, "총금액": total})
                save_current_user_data(); st.rerun()
            else:
                st.error("보유 수량이 부족합니다.")
    if st.session_state[log_key]:
        log = pd.DataFrame(st.session_state[log_key])
        log["가격"] = log["가격"].apply(fmt_curr)
        log["총금액"] = log["총금액"].apply(fmt_curr)
        st.dataframe(log.iloc[::-1], use_container_width=True, hide_index=True)


def render_scanner():
    st.markdown("### 여러 종목 스캐너")
    scan_type = st.radio("스캔 대상", ["주식", "코인", "직접 입력"], horizontal=True)
    if scan_type == "주식":
        base_list = STOCK_SCAN_LIST
    elif scan_type == "코인":
        base_list = COIN_SCAN_LIST
    else:
        raw = st.text_area("티커를 쉼표로 입력", "AAPL,NVDA,TSLA,BTC-USD,ETH-USD")
        base_list = [x.strip().upper() for x in raw.split(",") if x.strip()]
    max_scan = st.slider("최대 스캔 개수", 1, min(20, len(base_list)), min(8, len(base_list)))
    time_label = st.selectbox("스캐너 차트 주기", list(TIMEFRAME_MAP.keys()), index=0, key="scanner_time")
    interval, data_range = TIMEFRAME_MAP[time_label]
    if st.button("스캔 실행", use_container_width=True):
        rows = []
        progress = st.progress(0)
        for k, sym in enumerate(base_list[:max_scan]):
            progress.progress((k + 1) / max_scan)
            try:
                result = analyze_symbol(sym, interval, data_range)
                df, name, market, pred_df, latest, trades, equity_df, buys, sells, metrics, acc, acc_n, active = result
                if df is None or latest is None:
                    rows.append({"종목": sym, "이름": TICKER_NAME_MAP.get(sym, sym), "등급": "데이터부족"})
                    continue
                rows.append({
                    "종목": sym, "이름": name, "등급": latest["grade"], "사유": latest["reason"],
                    "상승확률": latest["prob_up"] * 100, "종합점수": latest.get("final_score", 0) * 100, "기술점수": latest.get("tech_score", 0) * 100, "비용차감 기대수익": latest["expected_net"] * 100,
                    "리스크": latest["risk_score"] * 100, "시장": market["text"],
                    "총수익률": metrics["총수익률"], "MDD": metrics["MDD"], "PF": metrics["Profit Factor"],
                    "Sharpe": metrics["Sharpe"], "거래횟수": metrics["거래횟수"], "방향적중률": acc,
                })
            except Exception as e:
                rows.append({"종목": sym, "이름": TICKER_NAME_MAP.get(sym, sym), "등급": "오류", "사유": str(e)[:80]})
        out = pd.DataFrame(rows)
        if not out.empty and "상승확률" in out:
            grade_order = {"S": 0, "A": 1, "B": 2, "관망": 3, "금지": 4, "데이터부족": 5, "오류": 6}
            out["grade_rank"] = out["등급"].map(grade_order).fillna(9)
            out = out.sort_values(["grade_rank", "종합점수", "비용차감 기대수익", "상승확률"], ascending=[True, False, False, False]).drop(columns=["grade_rank"])
            for col in ["상승확률", "종합점수", "기술점수", "비용차감 기대수익", "리스크", "총수익률", "MDD", "Sharpe", "방향적중률"]:
                if col in out:
                    out[col] = out[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
            if "PF" in out:
                out["PF"] = out["PF"].map(lambda x: "∞" if pd.notna(x) and x >= 999 else (f"{x:.2f}" if pd.notna(x) else ""))
        st.dataframe(out, use_container_width=True, hide_index=True)



page = st.radio(
    "",
    ["주식", "코인", "스캐너"],
    horizontal=True,
    label_visibility="collapsed",
)

if page == "주식":
    render_asset_tab("stock", False, ["애플", "삼성전자", "SK하이닉스", "테슬라", "엔비디아", "마이크로소프트", "구글", "네이버", "카카오", "현대차"], "애플")
elif page == "코인":
    render_asset_tab("coin", True, ["비트코인", "이더리움", "솔라나", "리플", "도지코인", "아발란체"], "비트코인")
else:
    render_scanner()
