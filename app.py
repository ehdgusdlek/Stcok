"""
시계열 수학 모델 기반 주식·코인 매수 타점 예측 프로그램
- 로그인 없음
- 핵심 항목만 노출하고 고급 설정은 접어두기
- 수학적 모델: Ridge 회귀, Logistic 회귀, Gradient Boosting, Random Forest, 선택형 ARIMA
- 예측: 미래 수익률, 상승확률, 종합점수
- 검증: 거래비용, 슬리피지, 손절, 익절, 추적손절, 추세이탈, MDD, PF, Sharpe
- 장중 실시간 갱신: 1분봉 지원 및 장중 자동 새로고침

주의: 이 프로그램은 교육·연구용입니다. 실제 수익을 보장하지 않습니다.
"""

from __future__ import annotations

import json
import math
import os
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False

# ============================================================
# 기본 설정
# ============================================================

st.set_page_config(
    page_title="수학 모델 매수 타점",
    layout="wide",
    initial_sidebar_state="expanded",
)

C = {
    "bg": "#f5f7fb",
    "card": "#ffffff",
    "line": "#d8e0ea",
    "text": "#0f172a",
    "muted": "#64748b",
    "blue": "#2563eb",
    "green": "#10b981",
    "red": "#ef4444",
    "orange": "#f59e0b",
    "purple": "#7c3aed",
    "cyan": "#06b6d4",
}

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@400;500;600;700;800;900&display=swap');
html, body, [class*="css"] {font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;}
.stApp {background:#f5f7fb;color:#0f172a;}
.block-container {max-width: 1500px; padding-top: 1rem; padding-bottom: 2rem;}
#MainMenu, footer, header {visibility:hidden;}
[data-testid="stSidebar"] {background:#0f172a;border-right:1px solid rgba(255,255,255,.08);}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,[data-testid="stSidebar"] label,[data-testid="stSidebar"] span {color:#f8fafc !important;}
[data-testid="stSidebar"] [data-baseweb="input"],
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] input {background:#ffffff !important;color:#0f172a !important;-webkit-text-fill-color:#0f172a !important;border-radius:10px !important;}
[data-testid="stSidebar"] [data-baseweb="select"] * {color:#0f172a !important;}
.stButton>button {border:0;border-radius:12px;background:#2563eb;color:white;font-weight:800;padding:.6rem 1rem;}
.stButton>button:hover {background:#1d4ed8;color:white;}
.card {background:#ffffff;border:1px solid #d8e0ea;border-radius:18px;padding:16px 18px;box-shadow:0 10px 24px rgba(15,23,42,.06);margin-bottom:12px;}
.small {font-size:12px;color:#64748b;}
.title {font-size:26px;font-weight:900;letter-spacing:-.04em;margin:0;color:#0f172a;}
.sub {font-size:14px;color:#64748b;margin-top:4px;}
.price {font-size:30px;font-weight:900;letter-spacing:-.04em;color:#0f172a;}
.up {color:#10b981;font-weight:900;}
.down {color:#ef4444;font-weight:900;}
.badge {display:inline-block;border-radius:999px;padding:6px 11px;font-size:12px;font-weight:900;}
.badge-buy {background:#dcfce7;color:#047857;border:1px solid #86efac;}
.badge-watch {background:#eff6ff;color:#1d4ed8;border:1px solid #93c5fd;}
.badge-wait {background:#f1f5f9;color:#475569;border:1px solid #cbd5e1;}
.badge-avoid {background:#fee2e2;color:#b91c1c;border:1px solid #fca5a5;}
.kpi-grid {display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;margin:10px 0 14px;}
.kpi {background:#fff;border:1px solid #d8e0ea;border-radius:16px;padding:13px 14px;box-shadow:0 8px 20px rgba(15,23,42,.05);}
.kpi-label {font-size:12px;color:#64748b;font-weight:800;}
.kpi-value {font-size:21px;color:#0f172a;font-weight:900;margin-top:2px;}
.kpi-unit {font-size:11px;color:#64748b;margin-top:3px;font-weight:700;}
.scan-status {background:#ffffff;border:1px solid #d8e0ea;border-radius:14px;padding:12px 14px;margin:10px 0;color:#0f172a;}
.scan-status b {color:#2563eb;}
.progress-label {display:flex;justify-content:space-between;font-size:12px;font-weight:800;margin:8px 0 4px;color:#334155;}
.progress-track {height:9px;background:#e2e8f0;border-radius:999px;overflow:hidden;}
.progress-bar {height:9px;border-radius:999px;background:linear-gradient(90deg,#2563eb,#10b981);}
.stPlotlyChart {background:#fff;border:1px solid #d8e0ea;border-radius:18px;padding:8px;box-shadow:0 10px 24px rgba(15,23,42,.06);}
[data-testid="stMetric"] {background:#fff;border:1px solid #d8e0ea;border-radius:16px;padding:12px;}
[data-testid="stDataFrame"] {border:1px solid #d8e0ea;border-radius:16px;overflow:hidden;}
hr {border-color:#d8e0ea;}
@media(max-width:1100px){.kpi-grid{grid-template-columns:repeat(2,minmax(0,1fr));}.price{font-size:24px;}}

/* 사이드바 입력값 단위 가독성 보강 */
[data-testid="stSidebar"] [data-testid="stNumberInput"] input,
[data-testid="stSidebar"] [data-testid="stSelectbox"] div,
[data-testid="stSidebar"] [data-testid="stSlider"] div {font-variant-numeric: tabular-nums;}
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] .stCaption * {color:#cbd5e1 !important;font-size:12px !important;}
[data-testid="stSidebar"] [data-testid="stNumberInput"] button {background:#ffffff !important;border-color:#e2e8f0 !important;}
[data-testid="stSidebar"] [data-testid="stNumberInput"] button svg {fill:#0f172a !important;}
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div {background:#ffffff !important;}
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] * {color:#0f172a !important;-webkit-text-fill-color:#0f172a !important;}


.live-row {display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:8px 0 10px;}
.live-pill {display:inline-flex;align-items:center;gap:6px;border-radius:999px;padding:7px 11px;font-size:12px;font-weight:900;border:1px solid #334155;background:#111827;color:#e5e7eb;}
.live-open {background:rgba(16,185,129,.12);color:#34d399;border-color:rgba(52,211,153,.35);}
.live-closed {background:rgba(148,163,184,.13);color:#cbd5e1;border-color:rgba(203,213,225,.25);}
.live-warn {background:rgba(245,158,11,.12);color:#fbbf24;border-color:rgba(251,191,36,.35);}
.chart-guide {background:#0f172a;border:1px solid #263244;color:#cbd5e1;border-radius:14px;padding:10px 12px;margin-bottom:10px;font-size:12.5px;line-height:1.55;}
.chart-guide b {color:#f8fafc;}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 표시 안정성 보정: 원화 표시 시 숫자 잘림 방지
# ============================================================
st.markdown(
    """
<style>
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] > div,
[data-testid="stMetricValue"] [data-testid="stMarkdownContainer"],
[data-testid="stMetricValue"] [data-testid="stMarkdownContainer"] p {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    max-width: none !important;
    line-height: 1.18 !important;
    font-size: clamp(18px, 2.0vw, 28px) !important;
    letter-spacing: -0.035em !important;
    word-break: keep-all !important;
    overflow-wrap: anywhere !important;
}
[data-testid="stMetricDelta"],
[data-testid="stMetricDelta"] * {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
}
.order-grid {display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin-top:10px;}
.order-item {background:#ffffff;border:1px solid #d8e0ea;border-radius:16px;padding:12px 13px;min-width:0;overflow:visible;box-shadow:0 8px 18px rgba(15,23,42,.04);}
.order-label {color:#475569;font-size:13px;font-weight:850;margin-bottom:6px;}
.order-value {color:#0f172a;font-size:clamp(20px, 2.25vw, 30px);font-weight:950;letter-spacing:-0.055em;line-height:1.12;white-space:normal;word-break:keep-all;overflow-wrap:anywhere;}
.order-value.small-value {font-size:clamp(18px, 1.75vw, 25px);}
.order-sub {margin-top:5px;color:#64748b;font-size:12px;line-height:1.35;word-break:keep-all;}
.order-down { color:#dc2626 !important; }
.order-up { color:#059669 !important; }
.order-note {margin-top:10px;color:#64748b;font-size:12.5px;line-height:1.55;word-break:keep-all;}
@media(max-width:900px){.order-grid{grid-template-columns:1fr;}.order-value{font-size:24px;}}
</style>
""",
    unsafe_allow_html=True,
)


st.markdown(
    """
<style>
/* 트레이딩 앱 스타일 차트 보정 */
.stPlotlyChart {
    background:#0b1120 !important;
    border:1px solid #1e293b !important;
    border-radius:18px !important;
    padding:6px !important;
    box-shadow:0 18px 38px rgba(2,6,23,.18) !important;
}
.chart-guide {
    background:#0b1120;
    color:#cbd5e1;
    border:1px solid #1e293b;
    border-radius:14px;
    padding:10px 12px;
    margin-bottom:8px;
    font-size:12px;
    line-height:1.55;
}
.chart-guide b {color:#ffffff;}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# 종목 데이터
# ============================================================

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
    "메타": "META", "meta": "META",
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
    "1분봉": ("1m", "7d"),
    "일봉": ("1d", "5y"),
    "1시간봉": ("1h", "730d"),
    "30분봉": ("30m", "60d"),
    "15분봉": ("15m", "60d"),
}

REALTIME_TIMEFRAMES = {"1분봉", "15분봉", "30분봉", "1시간봉"}


def safe_now(tz_name: str) -> datetime:
    """timezone DB가 있는 환경에서는 해당 시장 시간, 실패하면 서버 시간을 사용한다."""
    try:
        return datetime.now(ZoneInfo(tz_name))
    except Exception:
        return datetime.now()


def market_clock(symbol: str) -> Dict[str, object]:
    """
    장중 자동 갱신 여부 판단용 간단 시장 시계.
    주말과 정규장 시간만 반영한다. 공휴일과 조기폐장은 거래소 휴장 캘린더 API가 없으므로 반영하지 않는다.
    """
    if symbol.endswith("-USD"):
        now = safe_now("UTC")
        return {
            "open": True,
            "market": "코인",
            "timezone": "UTC",
            "now": now,
            "label": "코인 시장 24시간",
            "detail": f"UTC {now.strftime('%Y-%m-%d %H:%M')}",
        }

    if is_korean_ticker(symbol):
        now = safe_now("Asia/Seoul")
        open_time = dt_time(9, 0)
        close_time = dt_time(15, 30)
        is_open = now.weekday() < 5 and open_time <= now.time() < close_time
        return {
            "open": bool(is_open),
            "market": "국장",
            "timezone": "Asia/Seoul",
            "now": now,
            "label": "국장 정규장" if is_open else "국장 장마감",
            "detail": f"한국시간 {now.strftime('%Y-%m-%d %H:%M')} · 정규장 09:00~15:30",
        }

    now = safe_now("America/New_York")
    open_time = dt_time(9, 30)
    close_time = dt_time(16, 0)
    is_open = now.weekday() < 5 and open_time <= now.time() < close_time
    return {
        "open": bool(is_open),
        "market": "미장",
        "timezone": "America/New_York",
        "now": now,
        "label": "미장 정규장" if is_open else "미장 장마감",
        "detail": f"뉴욕시간 {now.strftime('%Y-%m-%d %H:%M')} · 정규장 09:30~16:00",
    }


def render_live_status(symbol: str, timeframe_label: str) -> Dict[str, object]:
    clock = market_clock(symbol)
    realtime_tf = timeframe_label in REALTIME_TIMEFRAMES
    can_refresh = bool(clock.get("open")) and realtime_tf and bool(globals().get("auto_refresh_enabled", True))
    refresh_sec = int(globals().get("auto_refresh_seconds", 60))

    if can_refresh:
        # Streamlit 기본 기능만 사용하기 위해 meta refresh로 전체 앱을 주기적으로 새로고침한다.
        st.markdown(f"<meta http-equiv='refresh' content='{refresh_sec}'>", unsafe_allow_html=True)
        status_cls = "live-open"
        status_text = f"실시간 갱신 ON · {refresh_sec}초마다 새로고침"
    elif not realtime_tf:
        status_cls = "live-warn"
        status_text = "실시간 갱신 대기 · 분봉 선택 시 장중 자동 갱신"
    elif not bool(clock.get("open")):
        status_cls = "live-closed"
        status_text = "장마감 · 자동 새로고침 중지"
    else:
        status_cls = "live-closed"
        status_text = "자동 새로고침 OFF"

    st.markdown(
        f"""
        <div class="live-row">
          <span class="live-pill {status_cls}">{status_text}</span>
          <span class="live-pill">{clock.get('label')}</span>
          <span class="live-pill">{clock.get('detail')}</span>
          <span class="live-pill">데이터 갱신 기준: Yahoo Finance</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return clock

# ============================================================
# 설정 데이터 클래스
# ============================================================

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
    stop_pct: float
    take_profit_pct: float
    trailing_stop_pct: float
    ma_exit: str
    use_time_stop: bool
    max_hold_bars: int

@dataclass
class StrategyConfig:
    style: str
    horizon: int
    lookback: int
    train_window: int
    min_prob: float
    min_expected: float
    min_score: float
    risk_penalty: float
    market_filter: bool
    use_arima: bool
    use_gb: bool
    use_rf: bool

# ============================================================
# 유틸리티
# ============================================================

def is_korean_ticker(symbol: str) -> bool:
    return symbol.endswith(".KS") or symbol.endswith(".KQ")


def safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None or pd.isna(x) or np.isinf(x):
            return default
        return float(x)
    except Exception:
        return default


def sigmoid(x: float) -> float:
    return float(1 / (1 + np.exp(-np.clip(x, -40, 40))))


def fmt_pct(x: float) -> str:
    return f"{x*100:+.2f}%"


def parse_number_text(value: str, default: float = 0.0) -> float:
    """
    사이드바 입력칸에 10,000.00 $ 처럼 단위가 붙어 있어도 숫자만 안전하게 추출한다.
    """
    try:
        text = str(value).replace(",", "").strip()
        cleaned = "".join(ch for ch in text if ch.isdigit() or ch in [".", "-"])
        if cleaned in ["", ".", "-", "-."]:
            return default
        num = float(cleaned)
        if not np.isfinite(num):
            return default
        return num
    except Exception:
        return default


def fmt_curr(v: float) -> str:
    if v is None or pd.isna(v):
        return "N/A"
    sign = "-" if float(v) < 0 else ""
    v = abs(float(v))
    if st.session_state.get("currency", "USD") == "KRW":
        return f"{sign}₩{v * EXCHANGE_RATE:,.0f}"
    return f"{sign}${v:,.2f}"


def money_unit_label() -> str:
    return "원" if st.session_state.get("currency", "USD") == "KRW" else "달러"


def money_tick_settings() -> Dict[str, object]:
    if st.session_state.get("currency", "USD") == "KRW":
        return {"tickprefix": "₩", "tickformat": ",.0f"}
    return {"tickprefix": "$", "tickformat": ",.2f"}


def fmt_compact_number(v: float) -> str:
    try:
        v = float(v)
    except Exception:
        return "N/A"
    sign = "-" if v < 0 else ""
    v = abs(v)
    if v >= 100000000:
        return f"{sign}{v/100000000:.2f}억"
    if v >= 10000:
        return f"{sign}{v/10000:.2f}만"
    return f"{sign}{v:,.0f}"


def fmt_money_compact_usd(v: float) -> str:
    if v is None or pd.isna(v):
        return "N/A"
    val = float(v)
    if st.session_state.get("currency", "USD") == "KRW":
        val *= EXCHANGE_RATE
        prefix = "₩"
    else:
        prefix = "$"
    sign = "-" if val < 0 else ""
    val = abs(val)
    if st.session_state.get("currency", "USD") == "KRW":
        if val >= 100000000:
            return f"{sign}{prefix}{val/100000000:.2f}억"
        if val >= 10000:
            return f"{sign}{prefix}{val/10000:.2f}만"
        return f"{sign}{prefix}{val:,.0f}"
    if val >= 1000000000:
        return f"{sign}{prefix}{val/1000000000:.2f}B"
    if val >= 1000000:
        return f"{sign}{prefix}{val/1000000:.2f}M"
    if val >= 1000:
        return f"{sign}{prefix}{val/1000:.2f}K"
    return f"{sign}{prefix}{val:,.2f}"


def search_ticker_by_name(query: str, is_coin: bool = False) -> Optional[str]:
    query = str(query).strip()
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
    try:
        r = requests.get(url, params={"q": query, "quotesCount": 5, "newsCount": 0}, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if r.status_code == 200:
            quotes = r.json().get("quotes", [])
            if quotes:
                return str(quotes[0].get("symbol", query)).upper()
    except Exception:
        pass
    return query.upper()

# ============================================================
# 데이터 수집 및 지표
# ============================================================

@st.cache_data(ttl=600, show_spinner=False)
def get_realtime_exchange_rate() -> float:
    url = "https://query1.finance.yahoo.com/v8/finance/chart/USDKRW=X?range=1d&interval=1m"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if r.status_code == 200:
            return float(r.json()["chart"]["result"][0]["meta"]["regularMarketPrice"])
    except Exception:
        pass
    return 1380.0

EXCHANGE_RATE = get_realtime_exchange_rate()

@st.cache_data(ttl=45, show_spinner=False)
def fetch_ohlcv(symbol: str, interval: str, data_range: str) -> Tuple[Optional[pd.DataFrame], str]:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={data_range}&interval={interval}"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=14)
        if r.status_code != 200:
            return None, symbol
        raw = r.json()
        result = raw.get("chart", {}).get("result")
        if not result:
            return None, symbol
        result = result[0]
        timestamps = result.get("timestamp")
        if not timestamps:
            return None, symbol
        q = result.get("indicators", {}).get("quote", [{}])[0]
    except Exception:
        return None, symbol

    date_format = "%Y-%m-%d %H:%M" if interval != "1d" else "%Y-%m-%d"
    try:
        df = pd.DataFrame({
            "date": [datetime.fromtimestamp(ts).strftime(date_format) for ts in timestamps],
            "open": q.get("open"),
            "high": q.get("high"),
            "low": q.get("low"),
            "close": q.get("close"),
            "volume": q.get("volume"),
        })
        df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
        df["volume"] = df["volume"].fillna(0)
        df = df[df["close"] > 0].reset_index(drop=True)
        if df.empty:
            return None, symbol
    except Exception:
        return None, symbol

    meta = result.get("meta", {})
    name = TICKER_NAME_MAP.get(symbol) or meta.get("shortName") or symbol
    api_currency = str(meta.get("currency", "USD")).upper()
    if is_korean_ticker(symbol) or api_currency == "KRW":
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col] / EXCHANGE_RATE
    return df, str(name)


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ret = df["close"].pct_change()

    for w in [5, 10, 20, 50, 60, 120, 200]:
        df[f"ma{w}"] = df["close"].rolling(w).mean()
        df[f"ret{w}"] = df["close"].pct_change(w)
        df[f"volatility{w}"] = ret.rolling(w).std()

    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df.loc[loss == 0, "rsi"] = 100

    # ATR
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr14"] / df["close"].replace(0, np.nan)

    # 거래량
    df["volMa20"] = df["volume"].rolling(20).mean()
    df["volRatio"] = df["volume"] / df["volMa20"].replace(0, np.nan)
    df["dollar_volume"] = df["close"] * df["volume"]

    # MACD
    df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema12"] - df["ema26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger Band
    df["bb_mid"] = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std

    # VWAP: 일 단위 리셋은 하지 않고 분석 범위 누적 VWAP로 계산
    typical = (df["high"] + df["low"] + df["close"]) / 3
    vol_cum = df["volume"].replace(0, np.nan).cumsum()
    df["vwap"] = (typical * df["volume"]).cumsum() / vol_cum

    # 지지/저항
    df["support60"] = df["low"].rolling(60).min()
    df["resistance60"] = df["high"].rolling(60).max()

    df["range_pct"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["body_pct"] = (df["close"] - df["open"]) / df["open"].replace(0, np.nan)
    return df

@st.cache_data(ttl=45, show_spinner=False)
def get_market_regime(symbol: str, interval: str, data_range: str) -> Dict:
    if symbol.endswith("-USD"):
        proxy = "BTC-USD"
        name = "코인 시장"
    elif is_korean_ticker(symbol):
        proxy = "^KS11"
        name = "국내 시장"
    else:
        proxy = "^GSPC"
        name = "미국 시장"

    df, _ = fetch_ohlcv(proxy, interval, data_range)
    if df is None or len(df) < 130:
        return {"proxy": proxy, "name": name, "score": 0.5, "ok": True, "text": "시장 데이터 부족"}
    df = calculate_indicators(df)
    i = len(df) - 1
    score = 0.0
    close = df.loc[i, "close"]
    if pd.notna(df.loc[i, "ma120"]) and close > df.loc[i, "ma120"]:
        score += 0.40
    if pd.notna(df.loc[i, "ma20"]) and close > df.loc[i, "ma20"]:
        score += 0.25
    if pd.notna(df.loc[i, "ma20"]) and pd.notna(df.loc[max(0, i - 20), "ma20"]) and df.loc[i, "ma20"] > df.loc[max(0, i - 20), "ma20"]:
        score += 0.25
    if pd.notna(df.loc[i, "rsi"]) and df.loc[i, "rsi"] > 45:
        score += 0.10
    ok = score >= 0.55
    return {"proxy": proxy, "name": name, "score": float(score), "ok": ok, "text": f"{name} {'양호' if ok else '주의'} {score*100:.0f}점"}

# ============================================================
# 수학적 모델 및 예측
# ============================================================

def risk_score_at(df: pd.DataFrame, idx: int) -> float:
    rsi = safe_float(df.loc[idx, "rsi"], 50)
    atr_pct = safe_float(df.loc[idx, "atr_pct"], 0.03)
    vol20 = safe_float(df.loc[idx, "volatility20"], 0.02)
    vol60 = safe_float(df.loc[idx, "volatility60"], vol20 if vol20 > 0 else 0.02)
    close = safe_float(df.loc[idx, "close"], 0)
    ma20 = safe_float(df.loc[idx, "ma20"], close)
    ma60 = safe_float(df.loc[idx, "ma60"], close)
    ma120 = safe_float(df.loc[idx, "ma120"], close)

    risk = 0.0
    if rsi > 75:
        risk += min((rsi - 75) / 25, 1) * 0.22
    if atr_pct > 0.03:
        risk += min((atr_pct - 0.03) / 0.10, 1) * 0.25
    if vol60 > 0 and vol20 / vol60 > 1.25:
        risk += min((vol20 / vol60 - 1.25) / 1.75, 1) * 0.20
    if close < ma120:
        risk += 0.15
    if ma20 < ma60 < ma120:
        risk += 0.18
    return float(np.clip(risk, 0, 1))


def technical_score_at(df: pd.DataFrame, idx: int, market_score: float) -> Tuple[float, Dict[str, float]]:
    close = safe_float(df.loc[idx, "close"], 0)
    ma20 = safe_float(df.loc[idx, "ma20"], np.nan)
    ma60 = safe_float(df.loc[idx, "ma60"], np.nan)
    ma120 = safe_float(df.loc[idx, "ma120"], np.nan)
    rsi = safe_float(df.loc[idx, "rsi"], 50)
    vol_ratio = safe_float(df.loc[idx, "volRatio"], 1)
    atr_pct = safe_float(df.loc[idx, "atr_pct"], 0.03)
    ret5 = safe_float(df.loc[idx, "ret5"], 0)
    ret20 = safe_float(df.loc[idx, "ret20"], 0)

    trend = 0.35
    if close > ma20:
        trend += 0.20
    if ma20 > ma60:
        trend += 0.20
    if ma60 > ma120:
        trend += 0.20
    if close > ma120:
        trend += 0.05
    trend = float(np.clip(trend, 0, 1))

    momentum = float(np.clip(0.50 + np.tanh(ret5 * 12) * 0.22 + np.tanh(ret20 * 6) * 0.22, 0, 1))

    if 42 <= rsi <= 62:
        rsi_score = 0.85
    elif 35 <= rsi < 42:
        rsi_score = 0.65
    elif 62 < rsi <= 72:
        rsi_score = 0.55
    elif rsi > 72:
        rsi_score = 0.30
    else:
        rsi_score = 0.40

    volume_score = float(np.clip(0.35 + min(max(vol_ratio, 0), 2.5) / 2.5 * 0.55, 0, 1))

    pullback = 0.50
    if ma20 and np.isfinite(ma20) and ma20 > 0:
        dist = close / ma20 - 1
        pullback = 1.0 - min(abs(dist) / 0.12, 1) * 0.70
        if -0.05 <= dist <= 0.04:
            pullback += 0.12
        pullback = float(np.clip(pullback, 0, 1))

    stability = float(np.clip(1.0 - min(max((atr_pct - 0.015) / 0.10, 0), 1) * 0.75, 0, 1))
    risk_inv = 1 - risk_score_at(df, idx)
    market = float(np.clip(market_score, 0, 1))

    # 너무 어려운 가중치는 숨기고, 내부에서는 균형형으로 사용
    comps = {
        "추세": trend,
        "모멘텀": momentum,
        "RSI": rsi_score,
        "거래량": volume_score,
        "눌림목": pullback,
        "안정성": stability,
        "시장": market,
        "저위험": risk_inv,
    }
    weights = {
        "추세": 1.25,
        "모멘텀": 1.05,
        "RSI": 0.85,
        "거래량": 0.75,
        "눌림목": 1.10,
        "안정성": 1.10,
        "시장": 1.00,
        "저위험": 1.20,
    }
    score = sum(comps[k] * weights[k] for k in comps) / sum(weights.values())
    return float(np.clip(score, 0, 1)), comps


def compute_feature_at(df: pd.DataFrame, idx: int, lookback: int, market_score: float) -> Optional[np.ndarray]:
    if idx < max(lookback, 220):
        return None
    close = safe_float(df.loc[idx, "close"], 0)
    if close <= 0:
        return None
    window = df.iloc[idx - lookback + 1 : idx + 1]
    returns = window["close"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    feats: List[float] = []

    for w in [1, 2, 3, 5, 10, 20, 60, 120]:
        if idx - w >= 0 and df.loc[idx - w, "close"] != 0:
            feats.append(close / df.loc[idx - w, "close"] - 1)
        else:
            feats.append(0.0)

    for w in [5, 10, 20, min(lookback, 60), min(lookback, 120)]:
        s = returns.tail(w)
        feats.append(safe_float(s.mean(), 0))
        feats.append(safe_float(s.std(), 0))
        feats.append(safe_float((s > 0).mean(), 0.5))

    for w in [5, 20, 60, 120, 200]:
        ma = safe_float(df.loc[idx, f"ma{w}"], np.nan)
        prev_ma = safe_float(df.loc[max(0, idx - min(w, 20)), f"ma{w}"], np.nan)
        feats.append(close / ma - 1 if ma and np.isfinite(ma) and ma != 0 else 0)
        feats.append(ma / prev_ma - 1 if ma and prev_ma and np.isfinite(ma) and np.isfinite(prev_ma) and prev_ma != 0 else 0)

    feats.extend([
        (safe_float(df.loc[idx, "rsi"], 50) - 50) / 50,
        safe_float(df.loc[idx, "volRatio"], 1) - 1,
        safe_float(df.loc[idx, "atr_pct"], 0.03),
        safe_float(df.loc[idx, "macd_hist"], 0) / close,
        safe_float(df.loc[idx, "range_pct"], 0),
        safe_float(df.loc[idx, "body_pct"], 0),
    ])

    high = window["high"].max()
    low = window["low"].min()
    if high > low:
        feats.append((close - low) / (high - low))
        feats.append((high - low) / close)
    else:
        feats.extend([0.5, 0.0])

    tech, comps = technical_score_at(df, idx, market_score)
    feats.append(tech)
    feats.append(risk_score_at(df, idx))
    feats.append(market_score)
    arr = np.array(feats, dtype=float)
    return np.nan_to_num(arr, nan=0, posinf=0, neginf=0)


def build_dataset(df: pd.DataFrame, lookback: int, horizon: int, market_score: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    X, y, idxs = [], [], []
    start = max(lookback, 220)
    end = len(df) - horizon
    for idx in range(start, end):
        feat = compute_feature_at(df, idx, lookback, market_score)
        if feat is None:
            continue
        future_ret = df.loc[idx + horizon, "close"] / df.loc[idx, "close"] - 1
        if np.isfinite(future_ret):
            X.append(feat)
            y.append(future_ret)
            idxs.append(idx)
    if len(X) < 80:
        return None, None, None
    return np.vstack(X), np.array(y, dtype=float), np.array(idxs, dtype=int)


def numpy_ridge_predict(X_train: np.ndarray, y_train: np.ndarray, x: np.ndarray, alpha: float = 3.0) -> float:
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    Xs = (X_train - mean) / std
    xs = (x - mean) / std
    X_aug = np.column_stack([np.ones(len(Xs)), Xs])
    x_aug = np.concatenate([[1.0], xs])
    reg = np.eye(X_aug.shape[1]) * alpha
    reg[0, 0] = 0
    try:
        beta = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ y_train)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(X_aug.T @ X_aug + reg) @ X_aug.T @ y_train
    return float(x_aug @ beta)


def arima_forecast_return(df: pd.DataFrame, horizon: int, max_points: int = 360) -> Optional[float]:
    if not STATSMODELS_AVAILABLE:
        return None
    try:
        close = df["close"].dropna().astype(float).tail(max_points)
        if len(close) < 120:
            return None
        logp = np.log(close)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(logp, order=(1, 1, 1))
            fitted = model.fit()
            fc = fitted.forecast(steps=horizon)
        pred_log = float(fc.iloc[-1])
        last_log = float(logp.iloc[-1])
        return float(np.exp(pred_log - last_log) - 1)
    except Exception:
        return None


def fit_predict_models(X_train: np.ndarray, y_train: np.ndarray, x: np.ndarray, cfg: StrategyConfig, df_for_arima: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    mask = np.isfinite(y_train)
    X_train = X_train[mask]
    y_train = y_train[mask]
    if len(y_train) < 60:
        return {"pred_return": 0.0, "prob_up": 0.5, "model_count": 0}

    scale = float(np.nanstd(y_train))
    if not np.isfinite(scale) or scale < 1e-5:
        scale = 0.02

    reg_preds: List[float] = []
    prob_preds: List[float] = []

    # 1. Ridge 회귀: 기본 수학 모델
    if SKLEARN_AVAILABLE:
        try:
            ridge = make_pipeline(StandardScaler(), Ridge(alpha=2.5))
            ridge.fit(X_train, y_train)
            p = float(ridge.predict(x.reshape(1, -1))[0])
            reg_preds.append(p)
            prob_preds.append(sigmoid(p / scale))
        except Exception:
            p = numpy_ridge_predict(X_train, y_train, x)
            reg_preds.append(p)
            prob_preds.append(sigmoid(p / scale))
    else:
        p = numpy_ridge_predict(X_train, y_train, x)
        reg_preds.append(p)
        prob_preds.append(sigmoid(p / scale))

    # 2. Logistic 회귀: 상승/하락 확률
    if SKLEARN_AVAILABLE:
        try:
            y_cls = (y_train > 0).astype(int)
            if len(np.unique(y_cls)) >= 2:
                logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=600, class_weight="balanced"))
                logit.fit(X_train, y_cls)
                prob_preds.append(float(logit.predict_proba(x.reshape(1, -1))[0][1]))
        except Exception:
            pass

    # 3. Gradient Boosting: 비선형 수익률 모델
    if cfg.use_gb and SKLEARN_AVAILABLE:
        try:
            gb = GradientBoostingRegressor(n_estimators=90, learning_rate=0.035, max_depth=2, subsample=0.85, random_state=42)
            gb.fit(X_train, y_train)
            p = float(gb.predict(x.reshape(1, -1))[0])
            reg_preds.append(p)
            prob_preds.append(sigmoid(p / scale))
        except Exception:
            pass

    # 4. Random Forest: 비선형 상승확률 모델
    if cfg.use_rf and SKLEARN_AVAILABLE:
        try:
            y_cls = (y_train > 0).astype(int)
            if len(np.unique(y_cls)) >= 2:
                rf = RandomForestClassifier(n_estimators=120, max_depth=5, min_samples_leaf=8, class_weight="balanced_subsample", random_state=42)
                rf.fit(X_train, y_cls)
                prob_preds.append(float(rf.predict_proba(x.reshape(1, -1))[0][1]))
        except Exception:
            pass

    # 5. ARIMA: 가격 자체의 시계열 예측. 옵션.
    if cfg.use_arima and df_for_arima is not None:
        ar = arima_forecast_return(df_for_arima, cfg.horizon)
        if ar is not None and np.isfinite(ar):
            # ARIMA는 노이즈가 커서 비중을 낮게 넣는다.
            reg_preds.append(float(ar) * 0.55)
            prob_preds.append(sigmoid(float(ar) / scale))

    pred_return = float(np.mean(reg_preds)) if reg_preds else 0.0
    prob_up = float(np.clip(np.mean(prob_preds), 0.01, 0.99)) if prob_preds else sigmoid(pred_return / scale)
    return {"pred_return": pred_return, "prob_up": prob_up, "model_count": len(reg_preds) + len(prob_preds)}


def grade_signal(prob: float, expected: float, final_score: float, risk: float, market_ok: bool, cfg: StrategyConfig) -> Tuple[str, str]:
    if cfg.market_filter and not market_ok:
        return "회피", "시장 흐름 약함"
    if risk >= 0.78:
        return "회피", "리스크 과다"
    if prob >= cfg.min_prob + 0.08 and expected >= cfg.min_expected + 0.012 and final_score >= cfg.min_score + 0.08 and risk <= 0.45:
        return "강한 매수", "조건 우수"
    if prob >= cfg.min_prob and expected >= cfg.min_expected and final_score >= cfg.min_score and risk <= 0.65:
        return "매수 후보", "조건 통과"
    if prob >= max(0.52, cfg.min_prob - 0.04) and final_score >= max(0.52, cfg.min_score - 0.06):
        return "관심", "추가 확인"
    return "대기", "조건 부족"


def latest_prediction(df: pd.DataFrame, cfg: StrategyConfig, cost: CostConfig, market: Dict) -> Optional[Dict]:
    X, y, idxs = build_dataset(df, cfg.lookback, cfg.horizon, market["score"])
    if X is None:
        return None
    idx = len(df) - 1
    x = compute_feature_at(df, idx, cfg.lookback, market["score"])
    if x is None:
        return None
    X_train = X[-cfg.train_window:]
    y_train = y[-cfg.train_window:]
    model = fit_predict_models(X_train, y_train, x, cfg, df_for_arima=df)
    risk = risk_score_at(df, idx)
    tech, comps = technical_score_at(df, idx, market["score"])
    pred = model["pred_return"] - cfg.risk_penalty * risk * max(float(np.nanstd(y_train)), 0.01)
    round_cost = cost.fee_pct * 2 + cost.slippage_pct * 2 + cost.sell_tax_pct
    expected = pred - round_cost
    prob = float(np.clip((model["prob_up"] * 0.68) + (sigmoid(expected / max(float(np.nanstd(y_train)), 0.01)) * 0.32), 0.01, 0.99))
    ml_score = float(np.clip(prob * 0.65 + sigmoid(expected / max(float(np.nanstd(y_train)), 0.01)) * 0.35, 0, 1))
    final_score = float(np.clip(ml_score * 0.65 + tech * 0.35, 0, 1))
    grade, reason = grade_signal(prob, expected, final_score, risk, market["ok"], cfg)
    return {
        "idx": idx,
        "date": df.loc[idx, "date"],
        "pred_return": pred,
        "expected": expected,
        "prob": prob,
        "risk": risk,
        "tech": tech,
        "ml_score": ml_score,
        "final_score": final_score,
        "grade": grade,
        "reason": reason,
        "components": comps,
        "model_count": model["model_count"],
    }


def walk_forward_predictions(df: pd.DataFrame, cfg: StrategyConfig, cost: CostConfig, market: Dict, step: int) -> pd.DataFrame:
    X, y, idxs = build_dataset(df, cfg.lookback, cfg.horizon, market["score"])
    if X is None:
        return pd.DataFrame()
    rows = []
    round_cost = cost.fee_pct * 2 + cost.slippage_pct * 2 + cost.sell_tax_pct
    start_i = max(70, min(cfg.train_window, len(idxs) - 1))
    for row_i in range(start_i, len(idxs), max(1, step)):
        idx = int(idxs[row_i])
        train_start = max(0, row_i - cfg.train_window)
        X_train = X[train_start:row_i]
        y_train = y[train_start:row_i]
        if len(y_train) < 60:
            continue
        model = fit_predict_models(X_train, y_train, X[row_i], cfg, df_for_arima=None)
        risk = risk_score_at(df, idx)
        scale = max(float(np.nanstd(y_train)), 0.01)
        pred = model["pred_return"] - cfg.risk_penalty * risk * scale
        expected = pred - round_cost
        prob = float(np.clip((model["prob_up"] * 0.68) + (sigmoid(expected / scale) * 0.32), 0.01, 0.99))
        tech, _ = technical_score_at(df, idx, market["score"])
        ml_score = float(np.clip(prob * 0.65 + sigmoid(expected / scale) * 0.35, 0, 1))
        final_score = float(np.clip(ml_score * 0.65 + tech * 0.35, 0, 1))
        rows.append({
            "idx": idx,
            "date": df.loc[idx, "date"],
            "pred_return": pred,
            "expected": expected,
            "prob": prob,
            "risk": risk,
            "tech": tech,
            "final_score": final_score,
            "actual": y[row_i],
        })
    return pd.DataFrame(rows)

# ============================================================
# 백테스트 및 리스크 관리
# ============================================================

def calc_position_size(capital: float, entry: float, stop: float, risk: RiskConfig) -> Tuple[float, float]:
    if entry <= 0 or stop <= 0 or stop >= entry:
        return 0.0, 0.0
    risk_cash = capital * risk.risk_per_trade
    qty_by_risk = risk_cash / (entry - stop)
    qty_by_cap = (capital * risk.max_position_pct) / entry
    qty = max(0.0, min(qty_by_risk, qty_by_cap))
    return qty, qty * entry


def run_backtest(df: pd.DataFrame, pred_df: pd.DataFrame, cfg: StrategyConfig, cost: CostConfig, risk_cfg: RiskConfig, market_ok: bool) -> Tuple[List[Dict], pd.DataFrame, List[Dict], List[Dict]]:
    if pred_df.empty:
        return [], pd.DataFrame(columns=["idx", "date", "equity"]), [], []
    signals = {int(r["idx"]): r for _, r in pred_df.iterrows()}
    cash = float(risk_cfg.start_capital)
    position = None
    trades, buys, sells, equity_rows = [], [], [], []

    for i in range(len(df)):
        close = float(df.loc[i, "close"])
        high = float(df.loc[i, "high"])
        low = float(df.loc[i, "low"])
        date = df.loc[i, "date"]

        if position is not None:
            position["max_high"] = max(position["max_high"], high)
            stop_line = position["stop"]
            if risk_cfg.trailing_stop_pct > 0:
                stop_line = max(stop_line, position["max_high"] * (1 - risk_cfg.trailing_stop_pct))
            exit_reason = None
            exit_ref = close

            if low <= stop_line:
                exit_reason = "손절/추적손절"
                exit_ref = stop_line
            elif high >= position["target"]:
                exit_reason = "익절"
                exit_ref = position["target"]
            elif risk_cfg.ma_exit != "사용 안 함":
                ma_col = "ma20" if risk_cfg.ma_exit == "MA20 이탈" else "ma60"
                ma_val = df.loc[i, ma_col]
                if pd.notna(ma_val) and close < ma_val:
                    exit_reason = risk_cfg.ma_exit
                    exit_ref = close
            elif risk_cfg.use_time_stop and (i - position["entry_idx"]) >= risk_cfg.max_hold_bars:
                exit_reason = "시간청산"
                exit_ref = close

            if exit_reason:
                exit_price = exit_ref * (1 - cost.slippage_pct)
                gross = position["qty"] * exit_price
                net = gross - gross * cost.fee_pct - gross * cost.sell_tax_pct
                cash += net
                ret = (net - position["total_cost"]) / position["total_cost"] * 100 if position["total_cost"] else 0
                trades.append({
                    "매수일": position["entry_date"], "매수가": position["entry"],
                    "매도일": date, "매도가": exit_price, "수량": position["qty"],
                    "수익률": ret, "청산": exit_reason, "상승확률": position["prob"],
                    "기대수익": position["expected"], "종합점수": position["score"],
                    "청산후자산": cash, "성공": ret > 0,
                })
                sells.append({"idx": i, "price": exit_price, "reason": exit_reason})
                position = None

        if position is None and i in signals:
            row = signals[i]
            market_pass = market_ok or not cfg.market_filter
            signal_ok = (
                market_pass and
                float(row["prob"]) >= cfg.min_prob and
                float(row["expected"]) >= cfg.min_expected and
                float(row["final_score"]) >= cfg.min_score and
                float(row["risk"]) <= 0.76
            )
            if signal_ok and cash > 0:
                entry = close * (1 + cost.slippage_pct)
                stop = entry * (1 - risk_cfg.stop_pct)
                target = entry * (1 + risk_cfg.take_profit_pct)
                qty, used = calc_position_size(cash, entry, stop, risk_cfg)
                if qty > 0 and used > 0:
                    fee = used * cost.fee_pct
                    total_cost = used + fee
                    if total_cost <= cash:
                        cash -= total_cost
                        position = {
                            "entry_idx": i,
                            "entry_date": date,
                            "entry": entry,
                            "qty": qty,
                            "total_cost": total_cost,
                            "stop": stop,
                            "target": target,
                            "max_high": high,
                            "prob": float(row["prob"]),
                            "expected": float(row["expected"]),
                            "score": float(row["final_score"]),
                        }
                        buys.append({"idx": i, "price": entry})

        if position is not None:
            value = position["qty"] * close * (1 - cost.slippage_pct) * (1 - cost.fee_pct - cost.sell_tax_pct)
            equity = cash + value
        else:
            equity = cash
        equity_rows.append({"idx": i, "date": date, "equity": equity})

    return trades, pd.DataFrame(equity_rows), buys, sells


def max_drawdown(equity: pd.Series) -> float:
    if equity is None or len(equity) == 0:
        return 0.0
    arr = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(arr)
    dd = arr / np.where(peak == 0, np.nan, peak) - 1
    return float(np.nanmin(dd) * 100)


def calc_metrics(trades: List[Dict], equity_df: pd.DataFrame, start_capital: float) -> Dict:
    final = float(equity_df["equity"].iloc[-1]) if equity_df is not None and not equity_df.empty else start_capital
    total = (final / start_capital - 1) * 100 if start_capital else 0.0
    n = len(trades)
    wins = [t["수익률"] for t in trades if t["수익률"] > 0]
    losses = [t["수익률"] for t in trades if t["수익률"] <= 0]
    win_rate = len(wins) / n * 100 if n else 0.0
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    profit_factor = sum(wins) / abs(sum(losses)) if losses and abs(sum(losses)) > 0 else (999.0 if wins else 0.0)
    expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)
    mdd = max_drawdown(equity_df["equity"]) if equity_df is not None and not equity_df.empty else 0.0
    sharpe = 0.0
    if equity_df is not None and not equity_df.empty and len(equity_df) > 5:
        r = equity_df["equity"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        if len(r) > 3 and r.std() > 0:
            sharpe = float(r.mean() / r.std() * math.sqrt(252))
    return {
        "총수익률": total,
        "MDD": mdd,
        "승률": win_rate,
        "거래횟수": n,
        "Profit Factor": profit_factor,
        "기대값": expectancy,
        "Sharpe": sharpe,
        "최종자산": final,
    }


def prediction_accuracy(pred_df: pd.DataFrame) -> Tuple[float, int]:
    if pred_df is None or pred_df.empty or "actual" not in pred_df.columns:
        return 0.0, 0
    d = pred_df.dropna(subset=["actual"])
    if d.empty:
        return 0.0, 0
    acc = ((d["pred_return"] > 0) == (d["actual"] > 0)).mean() * 100 if "pred_return" in d else ((d["expected"] > 0) == (d["actual"] > 0)).mean() * 100
    return float(acc), len(d)

# ============================================================
# 차트
# ============================================================

def visible_df(df: pd.DataFrame, bars: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    return df.tail(bars).copy() if len(df) > bars else df.copy()


def make_chart(df: pd.DataFrame, pred_df: Optional[pd.DataFrame], buys: List[Dict], sells: List[Dict], bars: int, show_bb: bool) -> go.Figure:
    """트레이딩 앱처럼 보기 쉬운 다크 테마 차트."""
    dfp = visible_df(df, bars)
    if dfp.empty:
        fig = go.Figure()
        fig.add_annotation(text="데이터 없음", x=0.5, y=0.5, showarrow=False, font=dict(color="#e2e8f0"))
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1120", plot_bgcolor="#0b1120", height=680)
        return fig

    start_idx = int(dfp.index.min())
    end_idx = int(dfp.index.max())
    mult = EXCHANGE_RATE if st.session_state.get("currency") == "KRW" else 1.0
    unit = money_unit_label()

    bg = "#0b1120"
    panel = "#0f172a"
    grid = "rgba(148,163,184,.18)"
    text_color = "#e5e7eb"
    muted = "#94a3b8"
    up = "#22c55e"
    down = "#ef4444"
    blue = "#60a5fa"
    orange = "#f59e0b"
    purple = "#a78bfa"
    cyan = "#22d3ee"

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.018,
        row_heights=[0.62, 0.16, 0.11, 0.11],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]],
    )

    x = dfp.index.tolist()
    dates = dfp["date"].astype(str).tolist()

    fig.add_trace(
        go.Candlestick(
            x=x,
            open=dfp["open"] * mult,
            high=dfp["high"] * mult,
            low=dfp["low"] * mult,
            close=dfp["close"] * mult,
            name="캔들",
            text=dates,
            increasing_line_color=up,
            increasing_fillcolor=up,
            decreasing_line_color=down,
            decreasing_fillcolor=down,
            whiskerwidth=0.45,
            hovertemplate=(
                "%{text}<br>"
                "시가 %{open:,.2f}<br>"
                "고가 %{high:,.2f}<br>"
                "저가 %{low:,.2f}<br>"
                "종가 %{close:,.2f}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    if show_bb and {"bb_upper", "bb_lower"}.issubset(dfp.columns):
        fig.add_trace(go.Scatter(x=x, y=dfp["bb_upper"] * mult, line=dict(color="rgba(167,139,250,.48)", width=1), name="볼린저 상단", hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=dfp["bb_lower"] * mult, fill="tonexty", fillcolor="rgba(167,139,250,.08)", line=dict(color="rgba(167,139,250,.48)", width=1), name="볼린저 하단", hoverinfo="skip"), row=1, col=1)

    ma_specs = [("ma5", "MA5", orange, 1.2), ("ma20", "MA20", cyan, 1.6), ("ma60", "MA60", "#fb923c", 1.35), ("ma120", "MA120", purple, 1.35)]
    for col, name, color, width in ma_specs:
        if col in dfp:
            fig.add_trace(go.Scatter(x=x, y=dfp[col] * mult, line=dict(color=color, width=width), name=name, hovertemplate=f"{name} %{{y:,.2f}}<extra></extra>"), row=1, col=1)

    if "vwap" in dfp:
        fig.add_trace(go.Scatter(x=x, y=dfp["vwap"] * mult, line=dict(color="#e879f9", width=1.15, dash="dot"), name="VWAP", hovertemplate="VWAP %{y:,.2f}<extra></extra>"), row=1, col=1)
    if "support60" in dfp:
        fig.add_trace(go.Scatter(x=x, y=dfp["support60"] * mult, line=dict(color="rgba(34,197,94,.65)", width=1, dash="dash"), name="지지60", hoverinfo="skip"), row=1, col=1)
    if "resistance60" in dfp:
        fig.add_trace(go.Scatter(x=x, y=dfp["resistance60"] * mult, line=dict(color="rgba(239,68,68,.65)", width=1, dash="dash"), name="저항60", hoverinfo="skip"), row=1, col=1)

    last_price = float(dfp["close"].iloc[-1]) * mult
    fig.add_hline(
        y=last_price,
        row=1,
        col=1,
        line_width=1.1,
        line_dash="dot",
        line_color=muted,
        annotation_text=f"현재가 {last_price:,.2f}",
        annotation_position="top right",
        annotation_font=dict(color=text_color, size=11),
    )

    # 백테스트 매수 후보 표시
    if pred_df is not None and not pred_df.empty and {"idx", "prob", "final_score"}.issubset(pred_df.columns):
        cand = pred_df[(pred_df["idx"] >= start_idx) & (pred_df["idx"] <= end_idx) & (pred_df["prob"] >= 0.58) & (pred_df["final_score"] >= 0.58)]
        if not cand.empty:
            ids = [int(i) for i in cand["idx"].tolist() if int(i) in df.index]
            if ids:
                fig.add_trace(go.Scatter(
                    x=ids,
                    y=df.loc[ids, "low"] * 0.975 * mult,
                    mode="markers",
                    marker=dict(size=8, color=blue, symbol="circle", line=dict(color="#ffffff", width=1), opacity=.85),
                    name="예측 후보",
                    hovertemplate="예측 후보<extra></extra>",
                ), row=1, col=1)

    # 백테스트 매수/매도 지점. 검증용 표시.
    b = [p for p in buys if start_idx <= int(p.get("idx", -1)) <= end_idx and int(p.get("idx", -1)) in df.index]
    if b:
        ids = [int(p["idx"]) for p in b]
        fig.add_trace(go.Scatter(
            x=ids,
            y=df.loc[ids, "low"] * 0.94 * mult,
            mode="markers+text",
            text=["BUY"] * len(ids),
            textposition="bottom center",
            marker=dict(symbol="triangle-up", size=15, color=up, line=dict(color="#ffffff", width=1)),
            textfont=dict(color=up, size=10),
            name="백테스트 매수",
        ), row=1, col=1)
    s = [p for p in sells if start_idx <= int(p.get("idx", -1)) <= end_idx and int(p.get("idx", -1)) in df.index]
    if s:
        ids = [int(p["idx"]) for p in s]
        fig.add_trace(go.Scatter(
            x=ids,
            y=df.loc[ids, "high"] * 1.055 * mult,
            mode="markers+text",
            text=["SELL"] * len(ids),
            textposition="top center",
            marker=dict(symbol="triangle-down", size=15, color=down, line=dict(color="#ffffff", width=1)),
            textfont=dict(color=down, size=10),
            name="백테스트 매도",
        ), row=1, col=1)

    vol_colors = [up if dfp.loc[i, "close"] >= dfp.loc[i, "open"] else down for i in dfp.index]
    fig.add_trace(go.Bar(x=x, y=dfp["volume"], marker=dict(color=vol_colors), opacity=0.52, name="거래량", hovertemplate="거래량 %{y:,.0f}<extra></extra>"), row=2, col=1)
    if "volMa20" in dfp:
        fig.add_trace(go.Scatter(x=x, y=dfp["volMa20"], line=dict(color=muted, width=1.2), name="거래량 MA20", hovertemplate="거래량 MA20 %{y:,.0f}<extra></extra>"), row=2, col=1)

    fig.add_trace(go.Scatter(x=x, y=dfp["rsi"], line=dict(color="#f472b6", width=1.45), name="RSI", hovertemplate="RSI %{y:.1f}<extra></extra>"), row=3, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,68,68,.10)", line_width=0, row=3, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(34,197,94,.10)", line_width=0, row=3, col=1)
    for y in [30, 50, 70]:
        fig.add_hline(y=y, line_width=0.7, line_dash="dot", line_color=grid, row=3, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)

    hist_colors = [up if v >= 0 else down for v in dfp["macd_hist"].fillna(0)]
    fig.add_trace(go.Bar(x=x, y=dfp["macd_hist"], marker=dict(color=hist_colors), opacity=0.48, name="MACD Hist"), row=4, col=1)
    fig.add_trace(go.Scatter(x=x, y=dfp["macd"], line=dict(color=blue, width=1.3), name="MACD"), row=4, col=1)
    fig.add_trace(go.Scatter(x=x, y=dfp["macd_signal"], line=dict(color=orange, width=1.15), name="Signal"), row=4, col=1)
    fig.add_hline(y=0, line_width=0.8, line_color=grid, row=4, col=1)

    step = max(1, len(dfp) // 8)
    tickvals = x[::step]
    ticktext = dates[::step]
    for r in range(1, 5):
        fig.update_xaxes(
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            showgrid=True,
            gridcolor=grid,
            zeroline=False,
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikecolor="rgba(226,232,240,.55)",
            spikethickness=1,
            row=r,
            col=1,
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor=grid,
            zeroline=False,
            showspikes=True,
            spikecolor="rgba(226,232,240,.55)",
            tickfont=dict(color=text_color, size=11),
            titlefont=dict(color=muted, size=11),
            row=r,
            col=1,
        )

    tick = money_tick_settings()
    fig.update_yaxes(title_text=f"가격({unit})", row=1, col=1, **tick)
    fig.update_yaxes(title_text="거래량(주/개)", tickformat=".2s", row=2, col=1)
    fig.update_yaxes(title_text="RSI", tickformat=".0f", row=3, col=1)
    fig.update_yaxes(title_text="MACD", tickformat=".3f", row=4, col=1)
    fig.update_xaxes(title_text="날짜", row=4, col=1, titlefont=dict(color=muted, size=11))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=bg,
        plot_bgcolor=panel,
        height=840,
        margin=dict(l=12, r=16, t=28, b=8),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        dragmode="pan",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.015,
            xanchor="left",
            x=0,
            font=dict(size=11, color=text_color),
            bgcolor="rgba(15,23,42,.55)",
            bordercolor="rgba(148,163,184,.22)",
            borderwidth=1,
        ),
        font=dict(family="Pretendard, sans-serif", color=text_color),
        hoverlabel=dict(bgcolor="#111827", bordercolor="#334155", font=dict(color="#f8fafc", size=12)),
        title=dict(text="", font=dict(color=text_color)),
        modebar=dict(bgcolor="rgba(15,23,42,.80)", color=muted, activecolor=blue),
    )
    return fig

def make_equity_chart(equity_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    unit = money_unit_label()
    tick = money_tick_settings()
    mult = EXCHANGE_RATE if st.session_state.get("currency") == "KRW" else 1.0
    if equity_df is not None and not equity_df.empty:
        x = list(range(len(equity_df)))
        equity_values = equity_df["equity"] * mult
        fig.add_trace(go.Scatter(
            x=x,
            y=equity_values,
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(37,99,235,.08)",
            line=dict(color=C["blue"], width=2),
            name=f"자산({unit})",
            hovertemplate="자산 %{y:,.2f}<extra></extra>",
        ))
        roll = equity_df["equity"].cummax()
        dd = (equity_df["equity"] / roll - 1) * 100
        fig.add_trace(go.Scatter(
            x=x,
            y=dd,
            yaxis="y2",
            line=dict(color=C["red"], width=1, dash="dot"),
            name="낙폭(%)",
            hovertemplate="낙폭 %{y:.2f}%<extra></extra>",
        ))
        step = max(1, len(x) // 6)
        fig.update_xaxes(tickmode="array", tickvals=x[::step], ticktext=equity_df["date"].astype(str).tolist()[::step])
    fig.update_layout(
        template="plotly_white",
        height=300,
        margin=dict(l=8, r=8, t=16, b=8),
        yaxis=dict(title=f"자산({unit})", gridcolor="#e2e8f0", **tick),
        yaxis2=dict(title="낙폭(%)", overlaying="y", side="right", showgrid=False, ticksuffix="%"),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.05, x=0),
        font=dict(color=C["text"]),
    )
    return fig



# ============================================================
# UI 구성
# ============================================================

st.sidebar.markdown("### 기본 설정")
currency_mode = st.sidebar.radio("표시 통화", ["달러 ($)", "원 (₩)"], horizontal=True)
st.session_state.currency = "KRW" if "원" in currency_mode else "USD"

st.sidebar.markdown("---")
st.sidebar.markdown("### 예측 설정")
style = st.sidebar.selectbox("예측 스타일", ["안전형", "균형형", "공격형"], index=1)
forecast_horizon = st.sidebar.selectbox("예측 기간", [3, 5, 10, 20], index=1, format_func=lambda v: f"{v}봉")
chart_bars = st.sidebar.slider("차트 표시 봉 수", 80, 1200, 300, 20, format="%d봉")
show_bb = st.sidebar.checkbox("볼린저밴드 표시", value=True)
st.sidebar.markdown("---")
st.sidebar.markdown("### 실시간 차트")
auto_refresh_enabled = st.sidebar.checkbox("장중 자동 새로고침", value=True)
auto_refresh_seconds = st.sidebar.selectbox("새로고침 간격", [30, 60, 120], index=1, format_func=lambda v: f"{v}초")
st.sidebar.caption("1분봉·15분봉·30분봉·1시간봉에서 장이 열려 있으면 자동으로 새로고침합니다.")
st.sidebar.markdown("---")
use_arima = st.sidebar.checkbox("ARIMA 보조모델 사용", value=False, disabled=not STATSMODELS_AVAILABLE)
st.sidebar.caption("예측 기간은 몇 개의 봉 뒤를 볼지 뜻합니다. 예: 5봉 = 현재 봉 기준 5개 봉 뒤.")

style_defaults = {
    "안전형": {"min_prob": 0.62, "min_expected": 0.015, "min_score": 0.66, "risk_penalty": 1.10, "use_gb": False, "use_rf": False},
    "균형형": {"min_prob": 0.59, "min_expected": 0.010, "min_score": 0.60, "risk_penalty": 0.85, "use_gb": True, "use_rf": False},
    "공격형": {"min_prob": 0.56, "min_expected": 0.005, "min_score": 0.55, "risk_penalty": 0.60, "use_gb": True, "use_rf": True},
}
sp = style_defaults[style]

with st.sidebar.expander("고급 예측 기준", expanded=False):
    min_prob = st.slider("최소 상승확률", 50, 90, int(round(sp["min_prob"] * 100)), 1, format="%d%%") / 100
    min_expected = st.slider("최소 기대수익", 0.0, 8.0, float(sp["min_expected"] * 100), 0.1, format="%.1f%%") / 100
    min_score = st.slider("최소 종합점수", 30, 95, int(round(sp["min_score"] * 100)), 1, format="%d%%") / 100
    use_market_filter = st.checkbox("시장 흐름 필터", value=True)
    use_gb = st.checkbox("비선형 회귀모델 추가", value=bool(sp["use_gb"]))
    use_rf = st.checkbox("상승확률 보조모델 추가", value=bool(sp["use_rf"]))
else_market_filter = True

use_market_filter = locals().get("use_market_filter", True)
use_gb = locals().get("use_gb", bool(sp["use_gb"]))
use_rf = locals().get("use_rf", bool(sp["use_rf"]))
min_prob = locals().get("min_prob", float(sp["min_prob"]))
min_expected = locals().get("min_expected", float(sp["min_expected"]))
min_score = locals().get("min_score", float(sp["min_score"]))

cfg = StrategyConfig(
    style=style,
    horizon=int(forecast_horizon),
    lookback=60,
    train_window=360,
    min_prob=float(min_prob),
    min_expected=float(min_expected),
    min_score=float(min_score),
    risk_penalty=float(sp["risk_penalty"]),
    market_filter=bool(use_market_filter),
    use_arima=bool(use_arima),
    use_gb=bool(use_gb),
    use_rf=bool(use_rf),
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 매매 관리")
raw_start_capital = st.sidebar.text_input("시작 자산", value="10,000.00 $", help="뒤의 $ 기호는 단위 표시입니다. 숫자만 바꿔도 됩니다.")
start_capital = max(100.0, parse_number_text(raw_start_capital, 10000.0))
st.sidebar.caption(f"적용값: {start_capital:,.2f} $ · 원화 표시는 결과 화면에서 환율을 적용합니다.")
risk_per_trade = st.sidebar.slider("1회 손실 한도", 0.2, 3.0, 1.0, 0.1, format="%.1f%%") / 100
max_position = st.sidebar.slider("최대 투입 비중", 5.0, 100.0, 30.0, 5.0, format="%.0f%%") / 100
stop_pct = st.sidebar.slider("손절폭", 1.0, 20.0, 5.0, 0.5, format="%.1f%%") / 100
take_profit = st.sidebar.slider("익절 목표", 2.0, 40.0, 12.0, 0.5, format="%.1f%%") / 100
trailing_stop = st.sidebar.slider("추적손절", 0.0, 20.0, 6.0, 0.5, format="%.1f%%") / 100
st.sidebar.caption("손절폭·익절 목표·추적손절은 모두 진입가 기준 비율입니다.")
ma_exit = st.sidebar.selectbox("추세 이탈 매도", ["MA20 이탈", "MA60 이탈", "사용 안 함"], index=0)
with st.sidebar.expander("시간청산 옵션", expanded=False):
    use_time_stop = st.checkbox("시간청산 사용", value=False)
    max_hold_bars = st.slider("최대 보유 봉 수", 20, 400, 160, 10, format="%d봉")

st.sidebar.markdown("---")
st.sidebar.markdown("### 거래비용")
fee_pct = st.sidebar.slider("수수료", 0.0, 1.0, 0.05, 0.01, format="%.2f%%") / 100
slippage_pct = st.sidebar.slider("슬리피지", 0.0, 1.0, 0.08, 0.01, format="%.2f%%") / 100
sell_tax_pct = st.sidebar.slider("매도세/기타", 0.0, 1.0, 0.00, 0.01, format="%.2f%%") / 100
st.sidebar.caption("거래비용은 매수·매도 체결 시 백테스트 수익률에서 차감됩니다.")

cost_cfg = CostConfig(fee_pct, slippage_pct, sell_tax_pct)
risk_cfg = RiskConfig(start_capital, risk_per_trade, max_position, stop_pct, take_profit, trailing_stop, ma_exit, use_time_stop, max_hold_bars)

if not SKLEARN_AVAILABLE:
    st.sidebar.warning("scikit-learn이 없어서 기본 Ridge 계산만 사용됩니다.")
if use_arima and not STATSMODELS_AVAILABLE:
    st.sidebar.warning("statsmodels가 없어 ARIMA를 사용할 수 없습니다.")

# ============================================================
# 분석 및 렌더링
# ============================================================

def analyze_symbol(symbol: str, interval: str, data_range: str, scan_mode: bool = False):
    df_raw, name = fetch_ohlcv(symbol, interval, data_range)
    min_len = max(cfg.lookback, 240) + cfg.horizon + 90
    if df_raw is None or len(df_raw) < min_len:
        return None, name, None, pd.DataFrame(), None, [], pd.DataFrame(), [], [], {}, 0, 0
    df = calculate_indicators(df_raw)
    market = get_market_regime(symbol, interval, data_range)
    step = 18 if scan_mode else (12 if style == "안전형" else 9 if style == "균형형" else 7)
    pred_df = walk_forward_predictions(df, cfg, cost_cfg, market, step=step)
    latest = latest_prediction(df, cfg, cost_cfg, market)
    trades, equity, buys, sells = run_backtest(df, pred_df, cfg, cost_cfg, risk_cfg, market["ok"])
    metrics = calc_metrics(trades, equity, risk_cfg.start_capital)
    acc, acc_n = prediction_accuracy(pred_df)
    return df, name, market, pred_df, latest, trades, equity, buys, sells, metrics, acc, acc_n


def make_scan_config() -> StrategyConfig:
    # 스캐너는 여러 종목을 반복 분석하므로 계산량이 큰 ARIMA/RF는 기본적으로 제외한다.
    return StrategyConfig(
        style=cfg.style,
        horizon=cfg.horizon,
        lookback=cfg.lookback,
        train_window=min(cfg.train_window, 260),
        min_prob=cfg.min_prob,
        min_expected=cfg.min_expected,
        min_score=cfg.min_score,
        risk_penalty=cfg.risk_penalty,
        market_filter=cfg.market_filter,
        use_arima=False,
        use_gb=cfg.use_gb,
        use_rf=False,
    )


def analyze_symbol_for_scanner(symbol: str, interval: str, data_range: str) -> Dict:
    row = {"종목": symbol, "이름": TICKER_NAME_MAP.get(symbol, symbol)}
    try:
        df_raw, name = fetch_ohlcv(symbol, interval, data_range)
        row["이름"] = name or row["이름"]
        min_len = max(cfg.lookback, 240) + cfg.horizon + 30
        if df_raw is None or len(df_raw) < min_len:
            row.update({"판단": "데이터 부족", "이유": "가격 데이터 부족"})
            return row
        df = calculate_indicators(df_raw)
        market = get_market_regime(symbol, interval, data_range)
        scan_cfg = make_scan_config()
        latest = latest_prediction(df, scan_cfg, cost_cfg, market)
        if latest is None:
            row.update({"판단": "데이터 부족", "이유": "모델 학습 데이터 부족"})
            return row
        last = len(df) - 1
        current = float(df.loc[last, "close"])
        prev = float(df.loc[last - 1, "close"]) if last > 0 else current
        day_pct = (current / prev - 1) * 100 if prev else 0.0
        dollar_volume = float(df.loc[last, "close"] * df.loc[last, "volume"]) if "volume" in df else 0.0
        row.update({
            "판단": latest["grade"],
            "이유": latest["reason"],
            "현재가": fmt_curr(current),
            "등락률": f"{day_pct:+.2f}%",
            "상승확률": latest["prob"] * 100,
            "예상수익률": latest["pred_return"] * 100,
            "비용차감 기대수익": latest["expected"] * 100,
            "종합점수": latest["final_score"] * 100,
            "리스크": latest["risk"] * 100,
            "거래량": fmt_compact_number(float(df.loc[last, "volume"])),
            "거래대금": fmt_money_compact_usd(dollar_volume),
            "시장": market.get("text", ""),
        })
        return row
    except Exception as e:
        row.update({"판단": "오류", "이유": str(e)[:120]})
        return row


def format_scan_output(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    order = {"강한 매수": 0, "매수 후보": 1, "관심": 2, "대기": 3, "회피": 4, "데이터 부족": 5, "오류": 6}
    if "판단" in out:
        out["rank"] = out["판단"].map(order).fillna(9)
        sort_cols = ["rank"]
        for col in ["종합점수", "비용차감 기대수익", "상승확률"]:
            if col in out:
                sort_cols.append(col)
        out = out.sort_values(sort_cols, ascending=[True] + [False] * (len(sort_cols) - 1)).drop(columns=["rank"])
    for col in ["상승확률", "예상수익률", "비용차감 기대수익", "종합점수", "리스크"]:
        if col in out:
            out[col] = out[col].map(lambda x: f"{float(x):+.2f}%" if col in ["예상수익률", "비용차감 기대수익"] and pd.notna(x) else (f"{float(x):.2f}%" if pd.notna(x) else ""))
    preferred = ["판단", "종목", "이름", "현재가", "등락률", "상승확률", "예상수익률", "비용차감 기대수익", "종합점수", "리스크", "거래량", "거래대금", "시장", "이유"]
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    return out[cols]


def badge_html(grade: str, reason: str) -> str:
    cls = "badge-wait"
    if grade in ["강한 매수", "매수 후보"]:
        cls = "badge-buy"
    elif grade == "관심":
        cls = "badge-watch"
    elif grade == "회피":
        cls = "badge-avoid"
    return f'<span class="badge {cls}">{grade} · {reason}</span>'


def progress_bar(label: str, value: float, color: str = "linear-gradient(90deg,#2563eb,#10b981)"):
    value = float(np.clip(value, 0, 1))
    st.markdown(f'''
    <div class="progress-label"><span>{label}</span><span>{value*100:.1f}%</span></div>
    <div class="progress-track"><div class="progress-bar" style="width:{value*100:.1f}%;background:{color};"></div></div>
    ''', unsafe_allow_html=True)


def render_signal_card(latest: Optional[Dict], market: Dict):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 현재 판단")
    if latest is None:
        st.warning("예측에 필요한 데이터가 부족합니다.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    st.markdown(badge_html(latest["grade"], latest["reason"]), unsafe_allow_html=True)
    st.write("")
    st.write(f"예측 기준일: **{latest['date']}**")
    st.write(f"시장 상태: **{market['text']}**")
    st.write(f"예상 수익률: **{fmt_pct(latest['pred_return'])}**")
    st.write(f"거래비용 차감 후: **{fmt_pct(latest['expected'])}**")
    progress_bar("상승확률", latest["prob"], "linear-gradient(90deg,#2563eb,#06b6d4)")
    progress_bar("종합점수", latest["final_score"], "linear-gradient(90deg,#2563eb,#10b981)")
    progress_bar("리스크", latest["risk"], "linear-gradient(90deg,#f59e0b,#ef4444)")
    with st.expander("점수 구성", expanded=False):
        comp = pd.DataFrame([{"항목": k, "점수": f"{v*100:.1f}%"} for k, v in latest.get("components", {}).items()])
        st.dataframe(comp, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_order_plan(df: pd.DataFrame):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 주문 계획")
    if df is None or df.empty:
        st.markdown('</div>', unsafe_allow_html=True)
        return

    close = float(df["close"].iloc[-1])
    stop = close * (1 - risk_cfg.stop_pct)
    target = close * (1 + risk_cfg.take_profit_pct)
    qty, used = calc_position_size(risk_cfg.start_capital, close, stop, risk_cfg)
    rr = (target - close) / max(close - stop, 1e-9)

    def money_pair(v: float):
        full = fmt_curr(v)
        compact = fmt_money_compact_usd(v)
        if st.session_state.get("currency", "USD") == "KRW" and compact != full:
            return full, f"약 {compact}"
        return full, ""

    entry_full, entry_sub = money_pair(close)
    stop_full, stop_sub = money_pair(stop)
    target_full, target_sub = money_pair(target)
    used_full, used_sub = money_pair(used)

    html = f"""
    <div class="order-grid">
        <div class="order-item">
            <div class="order-label">진입가</div>
            <div class="order-value">{entry_full}</div>
            <div class="order-sub">1주 또는 1개 기준 {entry_sub}</div>
        </div>
        <div class="order-item">
            <div class="order-label">손익비</div>
            <div class="order-value small-value">1 : {rr:.2f}</div>
            <div class="order-sub">위험 1 대비 기대 보상</div>
        </div>
        <div class="order-item">
            <div class="order-label">손절가</div>
            <div class="order-value order-down">{stop_full}</div>
            <div class="order-sub">진입가 대비 -{risk_cfg.stop_pct*100:.1f}% {stop_sub}</div>
        </div>
        <div class="order-item">
            <div class="order-label">목표가</div>
            <div class="order-value order-up">{target_full}</div>
            <div class="order-sub">진입가 대비 +{risk_cfg.take_profit_pct*100:.1f}% {target_sub}</div>
        </div>
        <div class="order-item">
            <div class="order-label">권장 수량</div>
            <div class="order-value small-value">{qty:,.4f}</div>
            <div class="order-sub">단위: 주 또는 개</div>
        </div>
        <div class="order-item">
            <div class="order-label">예상 투입</div>
            <div class="order-value">{used_full}</div>
            <div class="order-sub">총 매수 금액 {used_sub}</div>
        </div>
    </div>
    <div class="order-note">
        수량은 1회 손실 한도와 최대 투입 비중을 기준으로 계산됩니다.<br>
        원화 표시는 전체 금액을 먼저 보여주고 큰 금액은 만 원 또는 억 원 단위 보조 표시를 함께 제공합니다.
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def render_kpis(metrics: Dict, acc: float, acc_n: int):
    pf = "∞" if metrics.get("Profit Factor", 0) >= 999 else f"{metrics.get('Profit Factor', 0):.2f}"
    cards = [
        ("총수익률", f"{metrics.get('총수익률', 0):+.2f}%", "단위: %"),
        ("최대낙폭 MDD", f"{metrics.get('MDD', 0):.2f}%", "단위: %"),
        ("승률", f"{metrics.get('승률', 0):.1f}%", "단위: %"),
        ("Profit Factor", pf, "총수익 ÷ 총손실"),
        ("Sharpe", f"{metrics.get('Sharpe', 0):.2f}", "위험 대비 수익"),
    ]
    html = '<div class="kpi-grid">'
    for label, value, unit in cards:
        html += f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-unit">{unit}</div></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)
    st.caption(f"거래횟수 {metrics.get('거래횟수',0)}회 · 방향 적중률 {acc:.1f}% · 검증표본 {acc_n}개")




def render_asset_page(asset_type: str, is_coin: bool, defaults: List[str], default_text: str):
    label = "코인" if is_coin else "주식"
    st.markdown(f'<div class="card"><p class="title">{label} 분석</p><p class="sub">수학적 모델과 기술적 지표를 결합해 매수 후보를 판단합니다.</p></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([3, 4, 3])
    with c1:
        method = st.radio("검색 방식", ["목록", "직접 입력"], horizontal=True, key=f"method_{asset_type}")
    with c2:
        if method == "목록":
            user_input = st.selectbox("종목", defaults, key=f"select_{asset_type}")
        else:
            user_input = st.text_input("종목명 또는 티커", default_text, key=f"input_{asset_type}")
    with c3:
        tf = st.selectbox("차트 주기", list(TIMEFRAME_MAP.keys()), index=1, key=f"tf_{asset_type}")
    interval, data_range = TIMEFRAME_MAP[tf]
    symbol = search_ticker_by_name(user_input, is_coin)
    if not symbol:
        st.warning("종목을 입력하세요.")
        return

    live_clock = render_live_status(symbol, tf)

    with st.spinner("데이터 분석 중..."):
        df, name, market, pred_df, latest, trades, equity, buys, sells, metrics, acc, acc_n = analyze_symbol(symbol, interval, data_range)
    if df is None:
        st.error("데이터가 부족하거나 불러오지 못했습니다. 일봉을 선택하거나 다른 종목을 입력하세요.")
        return

    last = len(df) - 1
    current = float(df.loc[last, "close"])
    prev = float(df.loc[last - 1, "close"])
    diff = current - prev
    pct = diff / prev * 100 if prev else 0
    sign = "+" if diff > 0 else ""
    cls = "up" if diff >= 0 else "down"
    st.markdown(f'''
    <div class="card">
      <div class="small">{symbol} · {tf} · {cfg.style} · {live_clock.get('label', '')}</div>
      <div class="title">{name}</div>
      <div class="price">{fmt_curr(current)} <span class="{cls}" style="font-size:15px;">{sign}{fmt_curr(diff)} ({sign}{pct:.2f}%)</span></div>
    </div>
    ''', unsafe_allow_html=True)

    left, right = st.columns([7.3, 2.7])
    with left:
        st.markdown('<div class="chart-guide"><b>차트 조작</b> · 마우스 휠 확대/축소 · 드래그 이동 · 우측 상단 도구로 확대 영역 선택 가능 · 1분봉은 장중 자동 갱신을 지원합니다 · BUY/SELL 표시는 백테스트 검증 지점입니다.</div>', unsafe_allow_html=True)
        st.plotly_chart(
            make_chart(df, pred_df, buys, sells, chart_bars, show_bb),
            use_container_width=True,
            config={"scrollZoom": True, "displayModeBar": True, "responsive": True},
        )
        render_kpis(metrics, acc, acc_n)
        st.plotly_chart(make_equity_chart(equity), use_container_width=True, config={"displayModeBar": True, "responsive": True})
    with right:
        render_signal_card(latest, market)
        render_order_plan(df)

    st.markdown("### 백테스트 거래 기록")
    if trades:
        h = pd.DataFrame(trades)
        show = h[["매수일", "매수가", "매도일", "매도가", "수량", "수익률", "청산", "상승확률", "기대수익", "종합점수", "청산후자산", "성공"]].copy()
        for col in ["매수가", "매도가", "청산후자산"]:
            show[col] = show[col].apply(fmt_curr)
        show["수익률"] = show["수익률"].map(lambda x: f"{x:+.2f}%")
        for col in ["상승확률", "기대수익", "종합점수"]:
            show[col] = show[col].map(lambda x: f"{x*100:.1f}%" if abs(x) <= 1.5 else f"{x:.1f}")
        show["성공"] = show["성공"].map(lambda x: "성공" if x else "실패")
        st.dataframe(show.iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.info("현재 조건에서는 백테스트 거래가 없습니다. 공격형으로 바꾸거나 기준을 완화해 보세요.")



def render_scanner():
    st.markdown('<div class="card"><p class="title">종목 스캐너</p><p class="sub">여러 종목을 같은 기준으로 비교합니다. 스캐너는 멈추지 않도록 빠른 모델로 현재 매수 후보만 우선 판별합니다.</p></div>', unsafe_allow_html=True)
    scan_type = st.radio("대상", ["주식", "코인", "직접 입력"], horizontal=True)
    if scan_type == "주식":
        symbols = STOCK_SCAN_LIST
    elif scan_type == "코인":
        symbols = COIN_SCAN_LIST
    else:
        raw = st.text_area("티커를 쉼표로 입력", "AAPL,NVDA,TSLA,BTC-USD,ETH-USD")
        symbols = [x.strip().upper() for x in raw.split(",") if x.strip()]

    if not symbols:
        st.warning("스캔할 종목이 없습니다. 티커를 하나 이상 입력하세요.")
        return

    max_scan = st.slider("스캔 개수", 1, min(20, len(symbols)), min(8, len(symbols)), format="%d개")
    tf = st.selectbox("차트 주기", list(TIMEFRAME_MAP.keys()), index=1, key="scan_tf")
    interval, data_range = TIMEFRAME_MAP[tf]
    selected_symbols = symbols[:max_scan]

    st.caption("진행률은 완료 비율과 남은 비율을 같이 표시합니다. 네트워크가 느린 종목은 현재 종목명에서 잠시 머물 수 있습니다.")

    if st.button("스캔 실행", use_container_width=True):
        rows: List[Dict] = []
        start_time = time.time()
        progress = st.progress(0, text=f"0% 완료 · 100% 남음 · 0/{max_scan}개")
        status_box = st.empty()
        table_box = st.empty()

        for k, sym in enumerate(selected_symbols):
            done_before = k / max_scan
            remain_before = 1 - done_before
            status_box.markdown(
                f'<div class="scan-status"><b>{k}/{max_scan}개 완료</b><br>현재 분석 중: <b>{sym}</b><br>완료 {done_before*100:.1f}% · 남음 {remain_before*100:.1f}%</div>',
                unsafe_allow_html=True,
            )
            progress.progress(done_before, text=f"{done_before*100:.1f}% 완료 · {remain_before*100:.1f}% 남음 · {k}/{max_scan}개")

            row = analyze_symbol_for_scanner(sym, interval, data_range)
            rows.append(row)

            done_after = (k + 1) / max_scan
            remain_after = 1 - done_after
            elapsed = time.time() - start_time
            avg = elapsed / (k + 1)
            eta = avg * (max_scan - k - 1)
            progress.progress(done_after, text=f"{done_after*100:.1f}% 완료 · {remain_after*100:.1f}% 남음 · {k+1}/{max_scan}개")
            status_box.markdown(
                f'<div class="scan-status"><b>{k+1}/{max_scan}개 완료</b><br>최근 완료: <b>{sym}</b><br>완료 {done_after*100:.1f}% · 남음 {remain_after*100:.1f}% · 예상 남은 시간 {eta:.0f}초</div>',
                unsafe_allow_html=True,
            )
            table_box.dataframe(format_scan_output(pd.DataFrame(rows)), use_container_width=True, hide_index=True)

        progress.progress(1.0, text=f"100.0% 완료 · 0.0% 남음 · {max_scan}/{max_scan}개")
        status_box.success(f"스캔 완료: {max_scan}개 분석 · 소요 시간 {time.time() - start_time:.1f}초")
        table_box.dataframe(format_scan_output(pd.DataFrame(rows)), use_container_width=True, hide_index=True)



# ============================================================
# 메인
# ============================================================

st.markdown('<div class="card"><p class="title">시계열 수학 모델 매수 타점 예측</p><p class="sub">Ridge · Logistic · 비선형 모델 · 선택형 ARIMA · 1분봉 장중 자동 갱신 · 백테스트 · 리스크 관리</p></div>', unsafe_allow_html=True)

page = st.radio("", ["주식", "코인", "스캐너"], horizontal=True, label_visibility="collapsed")

if page == "주식":
    render_asset_page("stock", False, ["애플", "엔비디아", "테슬라", "마이크로소프트", "구글", "삼성전자", "SK하이닉스", "네이버", "카카오", "현대차"], "애플")
elif page == "코인":
    render_asset_page("coin", True, ["비트코인", "이더리움", "솔라나", "리플", "도지코인", "아발란체"], "비트코인")
else:
    render_scanner()
