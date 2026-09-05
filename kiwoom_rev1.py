import math
import os
import tempfile
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import ta
import yfinance as yf
from google import genai

# ----------------------------------------------------
# 0. Gemini API Key 로드 (Streamlit Secrets 활용)
# ----------------------------------------------------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

# ----------------------------------------------------
# Streamlit Community Cloud 등 클라우드 환경 대응
# ----------------------------------------------------
try:
  _cache_dir = os.path.join(tempfile.gettempdir(), "yfinance_cache")
  os.makedirs(_cache_dir, exist_ok=True)
  yf.set_tz_cache_location(_cache_dir)
except Exception:
  pass


@st.cache_resource(show_spinner=False)
def get_yf_session():
  try:
    from curl_cffi import requests as cffi_requests

    return cffi_requests.Session(impersonate="chrome")
  except Exception:
    return None


def _yf_ticker(symbol):
  session = get_yf_session()
  if session is not None:
    return yf.Ticker(symbol, session=session)
  return yf.Ticker(symbol)


def _fetch_history_with_retry(symbol, period="2y", retries=2, base_delay=1.5):
  last_err = None
  for attempt in range(retries + 1):
    try:
      obj = _yf_ticker(symbol)
      df = obj.history(period=period, timeout=15)
      if df is not None and not df.empty:
        return obj, df, None
      last_err = "빈 데이터 응답(empty response)"
    except Exception as e:
      last_err = f"{type(e).__name__}: {e}"
    if attempt < retries:
      time.sleep(base_delay * (attempt + 1))
  return None, pd.DataFrame(), last_err


# 1. 페이지 레이아웃 설정
st.set_page_config(
    page_title="월가&코스피 프롭트레이더 전략 시스템",
    page_icon="⚡",
    layout="wide",
)

# 화면 백지화 방지 및 AI 대화 내역 세션 초기화
if "analysis_cache" not in st.session_state:
  st.session_state["analysis_cache"] = None
if "chat_history" not in st.session_state:
  st.session_state["chat_history"] = []

st.title("⚡ 주가 정밀분석 및 TOP 5 추천시스템")
st.caption(
    "실시간 파동, 변동성(ATR), 수급 모멘텀, 시장 레짐 및 히스토리 백테스트 기반 매매 전략"
)
st.markdown("---")

# 국내 주식 사전 (한국어 명칭)
STOCKS_KR = {
    "삼성전자": "005930.KS",
    "SK하이닉스": "000660.KS",
    "한화오션": "042660.KS",
    "HD현대중공업": "329180.KS",
    "두산에너빌리티": "034020.KS",
    "HD현대일렉트릭": "267260.KS",
    "LS ELECTRIC": "010120.KS",
    "HD현대": "267250.KS",
    "삼성전기": "009150.KS",
    "SK텔레콤": "017670.KS",
    "현대모비스": "012330.KS",
    "LS": "006260.KS",
    "대한광통신": "010170.KQ",
    "한중엔시에스": "107640.KQ",
    "대한전선": "001440.KS",
    "한화에어로스페이스": "012450.KS",
    "로보티즈": "108490.KQ",
    "HJ중공업": "097230.KS",
    "리노공업": "058470.KQ",
    "✏️ 국내주식 코드 직접 입력": "CUSTOM",
}

# 해외 주식 주요 종목 사전
STOCKS_US = {
    "애플 (AAPL)": "AAPL",
    "AMD (AMD)": "AMD",
    "아마존닷컴 (AMZN)": "AMZN",
    "브로드컴 (AVGO)": "AVGO",
    "블룸 에너지 (BE)": "BE",
    "알파벳 A (GOOGL)": "GOOGL",
    "인텔 (INTC)": "INTC",
    "메타 플랫폼스 (META)": "META",
    "마이크로소프트 (MSFT)": "MSFT",
    "마이크론 테크놀로지 (MU)": "MU",
    "엔비디아 (NVDA)": "NVDA",
    "로켓 랩 (RKLB)": "RKLB",
    "샌디스크/웨스턴디지털 (WDC)": "WDC",
    "미국 반도체 3배 ETF (SOXL)": "SOXL",
    "QQQ 레버리지 3배 ETF (TQQQ)": "TQQQ",
    "테슬라 (TSLA)": "TSLA",
    "버티브 홀딩스 (VRT)": "VRT",
    "✏️ 해외주식 티커 직접 입력": "CUSTOM",
}

# 역방향 매핑 (티커/코드 -> 한국어 종목명)
TICKER_TO_NAME = {}
for k, v in STOCKS_KR.items():
  if v != "CUSTOM":
    TICKER_TO_NAME[v] = k
    TICKER_TO_NAME[v.split(".")[0]] = k
for k, v in STOCKS_US.items():
  if v != "CUSTOM":
    TICKER_TO_NAME[v] = k


# 실시간 환율 수집 함수
@st.cache_data(ttl=3600, show_spinner=False)
def get_usdkrw_rate():
  _obj, df, err = _fetch_history_with_retry("USDKRW=X", period="5d")
  if not df.empty:
    rate = float(df["Close"].dropna().iloc[-1])
    if rate > 0:
      return rate, None
  return 1350.0, err


# 시장 레짐(강세/약세) 판단
@st.cache_data(ttl=900, show_spinner=False)
def fetch_index_trend(is_krx):
  ticker = "^KS11" if is_krx else "^GSPC"
  _obj, df, err = _fetch_history_with_retry(ticker, period="2y")
  if not df.empty:
    df.columns = [c.lower() for c in df.columns]
    df["ma50"] = df["close"].rolling(50).mean()
    df["ma200"] = df["close"].rolling(200).mean()
    return df, None
  return pd.DataFrame(), err


def get_market_regime(idx_df):
  if idx_df.empty or len(idx_df) < 200 or pd.isna(idx_df["ma200"].iloc[-1]):
    return "판단 보류(데이터 부족)", "⚪", 0
  last = idx_df.iloc[-1]
  if last["close"] > last["ma50"] > last["ma200"]:
    return "상승장(강세)", "🟢", 1
  elif last["close"] < last["ma50"] < last["ma200"]:
    return "하락장(약세)", "🔴", -1
  else:
    return "혼조/전환구간", "🟡", 0


def add_relative_strength(df, idx_df, lookback=20):
  df = df.copy()
  if idx_df is None or idx_df.empty:
    df["rel_strength"] = np.nan
    return df
  idx_close = idx_df["close"].reindex(df.index).ffill().bfill()
  stock_ret = df["close"].pct_change(lookback)
  idx_ret = idx_close.pct_change(lookback)
  df["rel_strength"] = (stock_ret - idx_ret) * 100
  return df


# 지표 계산 함수
def calculate_indicators(df):
  df = df.copy()
  df.columns = [c.lower() for c in df.columns]

  df["ma5"] = ta.trend.sma_indicator(df["close"], window=5)
  df["ma20"] = ta.trend.sma_indicator(df["close"], window=20)
  df["ma60"] = ta.trend.sma_indicator(df["close"], window=60)
  df["rsi"] = ta.momentum.rsi(df["close"], window=14)
  df["atr"] = ta.volatility.average_true_range(
      df["high"], df["low"], df["close"], window=14
  )

  df["macd"] = ta.trend.macd(df["close"])
  df["macd_signal"] = ta.trend.macd_signal(df["close"])

  bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
  df["bb_upper"] = bb.bollinger_hband()
  df["bb_lower"] = bb.bollinger_lband()

  df["vol_ma20"] = ta.trend.sma_indicator(df["volume"], window=20)
  df["vol_ratio"] = (df["volume"] / df["vol_ma20"]) * 100

  try:
    adx_ind = ta.trend.ADXIndicator(
        high=df["high"], low=df["low"], close=df["close"], window=14
    )
    df["adx"] = adx_ind.adx()
  except Exception:
    df["adx"] = np.nan

  try:
    obv_ind = ta.volume.OnBalanceVolumeIndicator(
        close=df["close"], volume=df["volume"]
    )
    df["obv"] = obv_ind.on_balance_volume()
    df["obv_ma20"] = df["obv"].rolling(20).mean()
  except Exception:
    df["obv"] = np.nan
    df["obv_ma20"] = np.nan

  try:
    stoch_ind = ta.momentum.StochasticOscillator(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14,
        smooth_window=3,
    )
    df["stoch_k"] = stoch_ind.stoch()
    df["stoch_d"] = stoch_ind.stoch_signal()
  except Exception:
    df["stoch_k"] = np.nan
    df["stoch_d"] = np.nan

  df["high_52w"] = df["high"].rolling(252, min_periods=60).max()
  df["pct_from_52w_high"] = (df["close"] / df["high_52w"] - 1) * 100

  df = df.ffill().bfill()
  return df


# 퀀트 점수 계산
def compute_quant_score(curr):
  score = 0
  reasons = []
  warnings = []

  if curr["close"] > curr["ma20"]:
    score += 15
    reasons.append("20일선 상회(단기 상승추세)")
  if curr["ma20"] > curr["ma60"]:
    score += 15
    reasons.append("이동평균 정배열(20>60)")

  adx = curr.get("adx", np.nan)
  if not pd.isna(adx):
    if adx >= 25:
      score += 10
      reasons.append(f"추세 강도 우수(ADX {adx:.0f})")
    elif adx < 15:
      score -= 8
      warnings.append("추세 미약(횡보장) - 신호 신뢰도 낮음")

  rsi = curr.get("rsi", np.nan)
  if not pd.isna(rsi):
    if 40 <= rsi <= 65:
      score += 15
      reasons.append("RSI 건전한 상승모멘텀")
    elif rsi > 75:
      score -= 12
      warnings.append(f"RSI 과매수 경고({rsi:.0f})")
    elif rsi < 30:
      score -= 6
      warnings.append(f"RSI 과매도 구간({rsi:.0f})")

  stoch_k = curr.get("stoch_k", np.nan)
  stoch_d = curr.get("stoch_d", np.nan)
  if not pd.isna(stoch_k) and not pd.isna(stoch_d):
    if stoch_k > stoch_d and stoch_k < 80:
      score += 10
      reasons.append("스토캐스틱 골든크로스")

  if curr["macd"] > curr["macd_signal"]:
    score += 15
    reasons.append("MACD 매수 신호")

  if curr["vol_ratio"] >= 120:
    score += 10
    reasons.append("거래량 분출")

  obv = curr.get("obv", np.nan)
  obv_ma = curr.get("obv_ma20", np.nan)
  if not pd.isna(obv) and not pd.isna(obv_ma) and obv > obv_ma:
    score += 10
    reasons.append("OBV 매집신호(수급 우호적)")

  rel = curr.get("rel_strength", np.nan)
  if not pd.isna(rel) and rel > 0:
    score += 5
    reasons.append("시장 대비 상대강도 우위")

  pct_52 = curr.get("pct_from_52w_high", np.nan)
  if not pd.isna(pct_52) and pct_52 > -2:
    score -= 5
    warnings.append("52주 신고가 근접 - 단기 변동성 유의")

  # 기존에는 warnings가 점수에 전혀 반영되지 않아, 과매수/추세미약/고점근접
  # 등 리스크 요인이 있어도 스코어가 그대로 유지되는 결함이 있었음(위에서 감점 반영).
  # 상한뿐 아니라 하한도 0으로 고정해 음수 스코어가 나오지 않도록 클램프.
  return max(0, min(score, 100)), reasons, warnings


def get_recommendation_tier(score, regime_score):
  if score >= 75:
    tier = "🟢 적극 매수 관심"
  elif score >= 55:
    tier = "🟡 분할 매수 검토"
  elif score >= 35:
    tier = "⚪ 관망"
  else:
    tier = "🔴 회피 / 비중축소"

  if regime_score < 0 and score < 75:
    tier += " ⚠️시장 하락추세 주의"
  return tier


# 백테스트 함수
def backtest_strategy(
    df, score_threshold=60, sl_mult=1.5, tp_mult=3.0, max_hold=20
):
  trades = []
  n = len(df)
  if n < 90:
    return pd.DataFrame(trades), {
        "trades": 0,
        "win_rate": None,
        "avg_return": None,
        "profit_factor": None,
        "expectancy": None,
    }

  i = 60
  in_position = False
  entry_idx = entry_price = sl = tp = None

  while i < n - 1:
    if not in_position:
      curr = df.iloc[i]
      sc, _, _ = compute_quant_score(curr)
      atr = curr.get("atr", np.nan)
      if sc >= score_threshold and not pd.isna(atr) and atr > 0:
        entry_price = float(curr["close"])
        entry_idx = i
        sl = entry_price - atr * sl_mult
        tp = entry_price + atr * sl_mult * (tp_mult / sl_mult)
        in_position = True
      i += 1
      continue

    day = df.iloc[i]
    hold = i - entry_idx
    hit_sl = day["low"] <= sl
    hit_tp = day["high"] >= tp
    exit_price = None
    result = None

    if hit_sl:
      exit_price = sl
      result = "loss"
    elif hit_tp:
      exit_price = tp
      result = "win"
    elif hold >= max_hold:
      exit_price = float(day["close"])
      result = "win" if exit_price > entry_price else "loss"

    if exit_price is not None:
      ret_pct = (exit_price / entry_price - 1) * 100
      trades.append({
          "진입일": df.index[entry_idx].strftime("%Y-%m-%d"),
          "청산일": df.index[i].strftime("%Y-%m-%d"),
          "수익률(%)": round(ret_pct, 2),
          "결과": "승" if result == "win" else "패",
          "보유일수": hold,
      })
      in_position = False
    i += 1

  trades_df = pd.DataFrame(trades)
  if trades_df.empty:
    return trades_df, {
        "trades": 0,
        "win_rate": None,
        "avg_return": None,
        "profit_factor": None,
        "expectancy": None,
    }

  wins = trades_df[trades_df["결과"] == "승"]
  losses = trades_df[trades_df["결과"] == "패"]
  win_rate = len(wins) / len(trades_df) * 100
  avg_return = trades_df["수익률(%)"].mean()
  gross_profit = wins["수익률(%)"].sum()
  gross_loss = abs(losses["수익률(%)"].sum())
  profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.inf

  stats = {
      "trades": len(trades_df),
      "win_rate": win_rate,
      "avg_return": avg_return,
      "profit_factor": profit_factor,
      "expectancy": avg_return,
  }
  return trades_df, stats


def wilson_lower_bound(win_rate_pct, trades, z=1.96):
  """승률을 표본 크기로 보정한 통계적 하한값(95% 신뢰구간).

  거래 2회/승률100%와 거래 30회/승률65% 같은 경우를 그냥 승률(%)로만
  비교하면 표본이 적은 쪽이 부당하게 높게 평가된다. Wilson score
  하한을 쓰면 표본이 적을수록 승률이 자동으로 크게 할인되어,
  "실제로 믿을 만한" 승률에 가까운 값으로 정렬할 수 있다.
  """
  if trades in (None, 0) or win_rate_pct is None or pd.isna(win_rate_pct):
    return -1.0  # 백테스트 표본 자체가 없으면 최하위로 정렬
  n = trades
  phat = win_rate_pct / 100.0
  denom = 1 + (z ** 2) / n
  center = phat + (z ** 2) / (2 * n)
  margin = z * math.sqrt((phat * (1 - phat) / n) + (z ** 2) / (4 * n ** 2))
  return ((center - margin) / denom) * 100.0


# 주식 데이터 수집 함수
@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data(ticker, custom_display_name=None):
  if not ticker:
    return pd.DataFrame(), ticker, ticker, False, "티커가 비어 있습니다."

  is_krx = ticker.isdigit() or ticker.endswith(".KS") or ticker.endswith(".KQ")
  last_err = None

  display_name = (
      custom_display_name or TICKER_TO_NAME.get(ticker) or ticker
  )

  if ticker.isdigit():
    for suffix in [".KS", ".KQ"]:
      sym = f"{ticker}{suffix}"
      obj, df, err = _fetch_history_with_retry(sym, period="2y")
      if not df.empty and len(df) >= 60:
        return calculate_indicators(df), display_name, sym, True, None
      last_err = err
    return pd.DataFrame(), display_name, ticker, True, last_err

  obj, df, err = _fetch_history_with_retry(ticker, period="2y")
  if not df.empty and len(df) >= 60:
    return calculate_indicators(df), display_name, ticker, is_krx, None

  return pd.DataFrame(), display_name, ticker, is_krx, err


# 전체 종목 스캐닝 함수
def scan_all_stocks(
    stock_dict, is_krx, exchange_rate, capital, risk_pct, score_threshold
):
  results = []
  failed = []
  total = len([k for k, v in stock_dict.items() if v != "CUSTOM"])
  progress_bar = st.progress(0)
  count = 0

  idx_df, idx_err = fetch_index_trend(is_krx)
  regime_label, regime_icon, regime_score = get_market_regime(idx_df)
  if idx_err:
    failed.append(("(벤치마크 지수)", "^KS11" if is_krx else "^GSPC", idx_err))

  for name, ticker in stock_dict.items():
    if ticker == "CUSTOM":
      continue

    count += 1
    progress_bar.progress(count / total)

    df, s_name, sym, _, err = fetch_stock_data(ticker, custom_display_name=name)
    if df.empty or len(df) < 60:
      failed.append((name, ticker, err or "알 수 없는 오류"))
      continue

    df = add_relative_strength(df, idx_df)
    curr = df.iloc[-1]
    score, reasons, warns = compute_quant_score(curr)
    curr_price = float(curr["close"])
    curr_atr = (
        float(curr["atr"]) if not pd.isna(curr["atr"]) else curr_price * 0.02
    )

    cap_curr = capital if is_krx else capital / exchange_rate
    max_risk_cash = cap_curr * (risk_pct / 100.0)

    ma5_val = (
        float(curr["ma5"]) if not pd.isna(curr["ma5"]) else curr_price
    )
    ma20_val = (
        float(curr["ma20"]) if not pd.isna(curr["ma20"]) else curr_price * 0.98
    )
    ma60_val = (
        float(curr["ma60"]) if not pd.isna(curr["ma60"]) else curr_price * 0.95
    )

    e1 = min(ma5_val, curr_price - 0.5 * curr_atr)
    e2 = min(e1 - 1.0 * curr_atr, ma20_val)
    e3 = min(e1 - 2.0 * curr_atr, ma60_val)
    e2 = min(e2, e1 - 0.3 * curr_atr)
    e3 = min(e3, e2 - 0.3 * curr_atr)

    avg_price = (e1 * 0.3) + (e2 * 0.4) + (e3 * 0.3)

    # 손절가는 3차 진입가 기준(- 1.0*ATR 버퍼)으로 계산해 손절가가
    # 3차 진입가보다 높아지는 모순을 방지 (단일종목 분석과 동일 로직)
    sl_price = e3 - 1.0 * curr_atr
    stop_dist = avg_price - sl_price
    tp1_price = avg_price + (stop_dist * 1.5)
    tp2_price = avg_price + (stop_dist * 2.5)

    raw_qty = int(max_risk_cash / stop_dist) if stop_dist > 0 else 0
    max_affordable_qty = int(cap_curr // avg_price) if avg_price > 0 else 0
    qty = min(raw_qty, max_affordable_qty)
    if score < 35:
      qty = 0

    _, bt_stats = backtest_strategy(df, score_threshold=score_threshold)

    results.append({
        "종목명": name,
        "티커": sym,
        "퀀트점수": score,
        "추천등급": get_recommendation_tier(score, regime_score),
        "현재가": curr_price,
        "예상평단가": avg_price,
        "1차목표가": tp1_price,
        "2차목표가": tp2_price,
        "손절가": sl_price,
        "추천수량": qty,
        "핵심신호": ", ".join(reasons) if reasons else "관망 구간",
        "유의사항": ", ".join(warns) if warns else "-",
        "RSI": curr["rsi"],
        "과거승률(%)": (
            round(bt_stats["win_rate"], 1)
            if bt_stats["win_rate"] is not None
            else None
        ),
        "백테스트거래수": bt_stats["trades"],
        "평균수익률(%)": (
            round(bt_stats["avg_return"], 2)
            if bt_stats["avg_return"] is not None
            else None
        ),
        "손익비": (
            round(bt_stats["profit_factor"], 2)
            if bt_stats["profit_factor"] not in (None, np.inf)
            else bt_stats["profit_factor"]
        ),
    })

  progress_bar.empty()
  res_df = pd.DataFrame(results)
  if not res_df.empty:
    # 단순 승률(%) 대신 Wilson 신뢰구간 하한을 사용: 거래횟수가 적은 종목의
    # 승률을 통계적으로 할인해서, "표본 2회 100%"가 "표본 30회 65%"보다
    # 부당하게 우위에 서지 않도록 함.
    res_df["_wilson_sort"] = res_df.apply(
        lambda r: wilson_lower_bound(r["과거승률(%)"], r["백테스트거래수"]),
        axis=1,
    )
    # 손익비는 gross_loss=0일 때 inf가 될 수 있어 정렬 안정성을 위해 캡 처리
    res_df["_pf_sort"] = (
        res_df["손익비"].replace([np.inf, -np.inf], 999).fillna(-1)
    )
    res_df["_ret_sort"] = res_df["평균수익률(%)"].fillna(-999)

    res_df = (
        res_df.sort_values(
            by=["퀀트점수", "_wilson_sort", "_pf_sort", "_ret_sort"],
            ascending=[False, False, False, False],
        )
        .drop(columns=["_wilson_sort", "_pf_sort", "_ret_sort"])
        .reset_index(drop=True)
    )
  return res_df, regime_label, regime_icon, failed


# ----------------------------------------------------
# 2. 사이드바 설정
# ----------------------------------------------------
st.sidebar.header("⚙️ 트레이딩 분석 설정")

app_mode = st.sidebar.radio(
    "🎯 실행 모드 선택",
    ["선택 종목 개별 정밀 분석", "전체 종목 TOP 5 스캔"],
    index=0,
)

st.sidebar.markdown("---")

capital = st.sidebar.number_input(
    "총 자본금 (원화 KRW)", value=3000000, step=100000, format="%d"
)
risk_pct = st.sidebar.slider(
    "1회 트레이딩 리스크 허용치 (%)",
    min_value=0.25,
    max_value=5.0,
    value=3.0,
    step=0.25,
)

with st.sidebar.expander("🔧 고급 설정 (백테스트 신호 기준)"):
  score_threshold = st.slider(
      "백테스트/추천 신호 최소 점수",
      min_value=40,
      max_value=90,
      value=60,
      step=5,
  )
  debug_mode = st.checkbox(
      "🐞 디버그 모드 (데이터 수집 실패 원인 표시)", value=False
  )

exchange_rate, fx_err = get_usdkrw_rate()
if fx_err and debug_mode:
  st.sidebar.caption(f"⚠️ 환율 조회 실패(기본값 1350원 사용): {fx_err}")


# ----------------------------------------------------
# 모드 1: 선택 종목 개별 정밀 분석
# ----------------------------------------------------
if app_mode == "선택 종목 개별 정밀 분석":
  st.sidebar.markdown("---")
  market_type = st.sidebar.radio(
      "🌐 시장 선택", ["국내주식 (KRX)", "해외주식 (US)"], index=0, horizontal=True
  )

  selected_ticker = "005930"
  selected_display_name = "삼성전자"

  if market_type == "해외주식 (US)":
    selected_name = st.sidebar.selectbox(
        "🇺🇸 해외주식 종목 선택", list(STOCKS_US.keys())
    )
    if STOCKS_US[selected_name] == "CUSTOM":
      selected_ticker = (
          st.sidebar.text_input("티커 직접 입력 (예: PLTR)", value="NVDA")
          .strip()
          .upper()
      )
      selected_display_name = selected_ticker
    else:
      selected_ticker = STOCKS_US[selected_name]
      selected_display_name = selected_name
  else:
    selected_name = st.sidebar.selectbox(
        "🇰🇷 국내주식 종목 선택", list(STOCKS_KR.keys())
    )
    if STOCKS_KR[selected_name] == "CUSTOM":
      selected_ticker = (
          st.sidebar.text_input(
              "종목코드 6자리 입력 (예: 005930)", value="005930"
          )
          .strip()
          .upper()
      )
      selected_display_name = (
          TICKER_TO_NAME.get(selected_ticker) or selected_ticker
      )
    else:
      selected_ticker = STOCKS_KR[selected_name]
      selected_display_name = selected_name

  run_analysis = st.sidebar.button(
      "🚀 정밀 분석 & 전략 생성", type="primary", use_container_width=True
  )

  # 분석 실행 및 데이터 세션 캐싱
  if run_analysis or st.session_state["analysis_cache"] is None:
    df, stock_name, symbol_formatted, is_krx, fetch_err = fetch_stock_data(
        selected_ticker, custom_display_name=selected_display_name
    )

    if not df.empty and len(df) >= 60:
      idx_df, idx_err = fetch_index_trend(is_krx)
      regime_label, regime_icon, regime_score = get_market_regime(idx_df)
      df = add_relative_strength(df, idx_df)

      curr = df.iloc[-1]
      curr_price = float(curr["close"])
      curr_atr = float(curr["atr"]) if not pd.isna(curr["atr"]) else curr_price * 0.02
      curr_rsi = float(curr["rsi"]) if not pd.isna(curr["rsi"]) else 50.0
      vol_ratio = float(curr["vol_ratio"]) if not pd.isna(curr["vol_ratio"]) else 100.0

      score, reasons, warns = compute_quant_score(curr)
      tier = get_recommendation_tier(score, regime_score)

      currency = "(원)" if is_krx else "($)"
      fmt = "{:,.0f}" if is_krx else "{:,.2f}"

      capital_curr = capital if is_krx else capital / exchange_rate
      max_risk_cash = capital_curr * (risk_pct / 100.0)

      ma5_val = float(curr["ma5"]) if not pd.isna(curr["ma5"]) else curr_price
      ma20_val = float(curr["ma20"]) if not pd.isna(curr["ma20"]) else curr_price * 0.98
      ma60_val = float(curr["ma60"]) if not pd.isna(curr["ma60"]) else curr_price * 0.95

      entry1_price = min(ma5_val, curr_price - 0.5 * curr_atr)
      entry2_price = min(entry1_price - 1.0 * curr_atr, ma20_val)
      entry3_price = min(entry1_price - 2.0 * curr_atr, ma60_val)

      # 이평선이 역배열되거나 60일선이 크게 하회할 경우에도 1차>2차>3차 순서를
      # 항상 보장 (그렇지 않으면 분할매수 구조 자체가 깨짐)
      entry2_price = min(entry2_price, entry1_price - 0.3 * curr_atr)
      entry3_price = min(entry3_price, entry2_price - 0.3 * curr_atr)

      avg_entry_price = (entry1_price * 0.3) + (entry2_price * 0.4) + (entry3_price * 0.3)

      # 손절가는 '평단가 - 고정폭(ATR*1.5)'이 아니라 '최종(3차) 진입가 - 버퍼'로 계산.
      # 기존 방식은 3차 진입가가 60일선에 의해 크게 밀려 내려갈 경우
      # 손절가가 3차 진입가보다 높아져(=3차 체결 전에 이미 손절 발동) 전략이 모순되는 문제가 있었음.
      sl_buffer_atr = 1.0
      sl_swing = entry3_price - sl_buffer_atr * curr_atr
      stop_dist_swing = avg_entry_price - sl_swing
      tp1_swing = avg_entry_price + (stop_dist_swing * 1.5)
      tp2_swing = avg_entry_price + (stop_dist_swing * 2.5)

      max_affordable_qty_swing = int(capital_curr // avg_entry_price) if avg_entry_price > 0 else 0
      raw_qty_swing = int(max_risk_cash / stop_dist_swing) if stop_dist_swing > 0 else 0
      qty_swing = min(raw_qty_swing, max_affordable_qty_swing)
      if score < 35:
        qty_swing = 0

      trades_df, bt_stats = backtest_strategy(df, score_threshold=score_threshold)

      st.session_state["analysis_cache"] = {
          "df": df,
          "stock_name": stock_name,
          "symbol_formatted": symbol_formatted,
          "is_krx": is_krx,
          "regime_label": regime_label,
          "regime_icon": regime_icon,
          "regime_score": regime_score,
          "curr_price": curr_price,
          "curr_atr": curr_atr,
          "curr_rsi": curr_rsi,
          "vol_ratio": vol_ratio,
          "score": score,
          "reasons": reasons,
          "warns": warns,
          "tier": tier,
          "currency": currency,
          "fmt": fmt,
          "capital_curr": capital_curr,
          "max_risk_cash": max_risk_cash,
          "entry1_price": entry1_price,
          "entry2_price": entry2_price,
          "entry3_price": entry3_price,
          "avg_entry_price": avg_entry_price,
          "stop_dist_swing": stop_dist_swing,
          "sl_swing": sl_swing,
          "tp1_swing": tp1_swing,
          "tp2_swing": tp2_swing,
          "qty_swing": qty_swing,
          "trades_df": trades_df,
          "bt_stats": bt_stats,
          "ma20_val": ma20_val,
          "ma60_val": ma60_val,
      }
      if run_analysis:
        st.session_state["chat_history"] = []

  cache = st.session_state.get("analysis_cache")

  if cache is None:
    st.info("👈 왼쪽 사이드바에서 종목 설정 후 **[🚀 정밀 분석 & 전략 생성]** 버튼을 클릭해주세요.")
  else:
    bench_name = "코스피(KOSPI)" if cache["is_krx"] else "S&P500"
    if cache["regime_score"] > 0:
      st.success(f"{cache['regime_icon']} 현재 {bench_name} 기준 시장 환경: **{cache['regime_label']}** — 추세 매매에 우호적인 환경입니다.")
    elif cache["regime_score"] < 0:
      st.error(f"{cache['regime_icon']} 현재 {bench_name} 기준 시장 환경: **{cache['regime_label']}** — 개별 종목 신호가 좋아도 전체 시장 역풍에 유의하세요.")
    else:
      st.info(f"{cache['regime_icon']} 현재 {bench_name} 기준 시장 환경: **{cache['regime_label']}**")

    st.subheader(f"📌 {cache['stock_name']} - 실시간 종합 진단")
    st.markdown(f"**종합 추천 등급: {cache['tier']}**")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("현재가", f"{cache['fmt'].format(cache['curr_price'])} {cache['currency']}")
    m2.metric("퀀트 점수", f"{cache['score']} / 100점")
    m3.metric("14일 ATR (변동폭)", f"{cache['fmt'].format(cache['curr_atr'])}")
    m4.metric("RSI (14)", f"{cache['curr_rsi']:.1f}")
    if cache["is_krx"]:
      m5.metric("거래량 (20일 대비)", f"{cache['vol_ratio']:.1f}%")
    else:
      m5.metric("적용 환율 (원/달러)", f"{exchange_rate:,.1f} 원")

    if cache["warns"]:
      st.warning("⚠️ 유의사항: " + " / ".join(cache["warns"]))

    st.markdown("---")

    st.markdown("### 📱 모바일 최적화 실시간 파동 차트")
    df_chart = cache["df"].tail(90)
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df_chart.index,
            open=df_chart["open"],
            high=df_chart["high"],
            low=df_chart["low"],
            close=df_chart["close"],
            name="주가",
            increasing_line_color="#E53935",
            decreasing_line_color="#1E88E5",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_chart.index,
            y=df_chart["ma20"],
            mode="lines",
            name="20일선",
            line=dict(color="#FF9800", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_chart.index,
            y=df_chart["ma60"],
            mode="lines",
            name="60일선",
            line=dict(color="#4CAF50", width=1.5),
        )
    )

    fig.add_hline(
        y=cache["sl_swing"],
        line_dash="dash",
        line_color="#D32F2F",
        annotation_text="손절(SL)",
        annotation_position="bottom right",
    )
    fig.add_hline(
        y=cache["tp1_swing"],
        line_dash="dash",
        line_color="#2E7D32",
        annotation_text="1차목표",
        annotation_position="top right",
    )
    fig.add_hline(
        y=cache["tp2_swing"],
        line_dash="dash",
        line_color="#1B5E20",
        annotation_text="2차목표",
        annotation_position="top right",
    )

    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=25, b=10),
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10),
        ),
        xaxis=dict(tickformat="%m/%d", tickfont=dict(size=10)),
        yaxis=dict(side="right", tickfont=dict(size=10)),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.markdown("### 🧪 히스토리 백테스트 (동일 로직의 과거 승률 검증)")
    bt_stats = cache["bt_stats"]
    trades_df = cache["trades_df"]

    if bt_stats["trades"] == 0:
      st.info("최근 2년간 해당 임계 점수 이상의 신호가 충분히 발생하지 않아 통계적으로 유의미한 백테스트 결과가 없습니다.")
    else:
      b1, b2, b3, b4 = st.columns(4)
      b1.metric("과거 승률", f"{bt_stats['win_rate']:.1f}%")
      b2.metric("거래 횟수", f"{bt_stats['trades']}회")
      b3.metric("평균 수익률/거래", f"{bt_stats['avg_return']:.2f}%")
      pf = bt_stats["profit_factor"]
      b4.metric("손익비(Profit Factor)", f"{pf:.2f}" if pf != np.inf else "∞")

      with st.expander("개별 거래 내역 보기"):
        st.dataframe(trades_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    st.markdown("### 🎯 매매 전략 및 산출 근거")

    if cache["score"] < 35:
      st.warning("🔴 현재 퀀트 점수가 35점 미만으로 매수 신호로 보기 어렵습니다. 실제 매수 수량은 0으로 제시됩니다.")

    if cache["is_krx"]:
      swing_buy_val = f"약 {cache['qty_swing'] * cache['avg_entry_price']:,.0f} 원"
      swing_risk_val = f"**{cache['max_risk_cash']:,.0f} 원**"
    else:
      swing_buy_val = f"약 ${cache['qty_swing'] * cache['avg_entry_price']:,.2f} (약 {cache['qty_swing'] * cache['avg_entry_price'] * exchange_rate:,.0f} 원)"
      swing_risk_val = f"**${cache['max_risk_cash']:,.2f}** (약 {cache['max_risk_cash'] * exchange_rate:,.0f} 원)"

    ma20_v = cache["ma20_val"]
    ma60_v = cache["ma60_val"]
    is_aligned = ma20_v > ma60_v
    trend_word = "정배열" if is_aligned else "역배열"
    trend_desc = (
        "안정적인 정배열 상승 흐름을 유지하고 있습니다"
        if is_aligned
        else "이동평균선이 역배열 상태라 추세 전환 여부를 주시할 필요가 있습니다"
    )
    trend_bullet = (
        f"**[추세]** {cache['stock_name']}은(는) 현재 {trend_word} 구도로 단기 및 중기 주가가 "
        f"이동평균선(20일: {cache['fmt'].format(ma20_v)}, 60일: {cache['fmt'].format(ma60_v)}) "
        f"{'위에서' if is_aligned else '주변에서'} {trend_desc}."
    )

    rsi_v = cache["curr_rsi"]
    if 40 <= rsi_v <= 65:
      rsi_desc = "과열되지 않은 건전한 수급 상승 영역에 위치해 안정적인 매수 구간"
    elif rsi_v > 75:
      rsi_desc = "단기 과매수 영역에 위치해 있어 추격 매수보다 되돌림 시 분할 진입이 유리한 구간"
    elif rsi_v < 30:
      rsi_desc = "단기 과매도 영역에 위치해 기술적 반등 가능성은 있으나 추세 확인이 필요한 구간"
    else:
      rsi_desc = "뚜렷한 방향성 없이 중립 영역에 위치한 구간"
    momentum_bullet = f"**[모멘텀]** RSI가 {rsi_v:.1f}로 {rsi_desc}입니다."

    vol_v = cache["vol_ratio"]
    if vol_v >= 120:
      supply_bullet = (
          f"**[수급]** 거래량이 20일 평균 대비 {vol_v:.1f}% 수준으로 뚜렷한 돌파형 거래량이 "
          f"동반되어 매수 신뢰도를 높이고 있습니다."
      )
    else:
      supply_bullet = (
          f"**[수급]** 거래량이 20일 평균의 {vol_v:.1f}% 수준으로 돌파형 거래량은 미진하므로, "
          f"지지선 근접 시 지지 여부를 확인하는 분할 접근이 안전합니다."
      )

    sl_pct = (
        (cache["avg_entry_price"] - cache["sl_swing"]) / cache["avg_entry_price"] * 100
        if cache["avg_entry_price"] else 0
    )
    bench_name = "코스피(KOSPI)" if cache["is_krx"] else "S&P500"
    risk_price_bullet = (
        f"**[가격/리스크]** 최근 14일 평균 변동폭(ATR: {cache['fmt'].format(cache['curr_atr'])})을 "
        f"반영하여, 손절가를 평단 대비 {sl_pct:.1f}% 하단"
        f"({cache['fmt'].format(cache['sl_swing'])})으로 정했습니다. "
        f"이는 계좌 자본 위험을 정확히 {risk_pct}%({swing_risk_val}) 이내로 제한하기 위함입니다. "
        f"\\* **[시장 레짐 종합]** 현재 {bench_name} 지수는 **{cache['regime_label']}** 상태로, "
        f"이 시장 환경과 {cache['stock_name']}의 기술적 지표(퀀트 {cache['score']}점)를 종합 계산하여 "
        f"최적의 수량({cache['qty_swing']}주)과 타겟 가격을 산출했습니다."
    )

    with st.expander(
        f"💡 [{cache['stock_name']} ({cache['symbol_formatted']})] 맞춤 분석 — 개별 전략 수립 근거",
        expanded=True,
    ):
      st.markdown(
          "\n".join(
              f"- {b}"
              for b in [trend_bullet, momentum_bullet, supply_bullet, risk_price_bullet]
          )
      )

    with st.container(border=True):
      st.markdown("#### 🏆 전략 어드바이스")
      st.caption("권장 보유기간: 3일 ~ 3주 | 이동평균선 및 ATR 조합 기반 3단계 분할 매수 & 2단계 분할 익절")
      st.markdown(
          f"""
              * **추천 매수 수량:** <span style="font-size:18px; color:#2E7D32; font-weight:bold;">{cache['qty_swing']:,} 주</span> ({swing_buy_val})
              * **최대 허용 손실금:** {swing_risk_val} (전체 자본의 {risk_pct}%)
              * **🎯 정밀 3단계 분할 매수 가이드:**
                  * **1차 진입 (비중 30%):** {cache['fmt'].format(cache['entry1_price'])} {cache['currency']}
                  * **2차 진입 (비중 40%):** <span style="color:#1976D2; font-weight:bold;">{cache['fmt'].format(cache['entry2_price'])} {cache['currency']}</span>
                  * **3차 진입 (비중 30%):** <span style="color:#1976D2; font-weight:bold;">{cache['fmt'].format(cache['entry3_price'])} {cache['currency']}</span>
                  * **💡 예상 체결 평단가:** **{cache['fmt'].format(cache['avg_entry_price'])} {cache['currency']}**
              * **확정 손절가 (SL):** <span style="color:#D32F2F; font-weight:bold;">{cache['fmt'].format(cache['sl_swing'])} {cache['currency']}</span>
              * **목표가 (분할 익절 가이드):**
                  * **1차 목표가 (50% 익절):** <span style="color:#2E7D32; font-weight:bold;">{cache['fmt'].format(cache['tp1_swing'])} {cache['currency']}</span>
                  * **2차 목표가 (50% 익절):** <span style="color:#2E7D32; font-weight:bold;">{cache['fmt'].format(cache['tp2_swing'])} {cache['currency']}</span>
              """,
          unsafe_allow_html=True,
      )

    # ----------------------------------------------------
    # [핵심] Gemini API 연동 실시간 Q&A AI 어드바이저
    # ----------------------------------------------------
    st.markdown("---")
    st.subheader(f"💬 {cache['stock_name']} 실시간 AI 어드바이저 Q&A")
    st.caption("Gemini AI가 현재 종목 데이터와 사용자의 질문을 직접 해석하고 진입가/손절가를 재계산해 드립니다.")

    # 1. 기존 대화 기록 출력
    for message in st.session_state["chat_history"]:
      with st.chat_message(message["role"]):
        st.write(message["content"])

    # 2. 질문 처리 및 Gemini 연동
    if prompt := st.chat_input(
        f"예: {cache['stock_name']} 1차 진입가가 너무 낮아. 현재가 근처로 조정해서 다시 산출해줘"
    ):
      st.session_state["chat_history"].append({"role": "user", "content": prompt})
      with st.chat_message("user"):
        st.write(prompt)

      if not GEMINI_API_KEY:
        advice = "⚠️ Streamlit Secrets에 `GEMINI_API_KEY`가 설정되어 있지 않습니다."
        st.session_state["chat_history"].append(
            {"role": "assistant", "content": advice}
        )
        with st.chat_message("assistant"):
          st.warning(advice)
      else:
        with st.chat_message("assistant"):
          with st.spinner("AI가 질문을 분석하여 맞춤 매매 전략을 재산출 중입니다..."):
            try:
              client = genai.Client(api_key=GEMINI_API_KEY)

              # AI 컨텍스트 생성
              system_prompt = f"""
                            당신은 월가 프롭트레이더 스타일의 전문 주식 분석 AI 어드바이저입니다.
                            현재 분석 중인 종목의 실시간 기술적 지표 데이터는 아래와 같습니다:
                            
                            - 종목명: {cache['stock_name']} ({cache['symbol_formatted']})
                            - 현재가: {cache['curr_price']} {cache['currency']}
                            - 퀀트 점수: {cache['score']}점 / 100점 (추천 등급: {cache['tier']})
                            - 14일 ATR (변동폭): {cache['curr_atr']}
                            - 기존 1차 진입가: {cache['entry1_price']} {cache['currency']}
                            - 기존 2차 진입가: {cache['entry2_price']} {cache['currency']}
                            - 기존 3차 진입가: {cache['entry3_price']} {cache['currency']}
                            - 기존 예상 평단가: {cache['avg_entry_price']} {cache['currency']}
                            - 기존 손절가(SL): {cache['sl_swing']} {cache['currency']}
                            - 1차 목표가: {cache['tp1_swing']}, 2차 목표가: {cache['tp2_swing']}
                            - 보유 수량: {cache['qty_swing']}주 (리스크 금: {cache['max_risk_cash']} {cache['currency']})
                            - RSI: {cache['curr_rsi']:.1f}, 거래량 비율: {cache['vol_ratio']:.1f}%

                            [사용자 질문]: {prompt}

                            [답변 수칙]:
                            1. 사용자가 진입가를 올려달라거나 조정해달라고 요청하는 경우:
                               - 현재가({cache['curr_price']}) 근처(예: 현재가 대비 -0.5% ~ -1.5% 수준)로 1차 진입가를 수정했을 때의 새로운 진입가, 평단가, 손절가, 추천 수량을 즉시 수식 기반으로 재산출하여 대안으로 제시하세요.
                            2. 뻔하거나 반복되는 기계적 문구를 사용하지 말고, 트레이더 관점에서 정밀하게 분석하세요.
                            3. 가독성을 높이기 위해 불렛포인트, bold 강조를 활용하세요.
                            """

              response = client.models.generate_content(
                  model="gemini-3.6-flash",
                  contents=system_prompt,
              )
              advice = response.text

            except Exception as e:
              advice = f"❌ AI 연동 오류: {str(e)}"

            st.markdown(advice)
            st.session_state["chat_history"].append(
                {"role": "assistant", "content": advice}
            )


# ----------------------------------------------------
# 모드 2: 전체 종목 TOP 5 추천 스캔
# ----------------------------------------------------
else:
  st.subheader("🏆 퀀트 스코어 + 히스토리 승률 기준 TOP 5 추천 종목")
  st.info(
      f"현재 환율 **{exchange_rate:,.1f}원/$** 적용 | 총 자본금"
      f" **{capital:,.0f}원** (1회 리스크 {risk_pct}%) 기준 스캔"
  )

  run_scan = st.sidebar.button(
      "🚀 실시간 TOP 5 스캔 실행", type="primary", use_container_width=True
  )

  if run_scan:
    tab_kr, tab_us = st.tabs(["🇰🇷 국내주식 TOP 5", "🇺🇸 해외주식 TOP 5"])

    with tab_kr:
      with st.spinner("국내 19개 종목 실시간 차트, 퀀트 지표 및 백테스트 분석 중..."):
        df_top_kr, regime_label_kr, regime_icon_kr, failed_kr = scan_all_stocks(
            STOCKS_KR, True, exchange_rate, capital, risk_pct, score_threshold
        )
      st.markdown(f"**{regime_icon_kr} 코스피 시장 환경: {regime_label_kr}**")

      if failed_kr and debug_mode:
        st.warning(f"⚠️ {len(failed_kr)}개 종목의 데이터를 불러오지 못했습니다.")

      if not df_top_kr.empty:
        top5_kr = df_top_kr.head(5)
        for idx, row in top5_kr.iterrows():
          with st.container(border=True):
            col1, col2, col3, col4, col5 = st.columns([2, 1.7, 2, 2.2, 3])
            col1.markdown(
                f"### **{idx+1}위. {row['종목명']}**\n`{row['티커']}`\n\n{row['추천등급']}"
            )
            col2.metric("퀀트 점수", f"{row['퀀트점수']}점 / 100")
            win_disp = (
                f"{row['과거승률(%)']}%"
                if row["과거승률(%)"] is not None
                else "데이터부족"
            )
            col2.caption(f"과거승률: {win_disp} ({row['백테스트거래수']}회)")
            col3.metric("현재가", f"{row['현재가']:,.0f} 원")
            col3.caption(f"예상평단: {row['예상평단가']:,.0f} 원")
            col4.metric(
                "1차 목표가",
                f"{row['1차목표가']:,.0f} 원",
                f"+{((row['1차목표가']/row['예상평단가'])-1)*100:.1f}%",
            )
            col4.metric(
                "2차 목표가",
                f"{row['2차목표가']:,.0f} 원",
                f"+{((row['2차목표가']/row['예상평단가'])-1)*100:.1f}%",
            )
            col5.markdown(
                f"**추천 매수 수량:** `{row['추천수량']:,} 주`\n\n"
                f"**손절가 (SL):** `{row['손절가']:,.0f} 원`\n\n"
                f"**신호:** {row['핵심신호']}\n\n"
                f"**유의사항:** {row['유의사항']}"
            )
      else:
        st.info("조건에 맞는 데이터를 가져오지 못했습니다.")

    with tab_us:
      with st.spinner("해외 18개 종목 실시간 차트, 퀀트 지표 및 백테스트 분석 중..."):
        df_top_us, regime_label_us, regime_icon_us, failed_us = scan_all_stocks(
            STOCKS_US, False, exchange_rate, capital, risk_pct, score_threshold
        )
      st.markdown(f"**{regime_icon_us} S&P500 시장 환경: {regime_label_us}**")

      if failed_us and debug_mode:
        st.warning(f"⚠️ {len(failed_us)}개 종목의 데이터를 불러오지 못했습니다.")

      if not df_top_us.empty:
        top5_us = df_top_us.head(5)
        for idx, row in top5_us.iterrows():
          with st.container(border=True):
            col1, col2, col3, col4, col5 = st.columns([2, 1.7, 2, 2.2, 3])
            col1.markdown(
                f"### **{idx+1}위. {row['종목명']}**\n`{row['티커']}`\n\n{row['추천등급']}"
            )
            col2.metric("퀀트 점수", f"{row['퀀트점수']}점 / 100")
            win_disp = (
                f"{row['과거승률(%)']}%"
                if row["과거승률(%)"] is not None
                else "데이터부족"
            )
            col2.caption(f"과거승률: {win_disp} ({row['백테스트거래수']}회)")
            col3.metric("현재가", f"${row['현재가']:,.2f}")
            col3.caption(f"예상평단: ${row['예상평단가']:,.2f}")
            col4.metric(
                "1차 목표가",
                f"${row['1차목표가']:,.2f}",
                f"+{((row['1차목표가']/row['예상평단가'])-1)*100:.1f}%",
            )
            col4.metric(
                "2차 목표가",
                f"${row['2차목표가']:,.2f}",
                f"+{((row['2차목표가']/row['예상평단가'])-1)*100:.1f}%",
            )
            col5.markdown(
                f"**추천 매수 수량:** `{row['추천수량']:,} 주`\n\n"
                f"**손절가 (SL):** `${row['손절가']:,.2f}`\n\n"
                f"**신호:** {row['핵심신호']}\n\n"
                f"**유의사항:** {row['유의사항']}"
            )
      else:
        st.info("조건에 맞는 데이터를 가져오지 못했습니다.")
  else:
    st.info(
        "👈 왼쪽 사이드바의 **[🚀 실시간 TOP 5 스캔 실행]** 버튼을 누르시면 추천"
        " 리스트가 검색됩니다."
    )
