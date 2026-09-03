import os
import tempfile
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import ta
import yfinance as yf

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
  """일시적 오류/차단에 대비해 짧은 대기 후 재시도. 마지막 예외 메시지를 함께 반환."""
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

st.title("⚡ 월가 & 코스피 프롭트레이딩 정밀 분석 및 TOP 5 추천 시스템")
st.caption(
    "실시간 파동, 변동성(ATR), 수급 모멘텀, 시장 레짐 및 히스토리 백테스트 기반 매매 전략"
)
st.markdown("---")

# 국내 주식 19개 종목 사전
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


# 실시간 환율 수집 함수
@st.cache_data(ttl=3600, show_spinner=False)
def get_usdkrw_rate():
  _obj, df, err = _fetch_history_with_retry("USDKRW=X", period="5d")
  if not df.empty:
    rate = float(df["Close"].dropna().iloc[-1])
    if rate > 0:
      return rate, None
  return 1350.0, err


# ----------------------------------------------------
# 시장 레짐(강세/약세) 판단 - 벤치마크 지수 추세
# ----------------------------------------------------
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


# 지표 계산 및 NaN 정제 함수
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


# ----------------------------------------------------
# 퀀트 점수 계산 함수
# ----------------------------------------------------
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
      warnings.append("추세 미약(횡보장) - 신호 신뢰도 낮음")

  rsi = curr.get("rsi", np.nan)
  if not pd.isna(rsi):
    if 40 <= rsi <= 65:
      score += 15
      reasons.append("RSI 건전한 상승모멘텀")
    elif rsi > 75:
      warnings.append(f"RSI 과매수 경고({rsi:.0f})")
    elif rsi < 30:
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
    warnings.append("52주 신고가 근접 - 단기 변동성 유의")

  return min(score, 100), reasons, warnings


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


# ----------------------------------------------------
# 히스토리 백테스트
# ----------------------------------------------------
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


# 주식 데이터 수집 함수
@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data(ticker):
  if not ticker:
    return pd.DataFrame(), ticker, ticker, False, "티커가 비어 있습니다."

  is_krx = ticker.isdigit() or ticker.endswith(".KS") or ticker.endswith(".KQ")
  last_err = None

  if ticker.isdigit():
    for suffix in [".KS", ".KQ"]:
      sym = f"{ticker}{suffix}"
      obj, df, err = _fetch_history_with_retry(sym, period="2y")
      if not df.empty and len(df) >= 60:
        name = sym
        try:
          info = obj.info if hasattr(obj, "info") else {}
          name = info.get("shortName", info.get("longName", sym))
        except Exception:
          pass
        return calculate_indicators(df), name, sym, True, None
      last_err = err
    return pd.DataFrame(), ticker, ticker, True, last_err

  obj, df, err = _fetch_history_with_retry(ticker, period="2y")
  if not df.empty and len(df) >= 60:
    name = ticker
    try:
      info = obj.info if hasattr(obj, "info") else {}
      name = info.get("shortName", info.get("longName", ticker))
    except Exception:
      pass
    return calculate_indicators(df), name, ticker, is_krx, None

  return pd.DataFrame(), ticker, ticker, is_krx, err


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

    df, s_name, sym, _, err = fetch_stock_data(ticker)
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

    # ----------------------------------------------------
    # 3단계 분할 매수 산출 로직 (30%:40%:30%)
    # ----------------------------------------------------
    ma5_val = (
        float(curr["ma5"]) if not pd.isna(curr["ma5"]) else curr_price
    )
    ma20_val = (
        float(curr["ma20"]) if not pd.isna(curr["ma20"]) else curr_price * 0.98
    )
    ma60_val = (
        float(curr["ma60"]) if not pd.isna(curr["ma60"]) else curr_price * 0.95
    )

    # 1차 진입가: Min(5일선, 현재가 - 0.5 * ATR)
    e1 = min(ma5_val, curr_price - 0.5 * curr_atr)
    # 2차 진입가: Min(1차 진입가 - 1.0 * ATR, 20일선)
    e2 = min(e1 - 1.0 * curr_atr, ma20_val)
    # 3차 진입가: Min(1차 진입가 - 2.0 * ATR, 60일선)
    e3 = min(e1 - 2.0 * curr_atr, ma60_val)

    avg_price = (e1 * 0.3) + (e2 * 0.4) + (e3 * 0.3)

    stop_dist = curr_atr * 1.5
    sl_price = avg_price - stop_dist
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
        "손익비": (
            round(bt_stats["profit_factor"], 2)
            if bt_stats["profit_factor"] not in (None, np.inf)
            else bt_stats["profit_factor"]
        ),
    })

  progress_bar.empty()
  res_df = pd.DataFrame(results)
  if not res_df.empty:
    res_df["_win_sort"] = res_df["과거승률(%)"].fillna(-1)
    res_df = (
        res_df.sort_values(
            by=["퀀트점수", "_win_sort"], ascending=[False, False]
        )
        .drop(columns="_win_sort")
        .reset_index(drop=True)
    )
  return res_df, regime_label, regime_icon, failed


# 2. 사이드바 - 분석 모드 및 파라미터 설정
st.sidebar.header("⚙️ 트레이딩 분석 설정")

app_mode = st.sidebar.radio(
    "🎯 실행 모드 선택",
    ["🔍 선택 종목 개별 정밀 분석", "🚀 전체 종목 TOP 5 스캔"],
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
    help=(
        "1회 거래당 계좌의 몇 %까지 손실을 허용할지 정합니다. 값이 클수록 추천"
        " 매수 수량이 늘어나지만 손실 위험도 커집니다."
    ),
)

with st.sidebar.expander("🔧 고급 설정 (백테스트 신호 기준)"):
  score_threshold = st.slider(
      "백테스트/추천 신호 최소 점수",
      min_value=40,
      max_value=90,
      value=60,
      step=5,
      help=(
          "이 점수 이상일 때만 '매수 신호'로 간주하고 과거 승률을"
          " 검증합니다."
      ),
  )
  debug_mode = st.checkbox(
      "🐞 디버그 모드 (데이터 수집 실패 원인 표시)",
      value=False,
      help=(
          "클라우드 배포 환경 등에서 데이터를 못 불러올 때, 실제 오류 메시지를"
          " 화면에 표시합니다."
      ),
  )

exchange_rate, fx_err = get_usdkrw_rate()
if fx_err and debug_mode:
  st.sidebar.caption(f"⚠️ 환율 조회 실패(기본값 1350원 사용): {fx_err}")


# ----------------------------------------------------
# 모드 1: 선택 종목 개별 정밀 분석
# ----------------------------------------------------
if app_mode == "🔍 선택 종목 개별 정밀 분석":
  st.sidebar.markdown("---")
  market_type = st.sidebar.radio(
      "🌐 시장 선택", ["국내주식 (KRX)", "해외주식 (US)"], index=0, horizontal=True
  )

  selected_ticker = "005930"

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
    else:
      selected_ticker = STOCKS_US[selected_name]
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
    else:
      selected_ticker = STOCKS_KR[selected_name]

  run_analysis = st.sidebar.button(
      "🚀 정밀 분석 & 전략 생성", type="primary", use_container_width=True
  )

  if run_analysis or "analyzed" not in st.session_state:
    st.session_state["analyzed"] = True

    df, stock_name, symbol_formatted, is_krx, fetch_err = fetch_stock_data(
        selected_ticker
    )

    if df.empty or len(df) < 60:
      st.error(
          f"❌ [{selected_ticker}] 종목 데이터를 불러올 수 없습니다. 종목 코드나"
          " 데이터 수집 상태를 확인하세요."
      )
      if fetch_err:
        st.caption(f"오류 상세: {fetch_err}")
    else:
      idx_df, idx_err = fetch_index_trend(is_krx)
      regime_label, regime_icon, regime_score = get_market_regime(idx_df)
      if idx_err and debug_mode:
        st.caption(f"⚠️ 지수(벤치마크) 데이터 조회 실패: {idx_err}")
      df = add_relative_strength(df, idx_df)

      curr = df.iloc[-1]
      curr_price = float(curr["close"])
      curr_atr = (
          float(curr["atr"]) if not pd.isna(curr["atr"]) else curr_price * 0.02
      )
      curr_rsi = float(curr["rsi"]) if not pd.isna(curr["rsi"]) else 50.0
      vol_ratio = (
          float(curr["vol_ratio"]) if not pd.isna(curr["vol_ratio"]) else 100.0
      )

      score, reasons, warns = compute_quant_score(curr)
      tier = get_recommendation_tier(score, regime_score)

      currency = "KRW (원)" if is_krx else "USD ($)"
      fmt = "{:,.0f}" if is_krx else "{:,.2f}"

      bench_name = "코스피(KOSPI)" if is_krx else "S&P500"
      if regime_score > 0:
        st.success(
            f"{regime_icon} 현재 {bench_name} 기준 시장 환경:"
            f" **{regime_label}** — 추세 매매에 우호적인 환경입니다."
        )
      elif regime_score < 0:
        st.error(
            f"{regime_icon} 현재 {bench_name} 기준 시장 환경:"
            f" **{regime_label}** — 개별 종목 신호가 좋아도 전체 시장 역풍에"
            " 유의하세요."
        )
      else:
        st.info(
            f"{regime_icon} 현재 {bench_name} 기준 시장 환경:"
            f" **{regime_label}**"
        )

      st.subheader(f"📌 {stock_name} ({symbol_formatted}) - 실시간 종합 진단")
      st.markdown(f"**종합 추천 등급: {tier}**")

      m1, m2, m3, m4, m5 = st.columns(5)
      m1.metric("현재가", f"{fmt.format(curr_price)} {currency}")
      m2.metric("퀀트 점수", f"{score} / 100점")
      m3.metric("14일 ATR (변동폭)", f"{fmt.format(curr_atr)}")
      m4.metric("RSI (14)", f"{curr_rsi:.1f}")
      if is_krx:
        m5.metric("거래량 (20일 대비)", f"{vol_ratio:.1f}%")
      else:
        m5.metric("적용 환율 (원/달러)", f"{exchange_rate:,.1f} 원")

      if warns:
        st.warning("⚠️ 유의사항: " + " / ".join(warns))

      st.markdown("---")

      if is_krx:
        capital_curr = capital
      else:
        capital_curr = capital / exchange_rate
      max_risk_cash = capital_curr * (risk_pct / 100.0)

      # ----------------------------------------------------
      # 3단계 분할 매수 계산 (스윙 전략)
      # ----------------------------------------------------
      ma5_val = (
          float(curr["ma5"]) if not pd.isna(curr["ma5"]) else curr_price
      )
      ma20_val = (
          float(curr["ma20"]) if not pd.isna(curr["ma20"]) else curr_price * 0.98
      )
      ma60_val = (
          float(curr["ma60"]) if not pd.isna(curr["ma60"]) else curr_price * 0.95
      )

      # 1차 진입가: Min(5일선, 현재가 - 0.5 * ATR)
      entry1_price = min(ma5_val, curr_price - 0.5 * curr_atr)
      # 2차 진입가: Min(1차 진입가 - 1.0 * ATR, 20일선)
      entry2_price = min(entry1_price - 1.0 * curr_atr, ma20_val)
      # 3차 진입가: Min(1차 진입가 - 2.0 * ATR, 60일선)
      entry3_price = min(entry1_price - 2.0 * curr_atr, ma60_val)

      # 30% : 40% : 30% 비율 반영 평균 단가
      avg_entry_price = (
          (entry1_price * 0.3) + (entry2_price * 0.4) + (entry3_price * 0.3)
      )

      stop_dist_swing = curr_atr * 1.5
      sl_swing = avg_entry_price - stop_dist_swing
      tp1_swing = avg_entry_price + (stop_dist_swing * 1.5)
      tp2_swing = avg_entry_price + (stop_dist_swing * 2.5)

      max_affordable_qty_swing = (
          int(capital_curr // avg_entry_price) if avg_entry_price > 0 else 0
      )
      raw_qty_swing = (
          int(max_risk_cash / stop_dist_swing) if stop_dist_swing > 0 else 0
      )
      qty_swing = min(raw_qty_swing, max_affordable_qty_swing)

      if score < 35:
        qty_swing = 0

      weak_signal = score < 35

      def zero_qty_reason(raw_qty, stop_dist, max_aff):
        if max_aff == 0:
          return (
              f"보유 자본(약 {fmt.format(capital_curr)} {currency})으로 매수"
              " 가이드 가격의 1주도 구매할 수 없습니다."
          )
        if raw_qty == 0:
          return (
              f"1회 리스크 한도(약 {fmt.format(max_risk_cash)} {currency})가"
              " 이 전략의 손절폭보다 작아 리스크 관리 규칙상 0주가"
              " 산출되었습니다."
          )
        return None

      swing_zero_reason = (
          zero_qty_reason(
              raw_qty_swing, stop_dist_swing, max_affordable_qty_swing
          )
          if qty_swing == 0
          else None
      )

      st.markdown("### 📱 모바일 최적화 실시간 파동 차트")

      # 모바일 최적화: 최근 90일 데이터 사용
      df_chart = df.tail(90)

      fig = go.Figure()

      # 1. 캔들스틱
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

      # 2. 이동평균선
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

      # 3. 목표가 및 손절가 수평선
      fig.add_hline(
          y=sl_swing,
          line_dash="dash",
          line_color="#D32F2F",
          annotation_text="손절(SL)",
          annotation_position="bottom right",
      )
      fig.add_hline(
          y=tp1_swing,
          line_dash="dash",
          line_color="#2E7D32",
          annotation_text="1차목표",
          annotation_position="top right",
      )
      fig.add_hline(
          y=tp2_swing,
          line_dash="dash",
          line_color="#1B5E20",
          annotation_text="2차목표",
          annotation_position="top right",
      )

      # 모바일 호환성 100% 검증 레이아웃 설정 (ValueError 원인 완벽 제거)
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
      trades_df, bt_stats = backtest_strategy(
          df, score_threshold=score_threshold
      )

      if bt_stats["trades"] == 0:
        st.info(
            "최근 2년간 해당 임계 점수 이상의 신호가 충분히 발생하지 않아"
            " 통계적으로 유의미한 백테스트 결과가 없습니다."
        )
      else:
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("과거 승률", f"{bt_stats['win_rate']:.1f}%")
        b2.metric("거래 횟수", f"{bt_stats['trades']}회")
        b3.metric("평균 수익률/거래", f"{bt_stats['avg_return']:.2f}%")
        pf = bt_stats["profit_factor"]
        b4.metric(
            "손익비(Profit Factor)", f"{pf:.2f}" if pf != np.inf else "∞"
        )

        with st.expander("개별 거래 내역 보기"):
          st.dataframe(trades_df, use_container_width=True, hide_index=True)

      st.markdown("---")

      st.markdown("### 🎯 실전 트레이딩 전략 및 산출 근거")

      if weak_signal:
        st.warning(
            "🔴 현재 퀀트 점수가 35점 미만으로 매수 신호로 보기 어렵습니다. 아래"
            " 수치는 참고용 시뮬레이션이며, 실제 매수 수량은 0으로 제시됩니다."
        )

      if is_krx:
        cap_desc = f"총 자본금({capital:,.0f}원)"
        risk_desc = f"{max_risk_cash:,.0f}원"
        swing_buy_val = f"약 {qty_swing * avg_entry_price:,.0f} 원"
        swing_risk_val = f"**{max_risk_cash:,.0f} 원**"
      else:
        cap_desc = (
            f"총 자본금({capital:,.0f}원 / 약 ${capital_curr:,.2f}, 적용환율:"
            f" {exchange_rate:,.1f}원/$)"
        )
        risk_desc = (
            f"${max_risk_cash:,.2f} (약 {max_risk_cash * exchange_rate:,.0f}원)"
        )
        swing_buy_val = (
            f"약 ${qty_swing * avg_entry_price:,.2f} (약"
            f" {qty_swing * avg_entry_price * exchange_rate:,.0f} 원)"
        )
        swing_risk_val = (
            f"**${max_risk_cash:,.2f}** (약"
            f" {max_risk_cash * exchange_rate:,.0f} 원)"
        )

      # 선택된 종목의 지표에 따라 진단 근거 동적 작성
      dynamic_reasons = []

      # 1. 이동평균 및 추세 상태
      if curr_price > ma20_val and ma20_val > ma60_val:
        dynamic_reasons.append(
            f"**[추세]** {stock_name}은(는) 현재 정배열 구도로 단기 및 중기"
            f" 주가가 이동평균선(20일: {fmt.format(ma20_val)}, 60일:"
            f" {fmt.format(ma60_val)}) 위에서 안정적인 정배열 상승 흐름을"
            " 유지하고 있습니다."
        )
      elif curr_price > ma20_val:
        dynamic_reasons.append(
            f"**[추세]** 60일선 대비 상단 반등 중이며, 현재 20일선({fmt.format(ma20_val)})을"
            " 상회하여 단기 반등 모멘텀이 살아있습니다."
        )
      else:
        dynamic_reasons.append(
            f"**[추세]** 현재 주가가 20일 이동평균선({fmt.format(ma20_val)}) 밑에"
            " 위치해 있어 성급한 추격 매수보다는 눌림목 분할 진입이"
            " 유효합니다."
        )

      # 2. RSI 및 모멘텀 상태
      if curr_rsi >= 70:
        dynamic_reasons.append(
            f"**[모멘텀]** RSI가 {curr_rsi:.1f}로 과매수 구간에 진입해 있어 1차"
            f" 진입가({fmt.format(entry1_price)}) 수준까지의 단기 숨고르기를"
            " 기다리는 전략을 권장합니다."
        )
      elif 40 <= curr_rsi < 70:
        dynamic_reasons.append(
            f"**[모멘텀]** RSI가 {curr_rsi:.1f}로과열되지 않은 건전한 수급 상승"
            " 영역에 위치해 안정적인 매수 구간입니다."
        )
      else:
        dynamic_reasons.append(
            f"**[모멘텀]** RSI가 {curr_rsi:.1f}로 침체 구간에 위치하여"
            " 반등을 겨냥한 보수적 분할 매수가 타당합니다."
        )

      # 3. 거래량 수급 상태
      if vol_ratio >= 120:
        dynamic_reasons.append(
            f"**[수급]** 최근 거래량이 20일 평균 대비 {vol_ratio:.1f}% 증가하며"
            " 강한 매수 세력 유입 신호가 포착되었습니다."
        )
      else:
        dynamic_reasons.append(
            f"**[수급]** 거래량이 20일 평균의 {vol_ratio:.1f}% 수준으로 돌파형"
            " 거래량은 미진하므로, 지지선 근접 시 지지 여부를 확인하는 분할"
            " 접근이 안전합니다."
        )

      # 4. ATR 및 진입/리스크 설정 이유
      dynamic_reasons.append(
          f"**[가격/리스크]** 최근 14일 평균 변동폭(ATR: {fmt.format(curr_atr)})을"
          f" 반영하여, 손절가를 평단 대비 {stop_dist_swing:,.1f}"
          f" 하단({fmt.format(sl_swing)})으로 정했습니다. 이는 노이즈에 의해서"
          " 손절되지 않으면서도 계좌 자본 위험을 정확히"
          f" **{risk_pct}%({risk_desc})** 이내로 제한하기 위함입니다."
      )

      # 동적 근거 텍스트 변환
      reason_bullets = "\n".join([f"* {r}" for r in dynamic_reasons])

      with st.expander(
          f"💡 **[{stock_name} 맞춤 분석] 개별 전략 수립 근거**",
          expanded=True,
      ):
        st.markdown(f"""
                {reason_bullets}
                * **[시장 레짐 종합]** 현재 {bench_name} 지수는 **{regime_label}** 상태로, 이 시장 환경과 {stock_name}의 기술적 지표(퀀트 {score}점)를 종합 계산하여 최적의 수량({qty_swing}주)과 타겟 가격을 산출했습니다.
                """)

      # 스윙 전략 카드
      with st.container(border=True):
        st.markdown("#### 🏆 [스윙 트레이딩] 중기 파동 전략")
        st.caption(
            "권장 보유기간: 3일 ~ 3주 | 이동평균선 및 ATR 조합 기반 3단계 분할 매수 & 2단계 분할 익절"
        )
        st.markdown(
            f"""
                * **추천 매수 수량:** <span style="font-size:18px; color:#2E7D32; font-weight:bold;">{qty_swing:,} 주</span> ({swing_buy_val})
                * **최대 허용 손실금:** {swing_risk_val} (전체 자본의 {risk_pct}%)
                * **🎯 정밀 3단계 분할 매수 가이드:**
                    * **1차 진입 (비중 30% - 단기 눌림목):** {fmt.format(entry1_price)} {currency} `Min(5일선, 현재가-0.5*ATR)`
                    * **2차 진입 (비중 40% - 중기 추세 지지):** <span style="color:#1976D2; font-weight:bold;">{fmt.format(entry2_price)} {currency}</span> `Min(1차진입가-1.0*ATR, 20일선)`
                    * **3차 진입 (비중 30% - 강한 하한 지지):** <span style="color:#1976D2; font-weight:bold;">{fmt.format(entry3_price)} {currency}</span> `Min(1차진입가-2.0*ATR, 60일선)`
                    * **💡 예상 체결 평단가:** **{fmt.format(avg_entry_price)} {currency}**
                * **확정 손절가 (SL):** <span style="color:#D32F2F; font-weight:bold;">{fmt.format(sl_swing)} {currency}</span> (평단 대비 -{(stop_dist_swing/avg_entry_price)*100:.2f}%)
                * **목표가 (분할 익절 가이드):**
                    * **1차 목표가 (50% 익절):** <span style="color:#2E7D32; font-weight:bold;">{fmt.format(tp1_swing)} {currency}</span> (평단 대비 +{((tp1_swing/avg_entry_price)-1)*100:.2f}%)
                    * **2차 목표가 (50% 익절):** <span style="color:#2E7D32; font-weight:bold;">{fmt.format(tp2_swing)} {currency}</span> (평단 대비 +{((tp2_swing/avg_entry_price)-1)*100:.2f}%)
                """,
            unsafe_allow_html=True,
        )
        if swing_zero_reason:
          st.info(f"ℹ️ 추천 수량이 0주인 이유: {swing_zero_reason}")


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
