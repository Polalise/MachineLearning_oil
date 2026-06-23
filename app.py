import base64
import json
import os
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "artifacts/cpi_linear_model.joblib"
DATA_PATH = ROOT / "dataset/processed/monthly_analysis.csv"
ICON_DIR = ROOT / "assets" / "icons"

TEST_START_DATE = "2022-01-01"
TEST_END_DATE = "2025-12-01"


@st.cache_resource
def load_model_bundle():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_analysis_data():
    return pd.read_csv(
        DATA_PATH,
        parse_dates=["date", "target_date"],
    )


@st.cache_data
def compute_model_metrics(data, feature_columns):
    metric_df = data.dropna(
        subset=feature_columns + ["target_cpi_yoy"]
    ).copy()
    metric_df = metric_df[
        metric_df["target_date"] <= TEST_END_DATE
    ].reset_index(drop=True)

    train_df = metric_df[metric_df["target_date"] < TEST_START_DATE].copy()
    test_df = metric_df[metric_df["target_date"] >= TEST_START_DATE].copy()
    y_train = train_df["target_cpi_yoy"]
    y_test = test_df["target_cpi_yoy"]

    fx_feature = "fx_log_mom" if "fx_log_mom" in feature_columns else "fx_mom"
    ablation_features = {
        "CPI 단독 모델": ["cpi_yoy"],
        "CPI + 환율 + 기준금리": ["cpi_yoy", fx_feature, "base_rate"],
        "전체 모델": feature_columns,
    }

    rows = []
    for model_name, columns in ablation_features.items():
        candidate_model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ])
        candidate_model.fit(train_df[columns], y_train)
        pred = candidate_model.predict(test_df[columns])
        rows.append({
            "모델": model_name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": mean_squared_error(y_test, pred) ** 0.5,
            "R2": r2_score(y_test, pred),
        })

    metrics = pd.DataFrame(rows)
    cpi_only_rmse = float(
        metrics.loc[metrics["모델"] == "CPI 단독 모델", "RMSE"].iloc[0]
    )
    metrics["RMSE 개선폭"] = cpi_only_rmse - metrics["RMSE"]
    metrics["RMSE 개선율"] = metrics["RMSE 개선폭"] / cpi_only_rmse * 100
    return metrics


def get_openai_api_key():
    try:
        return st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    except FileNotFoundError:
        return os.getenv("OPENAI_API_KEY")


def generate_gpt_interpretation(api_key, analysis_payload):
    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model="gpt-5-mini",
        instructions=(
            "당신은 경제지표 대시보드 해설자입니다. 제공된 JSON 값만 사용해 "
            "한국어로 설명하세요. 새로운 예측값을 만들거나 인과관계를 단정하지 마세요. "
            "첫 문장에는 현재 CPI에서 다음 달 전망으로 얼마나 상승 또는 하락하는지 "
            "반드시 %p 단위로 쓰세요. 최근 추이와 주요 예측 기여 요인을 설명하고, "
            "마지막에는 RMSE가 신뢰구간이 아닌 과거 테스트 오차 참고값임을 밝히세요. "
            "제목 없이 3개의 짧은 문단으로, 전체 450자 이내로 작성하세요."
        ),
        input=json.dumps(analysis_payload, ensure_ascii=False),
        reasoning={"effort": "low"},
        max_output_tokens=1200,
    )
    return response.output_text


def pct_text(value):
    if value > 0:
        return f"▲ +{value:.2f}%"
    if value < 0:
        return f"▼ {value:.2f}%"
    return f"{value:.2f}%"


def pp_text(value):
    if value > 0:
        return f"▲ +{value:.2f}%p"
    if value < 0:
        return f"▼ {value:.2f}%p"
    return f"{value:.2f}%p"


def kpi_icon(name, tone="blue"):
    stroke = "#145cff" if tone == "blue" else "#ef3340"
    soft = "#eaf2ff" if tone == "blue" else "#fff0f2"
    icon_files = {
        "chart": "cpi.svg",
        "oil": "oil.svg",
        "fx": "fx.svg",
        "bank": "rate.svg",
        "shield": "rmse.svg",
    }
    icon_path = ICON_DIR / icon_files[name]
    if icon_path.exists():
        encoded = base64.b64encode(icon_path.read_bytes()).decode("ascii")
        return (
            f'<div class="kpi-icon kpi-icon-file" '
            f'style="--icon-stroke:{stroke}; --icon-soft:{soft};">'
            f'<img src="data:image/svg+xml;base64,{encoded}" alt="" /></div>'
        )

    icons = {
        "chart": """
            <svg viewBox="0 0 48 48" aria-hidden="true">
                <rect x="10" y="25" width="5" height="13" rx="2"></rect>
                <rect x="20" y="18" width="5" height="20" rx="2"></rect>
                <rect x="30" y="11" width="5" height="27" rx="2"></rect>
                <path d="M10 20 L19 14 L25 17 L36 7"></path>
                <path d="M32 7 H36 V11"></path>
            </svg>
        """,
        "oil": """
            <svg viewBox="0 0 48 48" aria-hidden="true">
                <ellipse class="oil-barrel" cx="20" cy="10" rx="13" ry="5"></ellipse>
                <path class="oil-barrel" d="M7 10 V34 C7 39 13 43 20 43 C24 43 27 42 30 40 V16 C27 19 23 20 20 20 C13 20 7 15 7 10 Z"></path>
                <path class="oil-cut" d="M8 22 C12 26 27 27 32 22"></path>
                <path class="oil-drop" d="M35 17 C31 24 27 30 27 36 C27 43 32 47 38 47 C44 47 48 43 48 36 C48 30 42 23 35 17 Z"></path>
                <path class="oil-shine" d="M34 35 C34 39 36 41 39 42"></path>
            </svg>
        """,
        "fx": """
            <svg viewBox="0 0 48 48" aria-hidden="true">
                <path d="M24 9 V39"></path>
                <path d="M16 31 C18 36 31 36 33 30 C35 24 16 24 18 17 C20 11 31 12 33 17"></path>
            </svg>
        """,
        "bank": """
            <svg viewBox="0 0 48 48" aria-hidden="true">
                <path d="M8 19 H40 L24 9 Z"></path>
                <path d="M12 22 V36"></path>
                <path d="M20 22 V36"></path>
                <path d="M28 22 V36"></path>
                <path d="M36 22 V36"></path>
                <path d="M9 39 H39"></path>
            </svg>
        """,
        "shield": """
            <svg viewBox="0 0 48 48" aria-hidden="true">
                <path d="M24 7 L38 12 V23 C38 32 32 39 24 42 C16 39 10 32 10 23 V12 Z"></path>
                <path d="M24 18 V29"></path>
                <path d="M24 35 H24.1"></path>
            </svg>
        """,
    }
    svg = icons[name]
    return (
        f'<div class="kpi-icon" style="--icon-stroke:{stroke}; --icon-soft:{soft};">'
        f"{svg}</div>"
    )


def kpi_card(title, value, sub="", tone="", icon=""):
    tone_class = "kpi-up" if tone == "up" else "kpi-down" if tone == "down" else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            {icon}
            <div class="kpi-title">{title}</div>
            <div class="kpi-value {tone_class}">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def reset_scenario(latest, oil_current_feature, fx_current_feature):
    st.session_state["cpi_yoy_input"] = float(latest["cpi_yoy"])
    st.session_state["oil_input"] = float(latest[oil_current_feature])
    st.session_state["fx_input"] = float(latest[fx_current_feature])
    st.session_state["base_rate_input"] = float(latest["base_rate"])
    st.session_state["scenario_preset"] = "기준 시나리오"


def apply_scenario_preset(preset_name, latest, oil_current_feature, fx_current_feature):
    reset_scenario(latest, oil_current_feature, fx_current_feature)

    if preset_name == "유가 급등":
        st.session_state["oil_input"] = min(float(latest[oil_current_feature]) + 8.0, 30.0)
    elif preset_name == "환율 상승":
        st.session_state["fx_input"] = min(float(latest[fx_current_feature]) + 2.0, 10.0)
    elif preset_name == "금리 인상":
        st.session_state["base_rate_input"] = min(float(latest["base_rate"]) + 0.5, 6.0)
    elif preset_name == "복합 충격":
        st.session_state["oil_input"] = min(float(latest[oil_current_feature]) + 8.0, 30.0)
        st.session_state["fx_input"] = min(float(latest[fx_current_feature]) + 2.0, 10.0)
    elif preset_name == "완화 시나리오":
        st.session_state["oil_input"] = max(float(latest[oil_current_feature]) - 3.0, -30.0)
        st.session_state["fx_input"] = max(float(latest[fx_current_feature]) - 1.0, -10.0)
        st.session_state["base_rate_input"] = max(float(latest["base_rate"]) - 0.25, 0.0)

    st.session_state["scenario_preset"] = preset_name


def mark_manual_scenario():
    st.session_state["scenario_preset"] = "사용자 조정"


st.set_page_config(
    page_title="두바이 유가 기반 소비자물가 상승률 예측",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --navy: #0b1736;
        --blue: #145cff;
        --blue-2: #0d47d9;
        --soft-blue: #eaf2ff;
        --line: #d8e4f7;
        --muted: #64748b;
        --red: #ef3340;
        --green: #059669;
    }
    html, body {
        min-width: 1660px;
        overflow-x: auto;
    }
    .stApp {
        min-width: 1660px;
        background: #eef5ff;
        color: var(--navy);
    }
    div[data-testid="stAppViewContainer"] {
        min-width: 1660px;
    }
    div[data-testid="stHorizontalBlock"] {
        flex-wrap: nowrap !important;
    }
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, .82);
        border-right: 1px solid #d8e4f7;
        box-shadow: 8px 0 28px rgba(18, 52, 104, .08);
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 1.2rem;
    }
    .block-container {
        width: 1320px;
        min-width: 1320px;
        max-width: 1320px;
        padding-top: 3.2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: var(--navy);
        letter-spacing: 0;
    }
    h1 {
        font-size: 36px !important;
        line-height: 1.18 !important;
        margin: 15px 0 27px 0 !important;
        padding: 0 !important;
    }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,.86);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 18px 18px;
        box-shadow: 0 10px 26px rgba(18, 52, 104, .08);
    }
    div[data-testid="stMetric"] label {
        color: #15254c !important;
        font-weight: 750;
    }
    div[data-testid="stMetricValue"] {
        color: var(--navy);
        font-weight: 850;
    }
    .hero-card {
        background: linear-gradient(135deg, #0c4fe8 0%, #0032b8 100%);
        border-radius: 18px;
        padding: 18px 26px 17px 26px;
        min-height: 238px;
        color: white;
        box-shadow: 0 18px 40px rgba(15, 82, 220, .26);
        border: 1px solid rgba(255,255,255,.24);
    }
    .hero-label {
        font-size: .95rem;
        font-weight: 750;
        opacity: .95;
    }
    .hero-value {
        font-size: 52px;
        line-height: 1;
        font-weight: 900;
        letter-spacing: 0;
        margin: 11px 0 9px 0;
        white-space: nowrap;
    }
    .hero-pill {
        display: inline-block;
        background: rgba(255,255,255,.98);
        color: var(--red);
        font-weight: 850;
        border-radius: 999px;
        padding: 6px 12px;
        margin-left: 10px;
        vertical-align: middle;
        font-size: .88rem;
    }
    .direction-badge {
        display: inline-block;
        background: linear-gradient(90deg, #ff4d66, #d91f3f);
        border-radius: 999px;
        color: white;
        padding: 6px 22px;
        font-weight: 800;
        margin: 0 0 15px 0;
    }
    .range-head {
        display: flex;
        align-items: center;
        gap: 14px;
        color: rgba(255,255,255,.95);
        font-size: .82rem;
        font-weight: 850;
        margin-bottom: 9px;
    }
    .range-head strong {
        color: #ffffff;
    }
    .range-head span {
        color: rgba(255,255,255,.94);
    }
    .range-row {
        display: grid;
        grid-template-columns: 1fr 68px;
        gap: 16px;
        align-items: center;
        color: rgba(255,255,255,.95);
        font-size: .8rem;
        font-weight: 800;
        margin-top: 0;
    }
    .range-track {
        position: relative;
        height: 8px;
        border-radius: 999px;
        background: rgba(255,255,255,.32);
        overflow: hidden;
        box-shadow: 0 10px 20px rgba(0, 39, 160, .18);
    }
    .range-track::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 50%;
        background: linear-gradient(90deg, #ff99aa, #ffffff);
        opacity: .82;
    }
    .insight {
        border: 1px solid #bad6ff;
        background: #ffffff;
        color: #0b3fbb;
        border-radius: 14px;
        padding: 20px 22px;
        margin-top: 28px;
        margin-bottom: 18px;
        font-weight: 720;
        box-shadow: 0 8px 22px rgba(18, 52, 104, .08);
    }
    .panel {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 20px 22px;
        box-shadow: 0 10px 28px rgba(18, 52, 104, .08);
    }
    .panel-title {
        font-weight: 850;
        color: var(--navy);
        margin-bottom: 8px;
        font-size: 1.05rem;
    }
    .small-muted {
        color: var(--muted);
        font-size: .88rem;
    }
    .driver-row {
        display: grid;
        grid-template-columns: 95px 1fr 78px;
        gap: 12px;
        align-items: center;
        border-bottom: 1px solid #eef3fb;
        padding: 10px 0;
    }
    .driver-row:last-child {
        border-bottom: 0;
    }
    .bar-bg {
        height: 10px;
        border-radius: 999px;
        background: #e6edf7;
        overflow: hidden;
    }
    .bar-pos, .bar-neg {
        height: 100%;
        border-radius: 999px;
    }
    .bar-pos {
        background: linear-gradient(90deg, #ff95a3, #ef3340);
    }
    .bar-neg {
        background: linear-gradient(90deg, #2d7ff9, #7bb7ff);
    }
    .top-meta {
        text-align: right;
        color: #274069;
        font-size: .82rem;
        padding-top: 3px;
        line-height: 1.4;
    }
    .kpi-card {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 16px 10px;
        min-height: 238px;
        text-align: center;
        box-shadow: 0 10px 26px rgba(18, 52, 104, .08);
        display: flex;
        flex-direction: column;
        justify-content: center;
        gap: 10px;
    }
    .kpi-icon {
        width: 52px;
        height: 52px;
        border-radius: 50%;
        margin: 0 auto 2px auto;
        display: flex;
        align-items: center;
        justify-content: center;
        background: var(--icon-soft);
        border: 1px solid #d7e6ff;
        box-shadow: 0 8px 18px rgba(20, 92, 255, .12);
    }
    .kpi-icon svg {
        width: 31px;
        height: 31px;
        fill: none;
        stroke: var(--icon-stroke);
        stroke-width: 3.5;
        stroke-linecap: round;
        stroke-linejoin: round;
    }
    .kpi-icon-file img {
        width: 32px;
        height: 32px;
        object-fit: contain;
        display: block;
    }
    .kpi-icon rect {
        fill: var(--icon-stroke);
        stroke: none;
    }
    .kpi-icon .oil-barrel {
        fill: #202020;
        stroke: #202020;
        stroke-width: 1.8;
    }
    .kpi-icon .oil-cut {
        fill: none;
        stroke: #ffffff;
        stroke-width: 3.2;
    }
    .kpi-icon .oil-drop {
        fill: #e3a126;
        stroke: #ffffff;
        stroke-width: 2.6;
    }
    .kpi-icon .oil-shine {
        fill: none;
        stroke: #ffffff;
        stroke-width: 3.6;
    }
    .kpi-title {
        font-size: .88rem;
        font-weight: 800;
        color: #15254c;
        white-space: normal;
    }
    .kpi-value {
        font-size: 28px;
        font-weight: 900;
        color: var(--navy);
        line-height: 1;
        white-space: nowrap;
    }
    .kpi-sub {
        color: var(--muted);
        font-size: .82rem;
        font-weight: 650;
    }
    .st-key-summary_chart_box,
    .st-key-summary_driver_box,
    .st-key-ai_interpretation_box,
    .st-key-driver_detail_box,
    .st-key-scenario_compare_box,
    .st-key-reliability_box,
    .st-key-detail_info_box,
    .st-key-detail_data_box {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 18px 20px;
        box-sizing: border-box;
        box-shadow: 0 10px 28px rgba(18, 52, 104, .08);
    }
    .st-key-summary_chart_box,
    .st-key-summary_driver_box {
        min-height: 420px;
    }
    .kpi-up {
        color: var(--red);
    }
    .kpi-down {
        color: var(--blue);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
        border-bottom: 1px solid #d8e4f7;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 750;
        color: #23345c;
    }
    .stTabs [aria-selected="true"] {
        color: var(--blue);
        border-bottom-color: var(--blue);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    bundle = load_model_bundle()
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    df = load_analysis_data()
    metrics_df = compute_model_metrics(df, feature_columns)
except (FileNotFoundError, KeyError, ValueError) as exc:
    st.error(f"모델 또는 분석 데이터를 불러오지 못했습니다. {exc}")
    st.stop()

full_metrics = metrics_df.loc[metrics_df["모델"] == "전체 모델"].iloc[0]
cpi_only_metrics = metrics_df.loc[metrics_df["모델"] == "CPI 단독 모델"].iloc[0]

model_r2 = float(full_metrics["R2"])
model_rmse = float(full_metrics["RMSE"])
model_mae = float(full_metrics["MAE"])
cpi_only_rmse = float(cpi_only_metrics["RMSE"])
rmse_improvement = float(full_metrics["RMSE 개선폭"])
rmse_improvement_rate = float(full_metrics["RMSE 개선율"])

available_df = df.dropna(subset=feature_columns)
if available_df.empty:
    st.error("예측에 사용할 수 있는 완전한 월별 데이터가 없습니다.")
    st.stop()

latest = available_df.iloc[-1].copy()
prediction_month = latest["date"] + pd.offsets.MonthBegin(1)

oil_current_feature = "oil_log_mom" if "oil_log_mom" in feature_columns else "oil_mom"
fx_current_feature = "fx_log_mom" if "fx_log_mom" in feature_columns else "fx_mom"
oil_lag_features = [
    column for column in [
        "oil_log_mom",
        "oil_log_lag_1",
        "oil_log_lag_2",
        "oil_log_lag_3",
        "oil_log_lag_6",
        "oil_mom",
        "oil_lag_1",
        "oil_lag_2",
        "oil_lag_3",
        "oil_lag_6",
    ]
    if column in feature_columns
]

if "cpi_yoy_input" not in st.session_state:
    reset_scenario(latest, oil_current_feature, fx_current_feature)

st.sidebar.markdown("### 시나리오 조건")
st.sidebar.caption("값을 변경하면 예측 결과가 즉시 반영됩니다.")
st.sidebar.divider()

st.sidebar.markdown("#### 시나리오 프리셋")
st.sidebar.caption(f"현재 선택: {st.session_state.get('scenario_preset', '기준 시나리오')}")
row1_col1, row1_col2 = st.sidebar.columns(2)
with row1_col1:
    if st.button("기준 시나리오", width="stretch"):
        apply_scenario_preset("기준 시나리오", latest, oil_current_feature, fx_current_feature)
        st.rerun()
with row1_col2:
    if st.button("유가 급등", width="stretch"):
        apply_scenario_preset("유가 급등", latest, oil_current_feature, fx_current_feature)
        st.rerun()

row2_col1, row2_col2 = st.sidebar.columns(2)
with row2_col1:
    if st.button("환율 상승", width="stretch"):
        apply_scenario_preset("환율 상승", latest, oil_current_feature, fx_current_feature)
        st.rerun()
with row2_col2:
    if st.button("금리 인상", width="stretch"):
        apply_scenario_preset("금리 인상", latest, oil_current_feature, fx_current_feature)
        st.rerun()

row3_col1, row3_col2 = st.sidebar.columns(2)
with row3_col1:
    if st.button("완화 시나리오", width="stretch"):
        apply_scenario_preset("완화 시나리오", latest, oil_current_feature, fx_current_feature)
        st.rerun()
with row3_col2:
    if st.button("복합 충격", width="stretch"):
        apply_scenario_preset("복합 충격", latest, oil_current_feature, fx_current_feature)
        st.rerun()

st.sidebar.divider()

scenario = latest.copy()
scenario["cpi_yoy"] = st.sidebar.slider(
    "현재 CPI 상승률 (%)",
    min_value=-2.0,
    max_value=8.0,
    value=float(st.session_state["cpi_yoy_input"]),
    step=0.01,
    key="cpi_yoy_input",
    on_change=mark_manual_scenario,
)
scenario[oil_current_feature] = st.sidebar.slider(
    "두바이 유가 월간 로그 변동률 (%)",
    min_value=-30.0,
    max_value=30.0,
    value=float(st.session_state["oil_input"]),
    step=0.01,
    key="oil_input",
    on_change=mark_manual_scenario,
)
scenario[fx_current_feature] = st.sidebar.slider(
    "원/달러 환율 월간 로그 변동률 (%)",
    min_value=-10.0,
    max_value=10.0,
    value=float(st.session_state["fx_input"]),
    step=0.01,
    key="fx_input",
    on_change=mark_manual_scenario,
)
scenario["base_rate"] = st.sidebar.slider(
    "한국은행 기준금리 (%)",
    min_value=0.0,
    max_value=6.0,
    value=float(st.session_state["base_rate_input"]),
    step=0.25,
    key="base_rate_input",
    on_change=mark_manual_scenario,
)

period_label = st.sidebar.selectbox(
    "추이 조회 기간",
    ["최근 5년", "최근 10년", "전체"],
)
period_months = {
    "최근 5년": 60,
    "최근 10년": 120,
    "전체": len(df),
}[period_label]

input_df = pd.DataFrame([scenario[feature_columns]]).astype(float)
prediction = float(model.predict(input_df)[0])
current_cpi = float(scenario["cpi_yoy"])
change_vs_current = prediction - current_cpi
change_vs_baseline = prediction - float(model.predict(pd.DataFrame([latest[feature_columns]]).astype(float))[0])
lower_reference = prediction - model_rmse
upper_reference = prediction + model_rmse

if change_vs_current >= 0.10:
    direction = "상승 압력 확대"
    direction_class = "up"
    direction_text = (
        f"다음 달 물가상승률은 현재 입력값보다 {change_vs_current:.2f}%p "
        "높아질 전망입니다."
    )
elif change_vs_current <= -0.10:
    direction = "상승세 둔화"
    direction_class = "down"
    direction_text = (
        f"다음 달 물가상승률은 현재 입력값보다 {abs(change_vs_current):.2f}%p "
        "낮아질 전망입니다."
    )
else:
    direction = "이전 달 수준 유지"
    direction_class = "flat"
    direction_text = "다음 달 물가상승률은 현재 입력값과 비슷한 수준으로 전망됩니다."

scaler = model.named_steps["scaler"]
regression = model.named_steps["model"]
standardized = scaler.transform(input_df)[0]
feature_contributions = pd.Series(
    standardized * regression.coef_,
    index=feature_columns,
)
grouped_contributions = pd.DataFrame(
    {
        "요인": ["두바이 유가", "원/달러 환율", "기준금리", "현재 CPI"],
        "예측 기여도(%p)": [
            feature_contributions[oil_lag_features].sum(),
            feature_contributions[fx_current_feature],
            feature_contributions["base_rate"],
            feature_contributions["cpi_yoy"],
        ],
        "현재값": [
            f"{scenario[oil_current_feature]:+.2f}%",
            f"{scenario[fx_current_feature]:+.2f}%",
            f"{scenario['base_rate']:.2f}%",
            f"{scenario['cpi_yoy']:.2f}%",
        ],
    }
)

max_abs_contribution = max(
    grouped_contributions["예측 기여도(%p)"].abs().max(),
    0.001,
)

header_cols = st.columns([2.6, 1])
with header_cols[0]:
    st.markdown("# 두바이 유가 기반 소비자물가 상승률 예측")
    st.caption("국제유가, 환율, 기준금리를 반영한 다음 달 CPI 상승률 예측 서비스")
with header_cols[1]:
    st.markdown(
        f"""
        <div class="top-meta">
            최신 관측월: <b>{latest['date']:%Y-%m}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

hero_col, m1, m2, m3, m4, m5 = st.columns([2.1, 1, 1, 1, 1, 1])
with hero_col:
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-label">다음 달 CPI 전망 ({prediction_month:%Y-%m})</div>
            <div class="hero-value">
                {prediction:.2f}% <span class="hero-pill">{pp_text(change_vs_current)}</span>
            </div>
            <div class="direction-badge">{direction}</div>
            <div class="range-head">
                <strong>예측 오차 참고</strong>
                <span>참고 범위 {lower_reference:.2f}% ~ {upper_reference:.2f}%</span>
            </div>
            <div class="range-row">
                <div class="range-track"></div>
                <span>{model_rmse:.2f}%p</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
oil_now = float(scenario[oil_current_feature])
fx_now = float(scenario[fx_current_feature])

with m1:
    kpi_card("현재 CPI 상승률", f"{current_cpi:.2f}%", f"{latest['date']:%Y-%m} 기준", icon=kpi_icon("chart"))
with m2:
    kpi_card("두바이 유가 변동", pct_text(oil_now), "하락 요인" if oil_now < 0 else "상승 요인", "down" if oil_now < 0 else "up", kpi_icon("oil"))
with m3:
    kpi_card("원/달러 환율 변동", pct_text(fx_now), "상승 요인" if fx_now > 0 else "하락 요인", "up" if fx_now > 0 else "down", kpi_icon("fx", "red"))
with m4:
    kpi_card("기준금리", f"{float(scenario['base_rate']):.2f}%", "현재 수준", icon=kpi_icon("bank"))
with m5:
    kpi_card("테스트 RMSE", f"±{model_rmse:.2f}%p", "모델 예측 오차", icon=kpi_icon("shield"))

st.markdown(
    f"""
    <div class="insight">
        한 줄 해석: {direction_text} 과거 테스트 RMSE를 단순 참고하면
        {lower_reference:.2f}%~{upper_reference:.2f}% 범위의 오차 가능성을 함께 봐야 합니다.
    </div>
    """,
    unsafe_allow_html=True,
)

summary_tab, driver_tab, scenario_tab, reliability_tab, detail_tab = st.tabs(
    ["전망 요약", "영향 요인", "시나리오 비교", "모델 신뢰도", "데이터·모델 정보"]
)

with summary_tab:
    chart_col, driver_col = st.columns([1.75, 1])
    with chart_col:
        with st.container(border=True, key="summary_chart_box"):
            st.markdown('<div class="panel-title">실제 CPI 흐름과 다음 달 전망</div>', unsafe_allow_html=True)

            actual_chart_df = (
                df[["date", "cpi_yoy"]]
                .dropna()
                .tail(period_months)
                .rename(columns={"date": "월", "cpi_yoy": "CPI 상승률"})
            )
            forecast_chart_df = pd.DataFrame(
                {
                    "월": [latest["date"], prediction_month],
                    "예측값": [float(scenario["cpi_yoy"]), prediction],
                }
            )
            band_df = pd.DataFrame({
                "월": [prediction_month],
                "하단": [lower_reference],
                "상단": [upper_reference],
                "예측값": [prediction],
            })

            actual_line = alt.Chart(actual_chart_df).mark_line(
                color="#145cff",
                strokeWidth=3,
            ).encode(
                x=alt.X("월:T", title=None),
                y=alt.Y("CPI 상승률:Q", title="CPI 상승률(%)"),
                tooltip=[
                    alt.Tooltip("월:T", title="월", format="%Y-%m"),
                    alt.Tooltip("CPI 상승률:Q", title="실제값", format=".2f"),
                ],
            )
            forecast_line = alt.Chart(forecast_chart_df).mark_line(
                color="#ef3340",
                strokeDash=[6, 4],
                strokeWidth=3,
            ).encode(
                x="월:T",
                y=alt.Y("예측값:Q"),
                tooltip=[
                    alt.Tooltip("월:T", title="월", format="%Y-%m"),
                    alt.Tooltip("예측값:Q", title="예측값", format=".2f"),
                ],
            )
            chart = (actual_line + forecast_line).properties(
                height=330,
            )
            st.altair_chart(chart, width="stretch")

    with driver_col:
        with st.container(border=True, key="summary_driver_box"):
            st.markdown('<div class="panel-title">주요 영향 요인</div>', unsafe_allow_html=True)
            for _, row in grouped_contributions.iterrows():
                value = float(row["예측 기여도(%p)"])
                width = min(abs(value) / max_abs_contribution * 100, 100)
                bar_class = "bar-pos" if value >= 0 else "bar-neg"
                label = "상승 요인" if value >= 0 else "하락 요인"
                st.markdown(
                    f"""
                    <div class="driver-row">
                        <div>
                            <b>{row["요인"]}</b><br>
                            <span class="small-muted">{row["현재값"]}</span>
                        </div>
                        <div class="bar-bg"><div class="{bar_class}" style="width:{width:.1f}%"></div></div>
                        <div style="text-align:right; color:{'#ef3340' if value >= 0 else '#145cff'};">
                            <b>{label}</b><br>{value:+.2f}%p
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown(
                f"""
                <div style="padding-top:16px; font-size:1.1rem; font-weight:850; text-align:right;">
                    총 예상 변화 <span style="color:#ef3340;">{change_vs_current:+.2f}%p</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    ai_box = st.container(border=True, key="ai_interpretation_box")
    with ai_box:
        st.markdown('<div class="panel-title">AI 그래프 해석</div>', unsafe_allow_html=True)
        recent_cpi = [
            {"월": row.date.strftime("%Y-%m"), "CPI 상승률(%)": round(row.cpi_yoy, 2)}
            for row in df[["date", "cpi_yoy"]].dropna().tail(6).itertuples()
        ]
        analysis_payload = {
            "최신 관측월": latest["date"].strftime("%Y-%m"),
            "예측 대상월": prediction_month.strftime("%Y-%m"),
            "현재 CPI 상승률(%)": round(current_cpi, 2),
            "다음 달 전망(%)": round(prediction, 2),
            "현재 대비 변화폭(%p)": round(change_vs_current, 2),
            "전망 방향": direction,
            "최근 6개월 CPI": recent_cpi,
            "예측 요인 기여도(%p)": {
                row["요인"]: round(float(row["예측 기여도(%p)"]), 3)
                for _, row in grouped_contributions.iterrows()
            },
            "테스트 RMSE(%p)": model_rmse,
            "주의": "요인 기여도는 예측 설명값이며 인과효과가 아님",
        }
        analysis_signature = json.dumps(
            analysis_payload,
            ensure_ascii=False,
            sort_keys=True,
        )

        st.write(
            "최근 CPI 흐름, 예측값, 주요 요인 기여도를 바탕으로 모델 결과를 자연어로 요약합니다. "
            "새로운 예측값을 생성하지 않고 현재 모델 결과만 해석합니다."
        )
        if st.button("AI로 해석해서 보기", type="primary"):
            api_key = get_openai_api_key()
            if not api_key:
                st.error(".streamlit/secrets.toml에 OPENAI_API_KEY를 등록해 주세요.")
            else:
                try:
                    with st.spinner("현재 전망과 그래프를 해석하고 있습니다..."):
                        interpretation = generate_gpt_interpretation(
                            api_key,
                            analysis_payload,
                        )
                    st.session_state["gpt_interpretation"] = {
                        "signature": analysis_signature,
                        "text": interpretation,
                    }
                except AuthenticationError:
                    st.error("OpenAI API 키를 확인해 주세요.")
                except RateLimitError:
                    st.error("API 사용 한도 또는 결제 설정을 확인해 주세요.")
                except APIConnectionError:
                    st.error("OpenAI API에 연결하지 못했습니다. 네트워크를 확인해 주세요.")
                except APIStatusError:
                    st.error("OpenAI API 요청을 처리하지 못했습니다. 잠시 후 다시 시도해 주세요.")

        saved_interpretation = st.session_state.get("gpt_interpretation")
        if saved_interpretation:
            if saved_interpretation["signature"] == analysis_signature:
                st.markdown(saved_interpretation["text"])
            else:
                st.info("입력 조건이 변경되었습니다. 버튼을 눌러 해석을 갱신해 주세요.")

with driver_tab:
    with st.container(border=True, key="driver_detail_box"):
        st.markdown('<div class="panel-title">이번 전망을 구성한 요인</div>', unsafe_allow_html=True)
        contribution_chart = (
            alt.Chart(grouped_contributions)
            .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
            .encode(
                x=alt.X("요인:N", sort=None, axis=alt.Axis(title=None, labelAngle=0)),
                y=alt.Y("예측 기여도(%p):Q", title="예측 기여도(%p)"),
                color=alt.condition(
                    alt.datum["예측 기여도(%p)"] >= 0,
                    alt.value("#ef3340"),
                    alt.value("#145cff"),
                ),
                tooltip=[
                    alt.Tooltip("요인:N"),
                    alt.Tooltip("현재값:N"),
                    alt.Tooltip("예측 기여도(%p):Q", format="+.3f"),
                ],
            )
            .properties(height=360)
        )
        st.altair_chart(contribution_chart, width="stretch")
        st.caption("기여도는 학습 기간 평균 조건과 비교한 예측 설명값이며, 인과효과를 의미하지 않습니다.")
        st.dataframe(
            grouped_contributions.style.format({"예측 기여도(%p)": "{:+.3f}"}),
            hide_index=True,
            width="stretch",
        )

with scenario_tab:
    with st.container(border=True, key="scenario_compare_box"):
        st.markdown('<div class="panel-title">최신 조건과 사용자 시나리오 비교</div>', unsafe_allow_html=True)
        base_prediction = float(model.predict(pd.DataFrame([latest[feature_columns]]).astype(float))[0])
        c1, c2, c3 = st.columns(3)
        c1.metric("최신 데이터 기준 전망", f"{base_prediction:.2f}%")
        c2.metric("입력 시나리오 전망", f"{prediction:.2f}%")
        c3.metric("시나리오 효과", f"{change_vs_baseline:+.2f}%p")
        if abs(change_vs_baseline) < 0.01:
            st.info("입력 조건이 최신 데이터와 거의 같아 기본 전망과 차이가 작습니다.")
        elif change_vs_baseline > 0:
            st.warning(f"입력 조건은 최신 조건보다 다음 달 물가 전망을 {change_vs_baseline:.2f}%p 높입니다.")
        else:
            st.success(f"입력 조건은 최신 조건보다 다음 달 물가 전망을 {abs(change_vs_baseline):.2f}%p 낮춥니다.")

with reliability_tab:
    with st.container(border=True, key="reliability_box"):
        st.markdown('<div class="panel-title">모델 신뢰도와 해석 유의점</div>', unsafe_allow_html=True)
        st.write(
            f"2022~2025년 테스트 기준 RMSE는 {model_rmse:.4f}%p입니다. "
            "이는 신뢰구간이 아니라 과거 테스트 기간에서 관측된 예측 오차의 참고값입니다."
        )
        reliability_df = pd.DataFrame(
            {
                "지표": ["R2", "RMSE", "MAE", "CPI 단독 RMSE", "전체 모델 RMSE", "RMSE 개선폭", "RMSE 개선율"],
                "값": [
                    f"{model_r2:.4f}",
                    f"{model_rmse:.4f}",
                    f"{model_mae:.4f}",
                    f"{cpi_only_rmse:.4f}",
                    f"{model_rmse:.4f}",
                    f"{rmse_improvement:.4f}%p",
                    f"{rmse_improvement_rate:.1f}%",
                ],
            }
        )
        st.dataframe(reliability_df, hide_index=True, width="stretch")

with detail_tab:
    info_col, data_col = st.columns([1, 1.25])
    with info_col:
        with st.container(border=True, key="detail_info_box"):
            st.markdown('<div class="panel-title">데이터·모델 정보</div>', unsafe_allow_html=True)
            st.write(f"- 원천 데이터 기간: **{df['date'].min():%Y-%m} ~ {df['date'].max():%Y-%m}**")
            st.write(f"- 최신 관측월: **{latest['date']:%Y-%m}**")
            st.write(f"- 예측 대상월: **{prediction_month:%Y-%m}**")
            st.write("- 출처: **한국은행 ECOS**")
            st.write("- 최종 모델: **StandardScaler + Linear Regression**")
    with data_col:
        with st.container(border=True, key="detail_data_box"):
            st.markdown('<div class="panel-title">최근 12개월 데이터</div>', unsafe_allow_html=True)
            recent_columns = [
                "date",
                "cpi_yoy",
                "dubai_price",
                oil_current_feature,
                "usd_krw",
                fx_current_feature,
                "base_rate",
            ]
            recent_labels = {
                "date": "월",
                "cpi_yoy": "CPI 상승률",
                "dubai_price": "두바이 유가",
                oil_current_feature: "유가 로그 변동률",
                "usd_krw": "원/달러 환율",
                fx_current_feature: "환율 로그 변동률",
                "base_rate": "기준금리",
            }
            st.dataframe(
                df[recent_columns].tail(12).rename(columns=recent_labels),
                hide_index=True,
                width="stretch",
            )
            with st.expander("모델 입력값 보기"):
                st.dataframe(input_df, hide_index=True, width="stretch")
