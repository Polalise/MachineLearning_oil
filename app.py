from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent

bundle = joblib.load(
    ROOT / "artifacts/cpi_linear_model.joblib"
)
model = bundle["model"]
feature_columns = bundle["feature_columns"]

df = pd.read_csv(
    ROOT / "dataset/processed/monthly_analysis.csv",
    parse_dates=["date", "target_date"],
)

latest = df.dropna(subset=feature_columns).iloc[-1].copy()
prediction_month = latest["date"] + pd.offsets.MonthBegin(1)

st.set_page_config(
    page_title="소비자물가 예측",
    layout="wide",
)

st.title("Dubai 유가 기반 소비자물가 상승률 예측")
st.caption("Linear Regression 기반 다음 달 CPI 상승률 예측")

st.sidebar.header("예측 조건")

latest["cpi_yoy"] = st.sidebar.number_input(
    "현재 CPI 상승률(%)",
    value=float(latest["cpi_yoy"]),
)
latest["oil_mom"] = st.sidebar.number_input(
    "Dubai 유가 월간 변동률(%)",
    value=float(latest["oil_mom"]),
)
latest["fx_mom"] = st.sidebar.number_input(
    "원/달러 환율 월간 변동률(%)",
    value=float(latest["fx_mom"]),
)
latest["base_rate"] = st.sidebar.number_input(
    "한국은행 기준금리(%)",
    value=float(latest["base_rate"]),
)

input_df = pd.DataFrame([
    latest[feature_columns]
])

prediction = model.predict(input_df)[0]

col1, col2, col3 = st.columns(3)

col1.metric(
    "예측 대상 월",
    prediction_month.strftime("%Y-%m"),
)
col2.metric(
    "예상 CPI 상승률",
    f"{prediction:.2f}%",
)
col3.metric(
    "최종 모델 R²",
    "0.9278",
)

st.subheader("소비자물가 상승률 추이")
st.line_chart(
    df.set_index("date")[["cpi_yoy"]]
)

st.subheader("Dubai 유가 추이")
st.line_chart(
    df.set_index("date")[["dubai_price"]]
)

with st.expander("모델 입력 데이터"):
    st.dataframe(input_df)