import json
import os
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st
from openai import APIConnectionError, APIStatusError, AuthenticationError, OpenAI, RateLimitError


ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "artifacts/cpi_linear_model.joblib"
DATA_PATH = ROOT / "dataset/processed/monthly_analysis.csv"

MODEL_R2 = 0.9278
MODEL_RMSE = 0.3602


@st.cache_resource
def load_model_bundle():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_analysis_data():
    return pd.read_csv(
        DATA_PATH,
        parse_dates=["date", "target_date"],
    )


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
            "당신은 경제지표 대시보드 해설자입니다. 제공된 JSON 수치만 사용하여 "
            "한국어로 설명하세요. 새로운 예측값을 만들거나 인과관계를 단정하지 마세요. "
            "첫 문장에는 현재 CPI에서 다음 달 전망으로 얼마나 상승하거나 하락하는지 "
            "반드시 %p 단위로 쓰세요. 이어서 최근 추이와 주요 예측 기여 요인을 설명하고, "
            "마지막에는 RMSE가 신뢰구간이 아닌 과거 테스트 오차 참고값임을 밝히세요. "
            "제목 없이 3개의 짧은 글머리표로, 전체 450자 이내로 작성하세요."
        ),
        input=json.dumps(analysis_payload, ensure_ascii=False),
        reasoning={"effort": "low"},
        max_output_tokens=1200,
    )
    return response.output_text


st.set_page_config(
    page_title="Dubai 유가 기반 소비자물가 전망 대시보드",
    page_icon="📈",
    layout="wide",
)

try:
    bundle = load_model_bundle()
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    df = load_analysis_data()
except (FileNotFoundError, KeyError, ValueError) as exc:
    st.error(f"모델 또는 분석 데이터를 불러오지 못했습니다: {exc}")
    st.stop()

available_df = df.dropna(subset=feature_columns)
if available_df.empty:
    st.error("예측에 사용할 수 있는 완전한 월별 데이터가 없습니다.")
    st.stop()

latest = available_df.iloc[-1].copy()
prediction_month = latest["date"] + pd.offsets.MonthBegin(1)

baseline_input = pd.DataFrame([latest[feature_columns]]).astype(float)
baseline_prediction = float(model.predict(baseline_input)[0])

st.title("Dubai 유가 기반 소비자물가 상승률 예측")
st.caption(
    "Dubai 유가·환율·기준금리와 현재 물가 흐름을 이용한 "
    "Linear Regression 예측"
)
st.caption(
    f"최신 관측월: {latest['date']:%Y-%m} · 예측 대상월: "
    f"{prediction_month:%Y-%m} · 데이터 출처: 한국은행 ECOS"
)

with st.container(border=True):
    st.markdown("#### 왜 Dubai 유가로 소비자물가를 예측하나요?")
    st.write(
        "원유 가격의 변화는 에너지·운송·생산비를 거쳐 국내 소비자물가에 영향을 줄 수 "
        "있습니다. 이 서비스는 국내 수입 원유 가격 여건과 연관성이 높은 Dubai 유가를 "
        "활용하고, 그 영향이 시차를 두고 나타날 가능성을 고려해 현재 변동률과 "
        "1·2·3·6개월 시차값을 함께 반영합니다."
    )
    st.caption(
        "물가는 유가만으로 결정되지 않으므로 원/달러 환율, 한국은행 기준금리, 현재의 "
        "물가 흐름도 함께 사용합니다. 결과는 통계적 예측이며 개별 변수의 인과효과를 "
        "의미하지 않습니다."
    )

st.sidebar.header("시나리오 조건")
st.sidebar.caption(
    "값을 바꾸면 최신 전망과 비교한 시나리오 결과가 즉시 계산됩니다. "
    "유가 시차 변수는 최신 관측값으로 고정됩니다."
)

scenario = latest.copy()
scenario["cpi_yoy"] = st.sidebar.number_input(
    "현재 CPI 상승률(%)",
    value=float(latest["cpi_yoy"]),
    step=0.1,
    format="%.2f",
    help="현재 발표된 소비자물가 전년 동월 대비 상승률입니다.",
)
scenario["oil_mom"] = st.sidebar.number_input(
    "Dubai 유가 월간 변동률(%)",
    value=float(latest["oil_mom"]),
    step=1.0,
    format="%.2f",
    help="직전 달 대비 Dubai 유가 변화율입니다.",
)
scenario["fx_mom"] = st.sidebar.number_input(
    "원/달러 환율 월간 변동률(%)",
    value=float(latest["fx_mom"]),
    step=0.5,
    format="%.2f",
    help="양수이면 원/달러 환율 상승(원화 약세)을 뜻합니다.",
)
scenario["base_rate"] = st.sidebar.number_input(
    "한국은행 기준금리(%)",
    value=float(latest["base_rate"]),
    step=0.25,
    format="%.2f",
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
change_vs_baseline = prediction - baseline_prediction

if change_vs_current >= 0.1:
    direction = "상승 압력 확대"
    change_symbol = "↑"
    direction_text = (
        f"다음 달 물가상승률은 현재 입력값보다 "
        f"{change_vs_current:.2f}%p 높아질 전망입니다."
    )
elif change_vs_current <= -0.1:
    direction = "상승세 둔화"
    change_symbol = "↓"
    direction_text = (
        f"다음 달 물가상승률은 현재 입력값보다 "
        f"{abs(change_vs_current):.2f}%p 낮아질 전망입니다."
    )
else:
    direction = "보합권"
    change_symbol = "→"
    direction_text = "다음 달 물가상승률은 현재 입력값과 비슷한 수준으로 전망됩니다."

lower_reference = prediction - MODEL_RMSE
upper_reference = prediction + MODEL_RMSE

scaler = model.named_steps["scaler"]
regression = model.named_steps["model"]
standardized = scaler.transform(input_df)[0]
feature_contributions = pd.Series(
    standardized * regression.coef_,
    index=feature_columns,
)
grouped_contributions = pd.DataFrame(
    {
        "요인": ["현재 물가의 지속성", "Dubai 유가", "환율", "기준금리"],
        "예측 기여도(%p)": [
            feature_contributions["cpi_yoy"],
            feature_contributions[
                ["oil_mom", "oil_lag_1", "oil_lag_2", "oil_lag_3", "oil_lag_6"]
            ].sum(),
            feature_contributions["fx_mom"],
            feature_contributions["base_rate"],
        ],
    }
)

with st.container(border=True):
    st.markdown(
        f"### {prediction_month:%Y-%m} 소비자물가 상승률은 "
        f" **{prediction:.2f}%** 로 전망됩니다"
    )
    current_col, forecast_col, change_col, error_col = st.columns(4)
    current_col.metric("현재 CPI 상승률", f"{current_cpi:.2f}%")
    forecast_col.metric(f"{prediction_month:%Y-%m} 전망", f"{prediction:.2f}%")
    change_col.metric(
        "현재 대비 예상 변화폭",
        f"{change_symbol} {change_vs_current:+.2f}%p",
        help="다음 달 전망에서 현재 CPI 상승률을 뺀 값입니다.",
    )
    error_col.metric(
        "전망 방향",
        direction,
        help=f"현재 대비 ±0.10%p 미만은 보합권으로 분류합니다. 테스트 RMSE는 ±{MODEL_RMSE:.2f}%p입니다.",
    )
    st.caption(
        f"현재 {current_cpi:.2f}% → 다음 달 {prediction:.2f}% · "
        f"예상 변화폭 {change_vs_current:+.2f}%p · 테스트 RMSE ±{MODEL_RMSE:.2f}%p"
    )

st.info(
    f"**한 줄 해석:** {direction_text} "
    f"과거 테스트 RMSE를 단순 참고하면 {lower_reference:.2f}%~"
    f"{upper_reference:.2f}% 범위의 오차 가능성을 함께 봐야 합니다."
)

summary_tab, driver_tab, scenario_tab, detail_tab = st.tabs(
    ["전망 요약", "영향 요인", "시나리오 비교", "데이터·모델 정보"]
)

with summary_tab:
    st.subheader("실제 물가 흐름과 다음 달 전망")

    actual_trend = (
        df[["date", "cpi_yoy"]]
        .dropna()
        .tail(period_months)
        .rename(columns={"cpi_yoy": "실제 CPI 상승률"})
        .set_index("date")
    )
    forecast_trend = pd.DataFrame(
        {
            "date": [latest["date"], prediction_month],
            "다음 달 전망": [float(scenario["cpi_yoy"]), prediction],
        }
    ).set_index("date")
    combined_trend = actual_trend.join(forecast_trend, how="outer")
    st.line_chart(combined_trend)

    st.subheader("GPT 그래프 해석")
    st.caption(
        "GPT는 새로운 물가 예측을 만들지 않고, 현재 화면의 모델 결과와 그래프 수치를 "
        "읽기 쉬운 문장으로 설명합니다. 버튼을 누를 때만 API가 호출됩니다."
    )

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
        "시나리오 입력": {
            "Dubai 유가 월간 변동률(%)": round(float(scenario["oil_mom"]), 2),
            "원/달러 환율 월간 변동률(%)": round(float(scenario["fx_mom"]), 2),
            "한국은행 기준금리(%)": round(float(scenario["base_rate"]), 2),
        },
        "테스트 RMSE(%p)": MODEL_RMSE,
        "주의": "요인 기여도는 예측 설명값이며 인과효과가 아님",
    }
    analysis_signature = json.dumps(
        analysis_payload,
        ensure_ascii=False,
        sort_keys=True,
    )

    if st.button("GPT로 현재 그래프 해석하기", type="primary"):
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
            st.caption("GPT 해석은 참고용이며, 모델의 수치 예측을 변경하지 않습니다.")
        else:
            st.info("입력 조건이 변경되었습니다. 버튼을 눌러 해석을 갱신해 주세요.")

    st.subheader("현재 거시경제 신호")
    signal_rows = [
        {
            "지표": "Dubai 유가 월간 변동",
            "현재값": f"{float(scenario['oil_mom']):+.2f}%",
            "읽는 법": "상승 시 물가 상방, 하락 시 하방 신호",
        },
        {
            "지표": "원/달러 환율 월간 변동",
            "현재값": f"{float(scenario['fx_mom']):+.2f}%",
            "읽는 법": "상승(원화 약세) 시 수입물가 상방 신호",
        },
        {
            "지표": "한국은행 기준금리",
            "현재값": f"{float(scenario['base_rate']):.2f}%",
            "읽는 법": "모델의 보조 변수이며 인과효과로 해석하지 않음",
        },
    ]
    st.dataframe(pd.DataFrame(signal_rows), hide_index=True, width="stretch")

with driver_tab:
    st.subheader("이번 전망을 구성한 요인")
    st.caption(
        "각 값은 학습기간 평균 조건과 비교한 예측 기여도입니다. "
        "양수는 전망을 높이고 음수는 낮추는 방향이며, 인과효과를 의미하지 않습니다."
    )

    contribution_chart = (
        alt.Chart(grouped_contributions)
        .mark_bar()
        .encode(
            x=alt.X(
                "요인:N",
                sort=None,
                axis=alt.Axis(title=None, labelAngle=0, labelLimit=160),
            ),
            y=alt.Y("예측 기여도(%p):Q", title="예측 기여도(%p)"),
            tooltip=[
                alt.Tooltip("요인:N"),
                alt.Tooltip("예측 기여도(%p):Q", format="+.3f"),
            ],
        )
    )
    st.altair_chart(contribution_chart, width="stretch")

    strongest = grouped_contributions.iloc[
        grouped_contributions["예측 기여도(%p)"].abs().argmax()
    ]
    st.write(
        f"가장 큰 예측 요인은 **{strongest['요인']}**이며, 학습기간 평균 조건 대비 "
        f"전망을 **{strongest['예측 기여도(%p)']:+.2f}%p** 움직이는 방향입니다."
    )
    st.dataframe(
        grouped_contributions.style.format({"예측 기여도(%p)": "{:+.3f}"}),
        hide_index=True,
        width="stretch",
    )

with scenario_tab:
    st.subheader("최신 조건과 사용자 시나리오 비교")
    base_col, scenario_col, difference_col = st.columns(3)
    base_col.metric("최신 데이터 기준 전망", f"{baseline_prediction:.2f}%")
    scenario_col.metric("입력 시나리오 전망", f"{prediction:.2f}%")
    difference_col.metric(
        "시나리오 효과",
        f"{change_vs_baseline:+.2f}%p",
        help="현재 사이드바 입력과 최신 데이터 기본값의 예측 차이입니다.",
    )

    if abs(change_vs_baseline) < 0.01:
        st.write("사이드바 값이 최신 데이터와 같아 기본 전망과 차이가 없습니다.")
    elif change_vs_baseline > 0:
        st.warning(
            f"입력한 조건은 최신 조건보다 다음 달 물가 전망을 "
            f"{change_vs_baseline:.2f}%p 높입니다."
        )
    else:
        st.success(
            f"입력한 조건은 최신 조건보다 다음 달 물가 전망을 "
            f"{abs(change_vs_baseline):.2f}%p 낮춥니다."
        )

    st.caption(
        "활용 예: 유가 급등·환율 상승·금리 변화 가정을 입력해 기본 전망 대비 "
        "물가 전망이 얼마나 달라지는지 비교합니다."
    )

with detail_tab:
    performance_col, freshness_col = st.columns(2)
    with performance_col:
        st.subheader("모델 성능")
        st.table(
            pd.DataFrame(
                {
                    "지표": ["R²", "RMSE", "MAE"],
                    "2022~2025 테스트": [f"{MODEL_R2:.4f}", "0.3602", "0.3036"],
                }
            )
        )
    with freshness_col:
        st.subheader("데이터 범위")
        st.write(f"- 원천 데이터: **{df['date'].min():%Y-%m}~{df['date'].max():%Y-%m}**")
        st.write(f"- 최신 완전 관측월: **{latest['date']:%Y-%m}**")
        st.write(f"- 예측 대상월: **{prediction_month:%Y-%m}**")
        st.write("- 출처: **한국은행 ECOS**")

    st.subheader("최근 12개월 데이터")
    recent_columns = ["date", "cpi_yoy", "dubai_price", "oil_mom", "usd_krw", "fx_mom", "base_rate"]
    st.dataframe(
        df[recent_columns].tail(12),
        hide_index=True,
        width="stretch",
    )

    with st.expander("모델 입력값과 유가 시차 변수"):
        st.dataframe(input_df, hide_index=True, width="stretch")
        st.caption(
            "유가 시차 변수는 최신 관측값을 사용합니다. 모델 결과는 예측 참고자료이며 "
            "정책·투자 판단의 단독 근거로 사용하지 않습니다."
        )
