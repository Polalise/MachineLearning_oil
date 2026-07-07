# MachineLearning Oil

두바이 유가, 원/달러 환율, 기준금리 등 거시경제 변수를 활용해 다음 달 소비자물가 상승률을 예측하고 Streamlit으로 시각화하는 머신러닝 프로젝트입니다.

## 한줄 소개

한국은행 ECOS 월별 데이터를 기반으로 Dubai 유가와 환율 변동을 반영해 다음 달 CPI 상승률을 예측하는 머신러닝 대시보드

## 프로젝트 개요

이 프로젝트는 국제유가와 환율 변화가 국내 소비자물가에 시차를 두고 반영될 수 있다는 가정에서 출발합니다. 한국은행 ECOS에서 제공하는 월별 경제통계 데이터를 수집·정제하고, 소비자물가지수(CPI), Dubai 유가, 원/달러 환율, 기준금리를 활용해 다음 달 소비자물가 전년 동월 대비 상승률을 예측합니다.

분석 노트북에서는 기준 모델과 여러 머신러닝 회귀 모델을 비교하고, 최종 모델을 학습해 `joblib` 산출물로 저장합니다. `app.py`는 저장된 모델과 가공 데이터를 불러와 Streamlit 기반 예측 대시보드를 제공합니다.

## 주요 기능

- 경제 데이터 전처리
  - 한국은행 ECOS 기반 월별 경제지표 CSV 활용
  - CPI, Dubai 유가, 원/달러 환율, 기준금리 데이터 결합
  - 월별 시계열 형태로 정리
  - 전년 동월 대비 CPI 상승률 생성
  - 유가 및 환율 월간 변동률, 로그 변동률 생성
  - 유가 시차 변수 생성

- 머신러닝 모델링
  - 다음 달 CPI 상승률(`target_cpi_yoy`) 예측
  - 지속성 기준 모델과 회귀 모델 비교
  - Linear Regression, Ridge, Random Forest, XGBoost 등 비교
  - MAE, RMSE, R² 기준 평가
  - 최종 모델 저장

- Streamlit 예측 서비스
  - 최신 관측월 기준 다음 달 CPI 상승률 예측
  - 실제 CPI 흐름과 예측값 시각화
  - 주요 변수별 예측 기여도 표시
  - 유가 급등, 환율 상승, 금리 인상, 완화, 복합 충격 시나리오 제공
  - 사용자가 직접 CPI, 유가, 환율, 기준금리 조건 조정 가능
  - 모델 신뢰도와 테스트 오차 지표 표시
  - 최근 12개월 데이터 및 모델 입력값 확인

- AI 해석 기능
  - OpenAI API 키가 설정된 경우 예측 결과와 그래프를 자연어로 요약
  - 예측값, 주요 기여 요인, RMSE 참고값을 한국어로 설명
  - 모델 결과 해석용 기능이며 새로운 예측값을 생성하지 않도록 설계

## 기술 스택

### Analysis / ML

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- joblib
- Jupyter Notebook

### Visualization / App

- Streamlit
- Altair
- Matplotlib
- Seaborn

### Optional AI Interpretation

- OpenAI API

## 프로젝트 구조

```text
MachineLearning_oil/
├── app.py                              # Streamlit 예측 대시보드
├── model.ipynb                         # 데이터 전처리, 모델 학습, 평가 노트북
├── requirements.txt                    # Python 의존성 목록
├── README.md
├── artifacts/
│   └── cpi_linear_model.joblib          # 학습된 최종 모델 및 입력 변수 목록
├── assets/
│   └── icons/                           # Streamlit KPI 아이콘
├── dataset/
│   ├── BankOfKoreaInterestRate.csv      # 한국은행 기준금리
│   ├── ConsumerPriceIndex.csv           # 소비자물가지수
│   ├── DollarExchangeRate.csv           # 원/달러 환율
│   ├── DubaiOilPrice.csv                # Dubai 유가
│   └── processed/
│       └── monthly_analysis.csv         # 모델 학습/서비스용 가공 데이터
└── report/
    └── 머신러닝 서비스모델 과제_유병현.pdf
```

## 데이터 설명

| 데이터 | 설명 | 활용 |
| --- | --- | --- |
| 소비자물가지수 | CPI 총지수 | CPI 상승률 및 예측 대상 생성 |
| Dubai 유가 | 국제상품가격 중 Dubai 유가 | 유가 변동률 및 시차 변수 생성 |
| 원/달러 환율 | 월별 환율 | 환율 변동률 변수 생성 |
| 한국은행 기준금리 | 월별 기준금리 | 보조 거시경제 변수 |

가공 데이터 `dataset/processed/monthly_analysis.csv`에는 다음과 같은 주요 컬럼이 포함됩니다.

- `date` : 관측월
- `dubai_price` : Dubai 유가
- `cpi_index` : 소비자물가지수
- `base_rate` : 기준금리
- `usd_krw` : 원/달러 환율
- `cpi_yoy` : CPI 전년 동월 대비 상승률
- `oil_mom`, `oil_log_mom` : 유가 월간 변동률
- `fx_mom`, `fx_log_mom` : 환율 월간 변동률
- `oil_lag_1`, `oil_lag_2`, `oil_lag_3`, `oil_lag_6` : 유가 시차 변수
- `target_cpi_yoy` : 다음 달 CPI 전년 동월 대비 상승률
- `target_date` : 예측 대상월

## 모델링 결과

노트북 기준 최종 모델은 `StandardScaler + Linear Regression` 파이프라인입니다.

- 평가 기간: 2022년 ~ 2025년
- 최종 모델: Linear Regression
- MAE: 0.3036
- RMSE: 0.3602
- R²: 0.9278
- 유가 변수를 포함한 전체 모델은 유가 제외 모델 대비 RMSE를 약 5.6% 개선
- 전체 데이터 재학습 모델의 2026년 6월 CPI 상승률 예측값: 3.28%

> 위 수치는 저장소의 분석 노트북 기준 결과입니다. RMSE는 신뢰구간이 아니라 과거 테스트 구간의 예측 오차 참고값입니다.

## 실행 방법

### 1. 저장소 클론

```bash
git clone https://github.com/Polalise/MachineLearning_oil.git
cd MachineLearning_oil
```

### 2. 가상환경 생성 및 활성화

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. Streamlit 앱 실행

```bash
streamlit run app.py
```

실행 후 브라우저에서 Streamlit이 안내하는 로컬 주소로 접속합니다.

```text
http://localhost:8501
```

## OpenAI 해석 기능 설정

AI 그래프 해석 기능을 사용하려면 OpenAI API 키를 환경 변수 또는 Streamlit secrets로 설정합니다.

환경 변수 방식:

```bash
set OPENAI_API_KEY=your_api_key
```

Streamlit secrets 방식:

```toml
# .streamlit/secrets.toml
OPENAI_API_KEY="your_api_key"
```

API 키가 없어도 기본 예측 대시보드와 시나리오 분석 기능은 사용할 수 있습니다.

## 사용 흐름

1. `model.ipynb`에서 원천 데이터를 정제하고 모델을 학습합니다.
2. 최종 모델과 입력 변수 목록을 `artifacts/cpi_linear_model.joblib`로 저장합니다.
3. 가공 데이터는 `dataset/processed/monthly_analysis.csv`로 저장합니다.
4. `app.py`가 모델과 데이터를 불러와 최신 예측, 시나리오 비교, 기여도 분석을 제공합니다.

## 해석 시 유의사항

- 이 프로젝트는 예측 모델이며 인과분석이 아닙니다.
- 유가, 환율, 기준금리의 기여도는 모델 예측을 설명하기 위한 값이며 실제 인과효과를 의미하지 않습니다.
- 급격한 물가 전환점에서는 모델 예측이 실제 변화보다 늦게 반응할 수 있습니다.
- 정책 판단이나 투자 판단에는 추가적인 경제 분석과 최신 데이터를 함께 검토해야 합니다.

## 기대 효과

- 유가와 환율 변화가 물가 예측에 제공하는 추가 정보를 정량적으로 확인할 수 있습니다.
- 단순 모델 비교를 넘어 Streamlit 서비스 형태로 예측 결과를 직관적으로 확인할 수 있습니다.
- 시나리오 조정을 통해 거시경제 변수 변화에 따른 CPI 전망 민감도를 탐색할 수 있습니다.
