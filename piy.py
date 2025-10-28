# app.py
# -------------------------------------------------------------
# 50개 국가의 출산률(Fertility)과 GDP, 성장률을 담은 CSV를
# 생성(또는 업로드)하고, 향후 GDP 기반 국가 순위 변화를
# 단순 투영(Projections)하여 시각적으로 탐색하는 앱.
# - 시각화: Altair (Streamlit Cloud 기본 제공)
# - 추가 설치 금지: pandas, numpy, streamlit, altair만 사용
# -------------------------------------------------------------

from __future__ import annotations
import io
import math
import random
import datetime as dt
from typing import List, Tuple

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

# -------------------------------------------------------------
# 기본 설정
# -------------------------------------------------------------
st.set_page_config(
    page_title="국가 지표 Top10 & 순위 투영(Altair)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .headline { font-weight: 800; font-size: 1.5rem; }
    .subtle { color: #6b7280; }
    .card { padding: 0.75rem 1rem; border: 1px solid #e5e7eb; border-radius: 12px; background: #fafafa; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------
# 샘플 50개국 리스트 (임의 예시)
# -------------------------------------------------------------
COUNTRIES_50 = [
    "United States", "China", "Japan", "Germany", "India", "United Kingdom", "France",
    "Italy", "Canada", "South Korea", "Russia", "Brazil", "Australia", "Spain",
    "Mexico", "Indonesia", "Netherlands", "Saudi Arabia", "Turkey", "Switzerland",
    "Taiwan", "Poland", "Sweden", "Belgium", "Thailand", "Ireland", "Israel",
    "Argentina", "Norway", "United Arab Emirates", "South Africa", "Denmark",
    "Singapore", "Malaysia", "Philippines", "Vietnam", "Portugal", "Greece",
    "Czechia", "Chile", "Colombia", "New Zealand", "Finland", "Austria",
    "Hungary", "Romania", "Peru", "Egypt", "Pakistan", "Bangladesh"
]

SCHEMA = [
    ("Country", "str", "국가명"),
    ("GDP_BillionUSD", "float", "현재 GDP (십억 달러)"),
    ("GDP_GrowthRate", "float", "연간 성장률(소수, 예: 0.03=3%)"),
    ("Fertility_Rate", "float", "출산율(여성 1인당 출생수)"),
    ("GDP_Rank", "int", "현재 GDP 순위(1=최상위)"),
]

DEFAULT_FILENAME = "country_metrics_50.csv"

# -------------------------------------------------------------
# 데이터 생성 유틸: 현실적인 범위에서 무작위 샘플 생성
# -------------------------------------------------------------

def seed_random(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def generate_sample_dataset(countries: List[str]) -> pd.DataFrame:
    """간단한 규칙으로 50개국 임의 데이터를 생성한다.
    - GDP_BillionUSD: 80 ~ 26,000 사이 로그분포에 가깝게
    - GDP_GrowthRate: -2% ~ 8% 사이(평균 3% 근처)
    - Fertility_Rate: 0.9 ~ 4.5 사이
    - GDP_Rank: GDP 역순으로 랭크
    """
    seed_random()

    # GDP: 로그정규에 가깝게 생성 → 스케일 조정
    gdp = np.exp(np.random.normal(loc=7.5, scale=1.1, size=len(countries)))  # ~ e^(6~9)
    gdp = np.interp(gdp, (gdp.min(), gdp.max()), (80, 26000))

    # 성장률: 베타/정규 혼합 대신 단순 정규 절단
    growth = np.random.normal(loc=0.03, scale=0.02, size=len(countries))  # 평균 3%
    growth = np.clip(growth, -0.02, 0.08)

    # 출산률: 대체로 1.0~2.5, 일부 국가는 3~4대
    fert = np.random.normal(loc=1.8, scale=0.6, size=len(countries))
    fert = np.clip(fert, 0.9, 4.5)

    df = pd.DataFrame({
        "Country": countries,
        "GDP_BillionUSD": gdp,
        "GDP_GrowthRate": growth,
        "Fertility_Rate": fert,
    })

    # GDP 랭크(내림차순으로 1위=최대)
    df = df.sort_values("GDP_BillionUSD", ascending=False).reset_index(drop=True)
    df["GDP_Rank"] = np.arange(1, len(df) + 1)
    # 원래 순서 정렬 복원은 불필요. 현재는 GDP 기준으로 정렬된 상태.
    return df

# -------------------------------------------------------------
# 파일 입출력
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def save_csv_to_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

@st.cache_data(show_spinner=False)
def load_csv(file_like) -> pd.DataFrame:
    df = pd.read_csv(file_like)
    # 스키마 최소 검증
    need_cols = {"Country", "GDP_BillionUSD", "GDP_GrowthRate", "Fertility_Rate", "GDP_Rank"}
    missing = need_cols - set(df.columns)
    if missing:
        raise ValueError(f"다음 필드가 필요합니다: {', '.join(sorted(missing))}")
    # 타입 캐스팅
    df["Country"] = df["Country"].astype(str)
    df["GDP_BillionUSD"] = pd.to_numeric(df["GDP_BillionUSD"], errors="coerce")
    df["GDP_GrowthRate"] = pd.to_numeric(df["GDP_GrowthRate"], errors="coerce")
    df["Fertility_Rate"] = pd.to_numeric(df["Fertility_Rate"], errors="coerce")
    df["GDP_Rank"] = pd.to_numeric(df["GDP_Rank"], errors="coerce").astype("Int64")
    # 결측 제거(핵심 지표 누락 행 제거)
    df = df.dropna(subset=["GDP_BillionUSD", "GDP_GrowthRate", "Fertility_Rate"])
    return df

# -------------------------------------------------------------
# 투영 로직: 단순 복리 성장 기반 GDP 예측 및 재랭킹
# -------------------------------------------------------------

def project_gdp_and_rank(df: pd.DataFrame, years: int = 5) -> pd.DataFrame:
    out = df.copy()
    out["Projected_GDP_BillionUSD"] = out["GDP_BillionUSD"] * (1.0 + out["GDP_GrowthRate"]) ** years
    # 재랭킹: GDP 큰 순서가 1위
    out = out.sort_values("Projected_GDP_BillionUSD", ascending=False).reset_index(drop=True)
    out["Projected_Rank"] = np.arange(1, len(out) + 1)
    # 원래 순위와 비교 위해 원본 열 합침
    out = out.merge(
        df[["Country", "GDP_Rank"]], on="Country", how="left", suffixes=("", "_orig")
    )
    out["Rank_Change"] = out["GDP_Rank"] - out["Projected_Rank"]  # +면 순위 상승(숫자 작아짐)
    # 보기 좋게 정렬
    out = out.sort_values("Projected_Rank").reset_index(drop=True)
    return out

# -------------------------------------------------------------
# 사이드바: 데이터 준비
# -------------------------------------------------------------
st.sidebar.header("데이터 준비")
opt = st.sidebar.radio("데이터 선택", ("샘플 생성(50개국)", "CSV 업로드"))

if opt == "CSV 업로드":
    upl = st.sidebar.file_uploader("CSV 업로드", type=["csv"], accept_multiple_files=False)
    if upl is None:
        st.sidebar.info("또는 '샘플 생성'을 선택하세요.")
        st.stop()
    df = load_csv(upl)
else:
    df = generate_sample_dataset(COUNTRIES_50)
    st.sidebar.success("샘플 50개국 데이터를 생성했습니다.")

years = st.sidebar.slider("투영 기간(년)", min_value=1, max_value=15, value=5)

st.markdown("<div class='headline'>국가 지표 Top10 & 순위 투영(Altair)</div>", unsafe_allow_html=True)
st.markdown("<div class='subtle'>출산률과 GDP, 성장률을 바탕으로 간단한 미래 순위를 추정합니다. (학습용 예시)</div>", unsafe_allow_html=True)

# 다운로드 제공
csv_bytes = save_csv_to_bytes(df)
st.download_button(
    label=f"CSV 다운로드 — {DEFAULT_FILENAME}",
    data=csv_bytes,
    file_name=DEFAULT_FILENAME,
    mime="text/csv",
)

# -------------------------------------------------------------
# 파생/집계
# -------------------------------------------------------------
proj = project_gdp_and_rank(df, years=years)

# KPI
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("국가 수", f"{len(df):,}")
with c2:
    st.metric("평균 성장률", f"{df['GDP_GrowthRate'].mean()*100:,.2f}%")
with c3:
    st.metric("평균 출산률", f"{df['Fertility_Rate'].mean():.2f}")
with c4:
    rank_up = int((proj["Rank_Change"] > 0).sum())
    st.metric("순위 상승 국가 수", f"{rank_up}")

# -------------------------------------------------------------
# 탭 구성
# -------------------------------------------------------------
TAB1, TAB2, TAB3 = st.tabs(["🏆 Top 10", "📈 순위 투영", "🔎 상관/산점도"])

# -------------------------------------------------------------
# TAB1 — 특정 분야가 높은 국가 Top10
# -------------------------------------------------------------
with TAB1:
    met_map = {
        "현재 GDP": "GDP_BillionUSD",
        "GDP 성장률": "GDP_GrowthRate",
        "출산률": "Fertility_Rate",
        f"투영 GDP(+{years}y)": "Projected_GDP_BillionUSD",
    }

    # proj가 필요한 항목 포함하므로 미리 합본 뷰 준비
    merged = df.merge(proj[["Country", "Projected_GDP_BillionUSD", "Projected_Rank"]], on="Country", how="left")

    c1, c2 = st.columns([1.2, 1])
    with c1:
        metric_label = st.selectbox("지표 선택", list(met_map.keys()), index=0)
    with c2:
        topn = st.slider("Top N", min_value=5, max_value=30, value=10)

    field = met_map[metric_label]

    top_df = merged[["Country", field]].dropna().sort_values(field, ascending=False).head(topn)

    # 값 포맷 결정
    is_pct = (field == "GDP_GrowthRate")
    x_title = metric_label + (" (%)" if is_pct else "")
    tooltip_fmt = ".2%" if is_pct else ".2f"

    chart = (
        alt.Chart(top_df)
        .mark_bar()
        .encode(
            x=alt.X(f"{field}:Q", title=x_title),
            y=alt.Y("Country:N", sort='-x', title="국가"),
            color=alt.Color(f"{field}:Q", scale=alt.Scale(scheme="blues"), legend=None),
            tooltip=["Country", alt.Tooltip(f"{field}:Q", format=tooltip_fmt, title=metric_label)],
        )
        .properties(height=520)
    )
    st.altair_chart(chart, use_container_width=True)

    with st.expander("표 보기"):
        show = top_df.copy()
        if is_pct:
            show[field] = (show[field] * 100).round(2)
        st.dataframe(show, use_container_width=True)

# -------------------------------------------------------------
# TAB2 — 순위 투영/변화
# -------------------------------------------------------------
with TAB2:
    view_cols = [
        "Country", "GDP_Rank", "Projected_Rank", "Rank_Change", "GDP_BillionUSD", "Projected_GDP_BillionUSD", "GDP_GrowthRate", "Fertility_Rate"
    ]
    st.markdown("#### 현재 vs 투영 순위")
    st.dataframe(proj[view_cols].sort_values("Projected_Rank").reset_index(drop=True), use_container_width=True, height=500)

    st.markdown("#### 순위 변화 Top10 (상승)")
    up10 = proj.sort_values("Rank_Change", ascending=False).head(10)
    ch1 = (
        alt.Chart(up10)
        .mark_bar()
        .encode(
            x=alt.X("Rank_Change:Q", title="순위 변화(+면 상승)"),
            y=alt.Y("Country:N", sort='-x'),
            color=alt.Color("Rank_Change:Q", scale=alt.Scale(scheme="tealblues"), legend=None),
            tooltip=["Country", alt.Tooltip("Rank_Change:Q", format="+.0f")],
        )
        .properties(height=360)
    )
    st.altair_chart(ch1, use_container_width=True)

    st.markdown("#### 투영 GDP Top10")
    proj_top = proj.nsmallest(10, "Projected_Rank")[["Country", "Projected_GDP_BillionUSD"]]
    ch2 = (
        alt.Chart(proj_top)
        .mark_bar()
        .encode(
            x=alt.X("Projected_GDP_BillionUSD:Q", title=f"투영 GDP(+{years}y) (십억 달러)"),
            y=alt.Y("Country:N", sort='-x'),
            color=alt.Color("Projected_GDP_BillionUSD:Q", scale=alt.Scale(scheme="blues"), legend=None),
            tooltip=["Country", alt.Tooltip("Projected_GDP_BillionUSD:Q", format=",.1f")],
        )
        .properties(height=360)
    )
    st.altair_chart(ch2, use_container_width=True)

# -------------------------------------------------------------
# TAB3 — 상관/산점도
# -------------------------------------------------------------
with TAB3:
    st.markdown("#### 출산률과 성장률/규모 관계")
    left, right = st.columns(2)

    with left:
        s1 = (
            alt.Chart(df)
            .mark_circle(size=90, opacity=0.85)
            .encode(
                x=alt.X("Fertility_Rate:Q", title="출산률"),
                y=alt.Y("GDP_GrowthRate:Q", title="GDP 성장률"),
                tooltip=["Country", alt.Tooltip("Fertility_Rate:Q", format=".2f"), alt.Tooltip("GDP_GrowthRate:Q", format=".2%")],
                color=alt.Color("GDP_GrowthRate:Q", scale=alt.Scale(scheme="greens"), legend=None),
            )
            .properties(height=380)
        )
        st.altair_chart(s1, use_container_width=True)

    with right:
        s2 = (
            alt.Chart(df)
            .mark_circle(size=90, opacity=0.85)
            .encode(
                x=alt.X("Fertility_Rate:Q", title="출산률"),
                y=alt.Y("GDP_BillionUSD:Q", title="GDP (십억 달러)"),
                tooltip=["Country", alt.Tooltip("Fertility_Rate:Q", format=".2f"), alt.Tooltip("GDP_BillionUSD:Q", format=",.1f")],
                color=alt.Color("GDP_BillionUSD:Q", scale=alt.Scale(scheme="purples"), legend=None),
            )
            .properties(height=380)
        )
        st.altair_chart(s2, use_container_width=True)

    # 간단 상관계수 표(참고)
    st.markdown("#### 상관계수(피어슨)")
    corr = df[["GDP_BillionUSD", "GDP_GrowthRate", "Fertility_Rate"]].corr()
    st.dataframe(corr.style.background_gradient(cmap="Blues"), use_container_width=True)

# -------------------------------------------------------------
# 푸터
# -------------------------------------------------------------
st.caption(
    "학습용 예시 데이터입니다. 실제 정책/연구 해석에는 최신 공식 통계를 사용하세요. "
    "CSV 업로드를 통해 여러분의 데이터로 즉시 대체할 수 있습니다."
)
