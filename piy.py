# app.py (robust, no extra installs)
# -------------------------------------------------------------
# 50개 국가의 출산률×GDP 데이터 생성/업로드 + Top10 & 순위 투영
# Altair로만 시각화, 추가 라이브러리 설치 금지(Styler gradient 사용 안 함)
# -------------------------------------------------------------

from __future__ import annotations
import io
from typing import List, Dict

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
# 상수/초기 국가 목록
# -------------------------------------------------------------
COUNTRIES_50 = [
    "United States","China","Japan","Germany","India","United Kingdom","France",
    "Italy","Canada","South Korea","Russia","Brazil","Australia","Spain",
    "Mexico","Indonesia","Netherlands","Saudi Arabia","Turkey","Switzerland",
    "Taiwan","Poland","Sweden","Belgium","Thailand","Ireland","Israel",
    "Argentina","Norway","United Arab Emirates","South Africa","Denmark",
    "Singapore","Malaysia","Philippines","Vietnam","Portugal","Greece",
    "Czechia","Chile","Colombia","New Zealand","Finland","Austria",
    "Hungary","Romania","Peru","Egypt","Pakistan","Bangladesh"
]

DEFAULT_FILENAME = "country_metrics_50.csv"

# 표준 컬럼명
STD = {
    "country": "Country",
    "gdp_billionusd": "GDP_BillionUSD",
    "gdp_growthrate": "GDP_GrowthRate",
    "fertility_rate": "Fertility_Rate",
    "gdp_rank": "GDP_Rank",
}

# 컬럼 별 허용 별칭 (소문자/공백/기호 제거 후 비교)
ALIASES: Dict[str, List[str]] = {
    "country": ["country","nation","countryname","countries"],
    "gdp_billionusd": ["gdp_billionusd","gdp","gdpusd","gdp_usd_billion","gdp_current_usd_billion","gdp(bn)","gdp_billion"],
    "gdp_growthrate": ["gdp_growthrate","growth","growthrate","gdp_growth","annual_growth","gdp_yoy"],
    "fertility_rate": ["fertility_rate","tfr","birthrate","birth_rate","fertility"],
    "gdp_rank": ["gdp_rank","rank","gdporder","rank_gdp"],
}

# -------------------------------------------------------------
# 유틸
# -------------------------------------------------------------

def norm(s: str) -> str:
    return ''.join(ch for ch in s.lower().strip() if ch.isalnum())

@st.cache_data(show_spinner=False)
def save_csv_to_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

# -------------------------------------------------------------
# 샘플 데이터 생성
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generate_sample_dataset(countries: List[str]) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    gdp_raw = np.exp(rng.normal(loc=7.5, scale=1.1, size=len(countries)))
    gdp_bil = np.interp(gdp_raw, (gdp_raw.min(), gdp_raw.max()), (80, 26000))

    growth = rng.normal(loc=0.03, scale=0.02, size=len(countries))
    growth = np.clip(growth, -0.02, 0.08)

    fert = rng.normal(loc=1.8, scale=0.6, size=len(countries))
    fert = np.clip(fert, 0.9, 4.5)

    df = pd.DataFrame({
        "Country": countries,
        "GDP_BillionUSD": gdp_bil,
        "GDP_GrowthRate": growth,
        "Fertility_Rate": fert,
    })
    df = df.sort_values("GDP_BillionUSD", ascending=False).reset_index(drop=True)
    df["GDP_Rank"] = np.arange(1, len(df) + 1)
    return df

# -------------------------------------------------------------
# CSV 로더(유연한 컬럼 매핑 + 자동 보정)
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(file_like) -> pd.DataFrame:
    df = pd.read_csv(file_like)
    # 1) 컬럼 전처리 & 매핑
    original = list(df.columns)
    mapping = {}
    for col in original:
        key = None
        nc = norm(col)
        for std_key, alist in ALIASES.items():
            if nc in alist:
                key = STD[std_key]
                break
        mapping[col] = key if key else col  # 모르는 컬럼은 유지
    df = df.rename(columns=mapping)

    # 최소 필수: Country, GDP_BillionUSD, GDP_GrowthRate, Fertility_Rate
    need = {STD["country"], STD["gdp_billionusd"], STD["gdp_growthrate"], STD["fertility_rate"]}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"다음 필드가 필요합니다: {', '.join(sorted(missing))}")

    # 타입 보정
    df["Country"] = df["Country"].astype(str)
    for c in ["GDP_BillionUSD", "GDP_GrowthRate", "Fertility_Rate"]:
        # 퍼센트 문자열 처리(예: "3.2%")
        if c == "GDP_GrowthRate" and df[c].dtype == object:
            df[c] = df[c].astype(str).str.replace('%','', regex=False)
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 성장률이 1보다 큰 값(예: 3.2)이면 %로 간주해 0.032로 변환
    if (df["GDP_GrowthRate"] > 1).any():
        df.loc[df["GDP_GrowthRate"] > 1, "GDP_GrowthRate"] = df.loc[df["GDP_GrowthRate"] > 1, "GDP_GrowthRate"] / 100.0

    # GDP_Rank 없으면 계산
    if "GDP_Rank" not in df.columns or df["GDP_Rank"].isna().all():
        df = df.sort_values("GDP_BillionUSD", ascending=False).reset_index(drop=True)
        df["GDP_Rank"] = np.arange(1, len(df) + 1)

    # 핵심 결측 제거
    df = df.dropna(subset=["GDP_BillionUSD", "GDP_GrowthRate", "Fertility_Rate"]).reset_index(drop=True)
    return df

# -------------------------------------------------------------
# 투영 로직
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def project_gdp_and_rank(df: pd.DataFrame, years: int = 5) -> pd.DataFrame:
    out = df.copy()
    out["Projected_GDP_BillionUSD"] = out["GDP_BillionUSD"] * (1.0 + out["GDP_GrowthRate"]) ** years
    out = out.sort_values("Projected_GDP_BillionUSD", ascending=False).reset_index(drop=True)
    out["Projected_Rank"] = np.arange(1, len(out) + 1)
    out = out.merge(df[["Country", "GDP_Rank"]], on="Country", how="left")
    out["Rank_Change"] = out["GDP_Rank"] - out["Projected_Rank"]
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
st.markdown("<div class='subtle'>출산률·GDP·성장률을 바탕으로 간단한 미래 순위를 추정합니다. (학습용)</div>", unsafe_allow_html=True)

# CSV 다운로드
st.download_button(
    label=f"CSV 다운로드 — {DEFAULT_FILENAME}",
    data=save_csv_to_bytes(df),
    file_name=DEFAULT_FILENAME,
    mime="text/csv",
)

# 파생/집계
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
    st.metric("순위 상승 국가 수", f"{int((proj['Rank_Change']>0).sum())}")

# 탭
TAB1, TAB2, TAB3 = st.tabs(["🏆 Top 10", "📈 순위 투영", "🔎 상관/산점도"])

# TAB1 — 특정 지표 Top10
with TAB1:
    met_map = {
        "현재 GDP": "GDP_BillionUSD",
        "GDP 성장률": "GDP_GrowthRate",
        "출산률": "Fertility_Rate",
        f"투영 GDP(+{years}y)": "Projected_GDP_BillionUSD",
    }
    merged = df.merge(proj[["Country","Projected_GDP_BillionUSD","Projected_Rank"]], on="Country", how="left")

    c1, c2 = st.columns([1.2, 1])
    with c1:
        metric_label = st.selectbox("지표 선택", list(met_map.keys()), index=0)
    with c2:
        topn = st.slider("Top N", min_value=5, max_value=30, value=10)

    field = met_map[metric_label]
    top_df = merged[["Country", field]].dropna().sort_values(field, ascending=False).head(topn)

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

# TAB2 — 순위 투영/변화
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

# TAB3 — 상관/산점도 (matplotlib 불필요, Altair만 사용)
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

    st.markdown("#### 상관계수 히트맵 (Altair)")
    corr = df[["GDP_BillionUSD","GDP_GrowthRate","Fertility_Rate"]].corr().reset_index().melt(id_vars="index")
    corr.columns = ["VarX","VarY","Corr"]
    heat = (
        alt.Chart(corr)
        .mark_rect()
        .encode(
            x=alt.X("VarX:N", title=""),
            y=alt.Y("VarY:N", title=""),
            color=alt.Color("Corr:Q", scale=alt.Scale(scheme="blueorange", domain=[-1,1])),
            tooltip=["VarX","VarY", alt.Tooltip("Corr:Q", format=".2f")],
        )
        .properties(height=220)
    )
    st.altair_chart(heat, use_container_width=True)

# 푸터
st.caption(
    "학습용 예시입니다. 실제 분석에는 최신 공식 통계를 사용하고, 업로드 기능으로 여러분의 데이터를 바로 적용하세요."
)
