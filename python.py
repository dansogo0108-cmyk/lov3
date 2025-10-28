import streamlit as st
st.title('👻 바이브코딩 웹사이트 제작 😈')
name = st.text_input('이름을 입력해주세요 : ')
menu = st.selectbox('좋아하는 음식을 선택해주세요:', ['한식🍚','양식🍕','일식🍣','중식🥮','분식🍥'])
if st.button('인사말') : 
  st.write(name+'! 너는 '+menu+'을 제일 좋아하는구나 ? 나두 ~ ~ ')
# app.py
# -------------------------------------------------------------
# Streamlit app for exploring MBTI type distribution by country
# - Focus: "특정 유형이 높은 국가 TOP 10" + 국가/유형 상세 분석
# - Visualization: Altair only (no extra installs)
# - Data: CSV with columns [Country, INTJ, INTP, ..., ESFP]
# -------------------------------------------------------------

import io
import sys
from typing import List

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="MBTI by Country — Top 10 Explorer",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Some tiny CSS polish (kept minimal; no external libs)
st.markdown(
    """
    <style>
    .headline { font-weight: 700; font-size: 1.4rem; margin-top: 0.5rem; }
    .subtle { color: #6b7280; font-size: 0.95rem; }
    .metric-kpi { display: flex; gap: 1rem; flex-wrap: wrap; }
    .metric-card { padding: 0.75rem 0.9rem; border: 1px solid #e5e7eb; border-radius: 12px; background: #fafafa; }
    .caption { color: #6b7280; font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Constants
# ----------------------------
MBTI_16: List[str] = [
    "INTJ","INTP","ENTJ","ENTP",
    "INFJ","INFP","ENFJ","ENFP",
    "ISTJ","ISFJ","ESTJ","ESFJ",
    "ISTP","ISFP","ESTP","ESFP",
]

DEFAULT_PATHS = [
    "countriesMBTI_16types.csv",  # repository root (Streamlit Cloud)
    "./data/countriesMBTI_16types.csv",  # common subfolder
    "/mnt/data/countriesMBTI_16types.csv",  # local sandbox fallback
]

# ----------------------------
# Loaders
# ----------------------------
@st.cache_data(show_spinner=False)
def load_csv(file_like) -> pd.DataFrame:
    df = pd.read_csv(file_like)
    # Normalize expected columns (strip whitespace)
    df.columns = [c.strip() for c in df.columns]
    # Minimal sanity: ensure Country exists
    if "Country" not in df.columns:
        # Try to infer possible country column
        candidates = [c for c in df.columns if c.lower() in {"nation","country_name","countries"}]
        if candidates:
            df = df.rename(columns={candidates[0]: "Country"})
        else:
            raise ValueError("CSV must contain a 'Country' column.")
    # Identify MBTI cols present
    mbti_cols = [c for c in MBTI_16 if c in df.columns]
    if len(mbti_cols) == 0:
        raise ValueError("CSV must include MBTI columns (e.g., INTJ, INTP, ..., ESFP).")
    # Cast MBTI cols to numeric
    for c in mbti_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Drop rows fully NA across MBTI cols
    if df[mbti_cols].isna().all(axis=1).any():
        df = df[~df[mbti_cols].isna().all(axis=1)].copy()
    return df

@st.cache_data(show_spinner=False)
def try_default_paths(paths: List[str]):
    last_err = None
    for p in paths:
        try:
            return load_csv(p)
        except Exception as e:
            last_err = e
            continue
    if last_err:
        raise last_err

# ----------------------------
# Sidebar — data input
# ----------------------------
st.sidebar.header("데이터 입력")
up = st.sidebar.file_uploader("CSV 업로드 (Country + 16 MBTI 열)", type=["csv"], accept_multiple_files=False)

if up is not None:
    df = load_csv(up)
else:
    # Attempt default file paths so the app works on Cloud if the file exists in repo
    try:
        df = try_default_paths(DEFAULT_PATHS)
        st.sidebar.info("기본 CSV를 불러왔어요. (countriesMBTI_16types.csv)")
    except Exception as e:
        st.sidebar.warning("CSV를 업로드해주세요. (Country와 16개 MBTI 열 필요)")
        st.stop()

# Recompute available MBTI columns from df
MBTI_COLS = [c for c in MBTI_16 if c in df.columns]

# Compute dominant type per country
_df = df.copy()
_df["_dominant_mbti"] = _df[MBTI_COLS].idxmax(axis=1)

# ----------------------------
# Header
# ----------------------------
st.markdown("<div class='headline'>MBTI by Country — Top 10 Explorer</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtle'>특정 유형이 높은 국가 TOP 10을 빠르게 찾고, 국가·유형별 상세를 탐색하세요.</div>",
    unsafe_allow_html=True,
)

# KPI row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("국가 수", f"{len(_df):,}")
with col2:
    st.metric("MBTI 열 수", f"{len(MBTI_COLS)}")
with col3:
    top_dom = _df["_dominant_mbti"].value_counts().idxtop() if not _df.empty else "-"
    st.metric("가장 흔한 지배 타입", top_dom)
with col4:
    st.metric("데이터 상태", "OK" if len(MBTI_COLS) >= 8 else "확인 필요")

# ----------------------------
# Tabs
# ----------------------------
TAB1, TAB2, TAB3 = st.tabs(["🧭 유형별 Top 10", "🏳️ 국가 상세", "📊 전체 분포"])

# ----------------------------
# TAB1 — Type Top N by Country
# ----------------------------
with TAB1:
    c1, c2, c3 = st.columns([1.3, 1, 1])
    with c1:
        selected_type = st.selectbox("유형 선택", MBTI_COLS, index=MBTI_COLS.index("INFP") if "INFP" in MBTI_COLS else 0)
    with c2:
        top_n = st.slider("Top N", min_value=5, max_value=30, value=10, step=1)
    with c3:
        fmt_pct = st.checkbox("값을 %로 표시", value=True)

    # Build topN table
    top_df = _df[["Country", selected_type]].dropna().sort_values(selected_type, ascending=False).head(top_n)
    top_df = top_df.reset_index(drop=True)

    # Bar chart with Altair
    value_field = selected_type
    base = alt.Chart(top_df).encode(
        x=alt.X(alt.Shorthand(value_field), title=f"{selected_type} 값" + (" (%)" if fmt_pct else "")),
        y=alt.Y("Country:N", sort='-x', title="국가"),
        tooltip=["Country", alt.Tooltip(value_field, format=".2%" if fmt_pct else ".4f", title=selected_type)],
    )

    color_scale = alt.Scale(scheme="blues")
    bars = base.mark_bar().encode(
        color=alt.Color(value_field, scale=color_scale, legend=None)
    )

    text = alt.Chart(top_df).mark_text(dx=5, dy=0, align="left", baseline="middle").encode(
        x=alt.X(value_field),
        y=alt.Y("Country"),
        text=alt.Text(value_field, format=".2%" if fmt_pct else ".4f"),
    )

    chart = (bars + text).properties(height=520)
    st.altair_chart(chart, use_container_width=True)

    with st.expander("표 보기"):
        show_df = top_df.copy()
        if fmt_pct:
            show_df[selected_type] = (show_df[selected_type] * 100).round(2)
            show_df = show_df.rename(columns={selected_type: f"{selected_type} (%)"})
        st.dataframe(show_df, use_container_width=True)

# ----------------------------
# TAB2 — Country detail
# ----------------------------
with TAB2:
    country = st.selectbox("국가 선택", _df["Country"].tolist(), index=0)
    row = _df[_df["Country"] == country]
    if row.empty:
        st.warning("선택 국가의 데이터가 없습니다.")
    else:
        s = row[MBTI_COLS].iloc[0].sort_values(ascending=False)
        dom = s.idxmax()
        with st.container():
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown("#### 지배 타입")
                st.metric(label=country, value=dom)
                st.write("Top 3")
                st.write(", ".join([f"{k}: {v:.2%}" for k, v in s.head(3).items()]))
            with c2:
                # Horizontal bar chart sorted
                cdf = s.reset_index()
                cdf.columns = ["MBTI", "Value"]
                chart = (
                    alt.Chart(cdf)
                    .mark_bar()
                    .encode(
                        x=alt.X("Value:Q", title="값 (%)"),
                        y=alt.Y("MBTI:N", sort='-x'),
                        color=alt.Color("Value:Q", scale=alt.Scale(scheme="tealblues"), legend=None),
                        tooltip=["MBTI", alt.Tooltip("Value:Q", format=".2%")],
                    )
                    .properties(height=480)
                )
                st.altair_chart(chart, use_container_width=True)

        with st.expander("원본 값 보기"):
            out = row[["Country"] + MBTI_COLS].T.reset_index()
            out.columns = ["항목", "값"]
            st.dataframe(out, use_container_width=True, height=420)

# ----------------------------
# TAB3 — Global distribution
# ----------------------------
with TAB3:
    c1, c2 = st.columns([1.1, 1])
    with c1:
        t1 = st.selectbox("분포 (히스토그램) — 유형 선택", MBTI_COLS, index=MBTI_COLS.index("INFP") if "INFP" in MBTI_COLS else 0)
        bins = st.slider("빈 수(bins)", min_value=10, max_value=50, value=20)
        hdf = _df[["Country", t1]].dropna()
        hist = (
            alt.Chart(hdf)
            .mark_bar()
            .encode(
                x=alt.X(f"{t1}:Q", bin=alt.Bin(maxbins=bins), title=f"{t1}"),
                y=alt.Y('count()', title='국가 수'),
                tooltip=[alt.Tooltip("count():Q", title="국가 수")],
            )
            .properties(height=380)
        )
        st.altair_chart(hist, use_container_width=True)

    with c2:
        tX = st.selectbox("산점도 — X축 유형", MBTI_COLS, index=MBTI_COLS.index("INFP") if "INFP" in MBTI_COLS else 0)
        tY = st.selectbox("산점도 — Y축 유형", MBTI_COLS, index=MBTI_COLS.index("ESTJ") if "ESTJ" in MBTI_COLS else 1)
        sdf = _df[["Country", tX, tY]].dropna()
        scatter = (
            alt.Chart(sdf)
            .mark_circle(size=80, opacity=0.85)
            .encode(
                x=alt.X(f"{tX}:Q", title=tX),
                y=alt.Y(f"{tY}:Q", title=tY),
                tooltip=["Country", alt.Tooltip(f"{tX}:Q", format=".2%"), alt.Tooltip(f"{tY}:Q", format=".2%")],
                color=alt.Color(f"{tX}:Q", scale=alt.Scale(scheme="purples"), legend=None),
            )
            .properties(height=380)
        )
        st.altair_chart(scatter, use_container_width=True)

    st.caption("Tip: 산점도에서 서로 음의 상관처럼 보이는 쌍(예: INFP vs ESTJ)을 살펴보세요.")

# ----------------------------
# Footer
# ----------------------------
st.markdown("""
<div class='caption'>
<strong>사용법</strong><br/>
1) 좌측에서 CSV 업로드(또는 repo에 <code>countriesMBTI_16types.csv</code>를 두면 자동 로드).<br/>
2) 탭에서 유형을 고르고 Top 10을 확인하거나, 특정 국가를 선택해 상세 분포를 확인하세요.<br/>
3) 모든 차트는 Altair 기반으로 상호작용(툴팁, 정렬) 가능.
</div>
""", unsafe_allow_html=True)
