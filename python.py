import streamlit as st
st.title('👻 바이브코딩 웹사이트 제작 😈')
name = st.text_input('이름을 입력해주세요 : ')
menu = st.selectbox('좋아하는 음식을 선택해주세요:', ['한식🍚','양식🍕','일식🍣','중식🥮','분식🍥'])
if st.button('인사말') : 
  st.write(name+'! 너는 '+menu+'을 제일 좋아하는구나 ? 나두 ~ ~ ')

# app.py (fixed)
# -------------------------------------------------------------
# Streamlit app for exploring MBTI type distribution by country
# Focus: 특정 유형이 높은 국가 TOP 10 + 국가/유형 상세 분석
# Visualization: Altair only (no extra installs)
# -------------------------------------------------------------

import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="MBTI by Country — Top 10 Explorer", page_icon="🧭", layout="wide")

MBTI_16 = [
    "INTJ","INTP","ENTJ","ENTP",
    "INFJ","INFP","ENFJ","ENFP",
    "ISTJ","ISFJ","ESTJ","ESFJ",
    "ISTP","ISFP","ESTP","ESFP"
]

@st.cache_data
def load_csv(file):
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]
    if "Country" not in df.columns:
        raise ValueError("CSV must contain 'Country' column")
    mbti_cols = [c for c in MBTI_16 if c in df.columns]
    for c in mbti_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["_dominant_mbti"] = df[mbti_cols].idxmax(axis=1)
    return df, mbti_cols

st.sidebar.header("데이터 입력")
up = st.sidebar.file_uploader("CSV 업로드 (Country + 16 MBTI 열)", type="csv")
if up is None:
    st.stop()

df, MBTI_COLS = load_csv(up)

st.title("🧭 MBTI by Country — Top 10 Explorer")
st.write("특정 유형이 높은 국가 TOP 10을 시각적으로 탐색합니다.")

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("국가 수", len(df))
col2.metric("MBTI 열 수", len(MBTI_COLS))
col3.metric("가장 흔한 지배 타입", df["_dominant_mbti"].value_counts().idxmax())

# Tabs
tab1, tab2 = st.tabs(["유형별 Top 10", "국가 상세"])

with tab1:
    selected_type = st.selectbox("유형 선택", MBTI_COLS, index=MBTI_COLS.index("INFP") if "INFP" in MBTI_COLS else 0)
    top_n = st.slider("Top N", 5, 30, 10)
    fmt_pct = st.checkbox("값을 %로 표시", value=True)

    top_df = df[["Country", selected_type]].sort_values(selected_type, ascending=False).head(top_n)

    chart = (
        alt.Chart(top_df)
        .mark_bar()
        .encode(
            x=alt.X(f"{selected_type}:Q", title=f"{selected_type} 값" + (" (%)" if fmt_pct else "")),
            y=alt.Y("Country:N", sort='-x', title="국가"),
            color=alt.Color(f"{selected_type}:Q", scale=alt.Scale(scheme="blues"), legend=None),
            tooltip=["Country", alt.Tooltip(f"{selected_type}:Q", format=".2%" if fmt_pct else ".4f")]
        )
        .properties(height=500)
    )

    st.altair_chart(chart, use_container_width=True)
    st.dataframe(top_df, use_container_width=True)

with tab2:
    country = st.selectbox("국가 선택", df["Country"].tolist())
    row = df[df["Country"] == country]
    if not row.empty:
        s = row[MBTI_COLS].iloc[0].sort_values(ascending=False)
        dom = s.idxmax()
        st.metric(label=f"{country}의 지배 타입", value=dom)

        bar = (
            alt.Chart(s.reset_index())
            .mark_bar()
            .encode(
                x=alt.X("Value:Q", title="값"),
                y=alt.Y("MBTI:N", sort='-x'),
                color=alt.Color("Value:Q", scale=alt.Scale(scheme="tealblues"), legend=None)
            )
        )
        s_df = s.reset_index()
        s_df.columns = ["MBTI", "Value"]
        st.altair_chart(bar, use_container_width=True)
        st.dataframe(s_df, use_container_width=True)
