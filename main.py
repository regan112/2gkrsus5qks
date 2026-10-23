import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="이상치·결측치 탐색", layout="wide")
st.title("📊 고객 이탈 데이터: 결측치 & 이상치 탐색")

FILE_PATH = "synthetic_customer_churn_100k.csv"

# 컬럼명 / 값 한국어 변환 사전
KOR_COLS = {
    "CustomerID": "고객번호",
    "Age": "나이",
    "Gender": "성별",
    "Tenure": "가입기간(개월)",
    "MonthlyCharges": "월요금",
    "Contract": "계약형태",
    "PaymentMethod": "결제방식",
    "TotalCharges": "총요금",
    "Churn": "이탈여부",
}
KOR_VALUES = {
    "Gender": {"Male": "남성", "Female": "여성", "Other": "기타"},
    "Contract": {"Month-to-month": "월 단위", "One year": "1년", "Two year": "2년"},
    "PaymentMethod": {
        "Bank transfer": "계좌이체",
        "Mailed check": "우편수표",
        "Electronic check": "전자수표",
        "Credit card": "신용카드",
    },
    "Churn": {"Yes": "이탈", "No": "유지"},
}


@st.cache_data
def load_data(path):
    data = pd.read_csv(path)
    for col, mapping in KOR_VALUES.items():
        if col in data.columns:
            data[col] = data[col].replace(mapping)
    return data.rename(columns=KOR_COLS)


try:
    df = load_data(FILE_PATH)
except FileNotFoundError:
    st.error(f"'{FILE_PATH}' 파일을 찾을 수 없습니다. main.py와 같은 폴더에 두세요.")
    st.stop()

# ---------------- 1. 데이터 개요 ----------------
st.subheader("1. 데이터 개요")
c1, c2, c3 = st.columns(3)
c1.metric("행 수", f"{df.shape[0]:,}")
c2.metric("열 수", df.shape[1])
c3.metric("전체 결측치 수", f"{int(df.isnull().sum().sum()):,}")
st.dataframe(df.head(20))

# ---------------- 2. 결측치 ----------------
st.header("2. 결측치 분석")
missing = df.isnull().sum()
missing_df = pd.DataFrame({
    "컬럼": missing.index,
    "결측치 수": missing.values,
    "결측 비율(%)": (missing.values / len(df) * 100).round(3),
})
st.dataframe(missing_df, hide_index=True)

col1, col2 = st.columns(2)
with col1:
    fig = px.bar(
        missing_df, x="컬럼", y="결측치 수",
        title="컬럼별 결측치 수",
        text="결측치 수",
        color_discrete_sequence=["tomato"],
    )
    fig.update_layout(xaxis_title="컬럼", yaxis_title="결측치 개수")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    sample = df.sample(min(2000, len(df)), random_state=0)
    miss_matrix = sample.isnull().astype(int)
    fig = go.Figure(data=go.Heatmap(
        z=miss_matrix.values,
        x=miss_matrix.columns,
        colorscale=[[0, "lightgray"], [1, "red"]],
        showscale=False,
        hovertemplate="컬럼: %{x}<br>행 위치: %{y}<br>결측(1=결측): %{z}<extra></extra>",
    ))
    fig.update_layout(
        title="결측치 히트맵 (2,000행 샘플, 빨간색 = 결측)",
        xaxis_title="컬럼",
        yaxis_title="행 (샘플)",
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------- 3. 이상치 ----------------
st.header("3. 이상치 분석")
num_cols = df.select_dtypes(include=np.number).columns.tolist()
if "고객번호" in num_cols:
    num_cols.remove("고객번호")

method = st.radio("이상치 판단 기준", ["IQR 방법", "Z-score 방법"], horizontal=True)
if method == "IQR 방법":
    k = st.slider("IQR 배수 (보통 1.5)", 1.0, 3.0, 1.5, 0.1)
else:
    z_thr = st.slider("Z-score 임계값 (보통 3)", 1.0, 5.0, 3.0, 0.1)


def find_outliers(series):
    s = series.dropna()
    if method == "IQR 방법":
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - k * iqr, q3 + k * iqr
    else:
        mean, std = s.mean(), s.std()
        lower, upper = mean - z_thr * std, mean + z_thr * std
    mask = ((series < lower) | (series > upper)).fillna(False)
    return mask, lower, upper


summary = []
for col in num_cols:
    mask, lo, up = find_outliers(df[col])
    summary.append({
        "컬럼": col,
        "이상치 수": int(mask.sum()),
        "이상치 비율(%)": round(mask.mean() * 100, 3),
        "하한": round(lo, 2),
        "상한": round(up, 2),
    })
st.dataframe(pd.DataFrame(summary), hide_index=True)

selected = st.selectbox("자세히 볼 컬럼 선택", num_cols)
mask, lo, up = find_outliers(df[selected])

col1, col2, col3 = st.columns(3)
with col1:
    fig = px.box(df, y=selected, title=f"{selected} 박스플롯",
                 color_discrete_sequence=["skyblue"])
    fig.update_layout(yaxis_title=selected)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.histogram(df, x=selected, nbins=50, title=f"{selected} 히스토그램",
                       color_discrete_sequence=["mediumseagreen"])
    fig.add_vline(x=lo, line_dash="dash", line_color="red",
                  annotation_text="하한", annotation_position="top left")
    fig.add_vline(x=up, line_dash="dash", line_color="red",
                  annotation_text="상한", annotation_position="top right")
    fig.update_layout(xaxis_title=selected, yaxis_title="빈도")
    st.plotly_chart(fig, use_container_width=True)

with col3:
    plot_df = df.sample(min(5000, len(df)), random_state=0).copy()
    plot_mask, _, _ = find_outliers(plot_df[selected])
    plot_df["구분"] = np.where(plot_mask, "이상치", "정상")
    plot_df["행 번호"] = plot_df.index
    fig = px.scatter(
        plot_df, x="행 번호", y=selected, color="구분",
        color_discrete_map={"정상": "steelblue", "이상치": "red"},
        title=f"{selected} 산점도 (5,000행 샘플)",
        opacity=0.6,
    )
    fig.update_layout(xaxis_title="행 번호", yaxis_title=selected, legend_title="구분")
    st.plotly_chart(fig, use_container_width=True)

if st.checkbox("이상치로 판정된 행 보기"):
    st.write(f"이상치로 판정된 행: {int(mask.sum()):,}개 (최대 500행 표시)")
    st.dataframe(df[mask].head(500))

# ---------------- 4. 논리적 이상 ----------------
st.header("4. 논리적 이상 점검 (총요금 ≈ 가입기간 × 월요금)")
tmp = df.copy()
tmp["예상총액"] = tmp["가입기간(개월)"] * tmp["월요금"]
tmp["차이비율"] = (tmp["총요금"] - tmp["예상총액"]) / tmp["예상총액"].replace(0, np.nan)
diff = tmp["차이비율"].dropna().clip(-1, 1).to_frame(name="차이비율")
fig = px.histogram(diff, x="차이비율", nbins=60,
                   title="예상 총액 대비 실제 총요금 차이 비율",
                   color_discrete_sequence=["orchid"])
fig.update_layout(xaxis_title="차이 비율 (실제-예상)/예상", yaxis_title="빈도")
st.plotly_chart(fig, use_container_width=True)

# ---------------- 5. 범주형 ----------------
st.header("5. 범주형 컬럼 값 분포 (오타·이상 범주 확인)")
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
cat_sel = st.selectbox("범주형 컬럼 선택", cat_cols)
counts = df[cat_sel].value_counts(dropna=False).reset_index()
counts.columns = [cat_sel, "개수"]
counts[cat_sel] = counts[cat_sel].astype(str)
fig = px.bar(counts, x=cat_sel, y="개수", text="개수",
             title=f"{cat_sel} 값 분포",
             color_discrete_sequence=["steelblue"])
fig.update_layout(xaxis_title=cat_sel, yaxis_title="개수")
st.plotly_chart(fig, use_container_width=True)
