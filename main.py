import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="이상치·결측치 탐색", layout="wide")
st.title("📊 고객 이탈 데이터: 결측치 & 이상치 탐색")

FILE_PATH = "synthetic_customer_churn_100k.csv"


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


try:
    df = load_data(FILE_PATH)
except FileNotFoundError:
    st.error(f"'{FILE_PATH}' 파일을 찾을 수 없습니다. app.py와 같은 폴더에 두세요.")
    st.stop()

st.subheader("1. 데이터 개요")
c1, c2, c3 = st.columns(3)
c1.metric("행 수", f"{df.shape[0]:,}")
c2.metric("열 수", df.shape[1])
c3.metric("전체 결측치 수", f"{int(df.isnull().sum().sum()):,}")
st.dataframe(df.head(20))

# ---------------- 결측치 ----------------
st.header("2. 결측치 분석")
missing = df.isnull().sum()
missing_df = pd.DataFrame({
    "결측치 수": missing,
    "결측 비율(%)": (missing / len(df) * 100).round(3),
})
st.dataframe(missing_df)

col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(missing.index, missing.values, color="tomato")
    ax.set_title("컬럼별 결측치 수")
    ax.set_ylabel("개수")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(7, 4))
    sample = df.sample(min(2000, len(df)), random_state=0)
    sns.heatmap(sample.isnull(), cbar=False, yticklabels=False, ax=ax)
    ax.set_title("결측치 히트맵 (2,000행 샘플, 밝은 부분=결측)")
    st.pyplot(fig)

# ---------------- 이상치 ----------------
st.header("3. 이상치 분석")
num_cols = df.select_dtypes(include=np.number).columns.tolist()
if "CustomerID" in num_cols:
    num_cols.remove("CustomerID")

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
        mask = (series < lower) | (series > upper)
    else:
        mean, std = s.mean(), s.std()
        lower, upper = mean - z_thr * std, mean + z_thr * std
        mask = (series < lower) | (series > upper)
    return mask.fillna(False), lower, upper


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
st.dataframe(pd.DataFrame(summary))

selected = st.selectbox("자세히 볼 컬럼 선택", num_cols)
mask, lo, up = find_outliers(df[selected])

col1, col2, col3 = st.columns(3)
with col1:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.boxplot(y=df[selected], ax=ax, color="skyblue")
    ax.set_title(f"{selected} 박스플롯")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(df[selected].dropna(), bins=50, color="mediumseagreen", edgecolor="white")
    ax.axvline(lo, color="red", linestyle="--", label="하한")
    ax.axvline(up, color="red", linestyle="--", label="상한")
    ax.set_title(f"{selected} 히스토그램")
    ax.legend()
    st.pyplot(fig)

with col3:
    fig, ax = plt.subplots(figsize=(5, 4))
    plot_df = df.sample(min(5000, len(df)), random_state=0)
    plot_mask, _, _ = find_outliers(plot_df[selected])
    ax.scatter(plot_df.index[~plot_mask], plot_df[selected][~plot_mask],
               s=3, alpha=0.4, label="정상")
    ax.scatter(plot_df.index[plot_mask], plot_df[selected][plot_mask],
               s=8, color="red", label="이상치")
    ax.set_title(f"{selected} 산점도 (5,000행 샘플)")
    ax.legend()
    st.pyplot(fig)

if st.checkbox("이상치로 판정된 행 보기"):
    st.dataframe(df[mask].head(500))

# ---------------- 논리적 이상 ----------------
st.header("4. 논리적 이상 점검 (TotalCharges ≈ Tenure × MonthlyCharges)")
tmp = df.copy()
tmp["예상총액"] = tmp["Tenure"] * tmp["MonthlyCharges"]
tmp["차이비율"] = (tmp["TotalCharges"] - tmp["예상총액"]) / tmp["예상총액"].replace(0, np.nan)
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(tmp["차이비율"].dropna().clip(-1, 1), bins=60, color="orchid", edgecolor="white")
ax.set_title("예상 총액 대비 실제 TotalCharges 차이 비율")
st.pyplot(fig)

# ---------------- 범주형 ----------------
st.header("5. 범주형 컬럼 값 분포 (오타·이상 범주 확인)")
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
cat_sel = st.selectbox("범주형 컬럼 선택", cat_cols)
fig, ax = plt.subplots(figsize=(7, 4))
df[cat_sel].value_counts(dropna=False).plot(kind="bar", ax=ax, color="steelblue")
ax.set_title(f"{cat_sel} 값 분포")
plt.xticks(rotation=0)
st.pyplot(fig)
