import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    recall_score, f1_score, precision_score, accuracy_score,
    confusion_matrix
)

st.set_page_config(page_title="고객 이탈 분류 모델", layout="wide")
st.title("📡 어떤 데이터가 통신사 고객 이탈로 이어지는가?")
st.caption("분석 방식: 로지스틱 회귀 · 의사결정트리 · 랜덤 포레스트 | 평가 지표: 재현율(Recall), F1")

FILE_PATH = "synthetic_customer_churn_100k.csv"

KOR_COLS = {
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
}


@st.cache_data
def load_data(path):
    data = pd.read_csv(path)
    data = data.drop(columns=["CustomerID"])
    for col, mapping in KOR_VALUES.items():
        data[col] = data[col].replace(mapping)
    data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})
    return data.rename(columns=KOR_COLS)


try:
    df = load_data(FILE_PATH)
except FileNotFoundError:
    st.error(f"'{FILE_PATH}' 파일을 찾을 수 없습니다. main.py와 같은 폴더에 두세요.")
    st.stop()

# 결측치 처리 (있을 경우 대비)
num_features = ["나이", "가입기간(개월)", "월요금", "총요금"]
cat_features = ["성별", "계약형태", "결제방식"]
for c in num_features:
    df[c] = df[c].fillna(df[c].median())
for c in cat_features:
    df[c] = df[c].fillna(df[c].mode()[0])
df = df.dropna(subset=["이탈여부"])


# ---------------- 트리 시각화 함수 (Plotly) ----------------
def plotly_tree(model, feature_names, max_depth_show=3):
    t = model.tree_
    positions = {}
    visible = []
    counter = {"x": 0}

    def layout(node, depth):
        if depth > max_depth_show:
            return None
        left, right = t.children_left[node], t.children_right[node]
        is_leaf = left == -1 or depth == max_depth_show
        visible.append(node)
        if is_leaf:
            x = counter["x"]
            counter["x"] += 1
        else:
            lx = layout(left, depth + 1)
            rx = layout(right, depth + 1)
            x = (lx + rx) / 2
        positions[node] = (x, -depth)
        return x

    layout(0, 0)

    fig = go.Figure()

    for node in visible:
        left, right = t.children_left[node], t.children_right[node]
        if left == -1:
            continue
        for child, label in [(left, "예"), (right, "아니오")]:
            if child in positions:
                x0, y0 = positions[node]
                x1, y1 = positions[child]
                fig.add_trace(go.Scatter(
                    x=[x0, x1], y=[y0, y1], mode="lines",
                    line=dict(color="gray", width=1),
                    hoverinfo="skip", showlegend=False,
                ))
                fig.add_annotation(
                    x=(x0 + x1) / 2, y=(y0 + y1) / 2, text=label,
                    showarrow=False, font=dict(size=10, color="dimgray"),
                )

    xs, ys, colors, texts, hovers = [], [], [], [], []
    for node in visible:
        x, y = positions[node]
        counts = t.value[node][0]
        total = counts.sum()
        churn_prob = counts[1] / total if total > 0 else 0
        n_samples = int(t.n_node_samples[node])
        left = t.children_left[node]
        is_split = left != -1 and left in positions

        if is_split:
            fname = feature_names[t.feature[node]]
            cond = f"{fname} ≤ {t.threshold[node]:.2f}"
        else:
            cond = "마지막 노드(리프)"

        xs.append(x)
        ys.append(y)
        colors.append(churn_prob)
        texts.append(f"{cond}<br>이탈확률 {churn_prob*100:.1f}%" if is_split
                     else f"이탈확률<br>{churn_prob*100:.1f}%")
        hovers.append(
            f"<b>{cond}</b><br>샘플 수: {n_samples:,}<br>"
            f"이탈 확률: {churn_prob*100:.1f}%<br>"
            f"예측: {'이탈' if churn_prob >= 0.5 else '유지'}"
        )

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text",
        marker=dict(
            size=34, color=colors, colorscale="RdBu_r", cmin=0, cmax=1,
            line=dict(width=1, color="black"),
            colorbar=dict(title="이탈 확률"),
        ),
        text=texts, textposition="top center", textfont=dict(size=10),
        hovertext=hovers, hoverinfo="text", showlegend=False,
    ))

    fig.update_layout(
        title="의사결정트리 구조 (노드 색: 빨강=이탈 확률 높음, 파랑=낮음)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=250 * (max_depth_show + 1),
        plot_bgcolor="white",
        margin=dict(t=60, b=20),
    )
    return fig


def tree_rules_text(model, feature_names):
    t = model.tree_
    rules = []

    def walk(node, conds):
        left, right = t.children_left[node], t.children_right[node]
        if left == -1:
            counts = t.value[node][0]
            total = counts.sum()
            prob = counts[1] / total if total > 0 else 0
            rules.append({
                "규칙": " 그리고 ".join(conds) if conds else "(전체)",
                "샘플 수": int(t.n_node_samples[node]),
                "이탈 확률(%)": round(prob * 100, 1),
                "예측": "이탈" if prob >= 0.5 else "유지",
            })
            return
        name = feature_names[t.feature[node]]
        thr = t.threshold[node]
        walk(left, conds + [f"{name} ≤ {thr:.2f}"])
        walk(right, conds + [f"{name} > {thr:.2f}"])

    walk(0, [])
    return pd.DataFrame(rules)


# ---------------- 1. 데이터 탐색 ----------------
st.header("1. 이탈 현황 탐색")
c1, c2, c3 = st.columns(3)
churn_rate = df["이탈여부"].mean() * 100
c1.metric("전체 고객 수", f"{len(df):,}")
c2.metric("이탈 고객 수", f"{int(df['이탈여부'].sum()):,}")
c3.metric("이탈률", f"{churn_rate:.1f}%")

df_view = df.copy()
df_view["이탈"] = df_view["이탈여부"].map({1: "이탈", 0: "유지"})

col1, col2 = st.columns(2)
with col1:
    cnt = df_view["이탈"].value_counts().reset_index()
    cnt.columns = ["이탈", "고객 수"]
    fig = px.pie(cnt, names="이탈", values="고객 수", title="이탈 vs 유지 비율",
                 color="이탈", color_discrete_map={"이탈": "tomato", "유지": "steelblue"})
    st.plotly_chart(fig, use_container_width=True)
with col2:
    cat_sel = st.selectbox("범주형 변수별 이탈률 보기", cat_features)
    rate = df.groupby(cat_sel)["이탈여부"].mean().mul(100).round(2).reset_index()
    rate.columns = [cat_sel, "이탈률(%)"]
    fig = px.bar(rate, x=cat_sel, y="이탈률(%)", text="이탈률(%)",
                 title=f"{cat_sel}별 이탈률", color_discrete_sequence=["tomato"])
    st.plotly_chart(fig, use_container_width=True)

num_sel = st.selectbox("수치형 변수 분포 비교 (이탈 vs 유지)", num_features)
fig = px.box(df_view, x="이탈", y=num_sel, color="이탈",
             color_discrete_map={"이탈": "tomato", "유지": "steelblue"},
             title=f"{num_sel} 분포: 이탈 vs 유지")
st.plotly_chart(fig, use_container_width=True)

# ---------------- 2. 모델 설정 ----------------
st.header("2. 모델 학습 설정")
st.sidebar.header("⚙️ 모델 설정")
test_size = st.sidebar.slider("테스트 데이터 비율", 0.1, 0.4, 0.2, 0.05)
use_balance = st.sidebar.checkbox("클래스 불균형 보정 (class_weight='balanced')", value=True)
threshold = st.sidebar.slider("이탈 판정 임계값 (확률)", 0.1, 0.9, 0.5, 0.05
