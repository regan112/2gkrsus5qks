import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
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
    data = data.drop(columns=["CustomerID"])  # 식별자는 예측에 불필요
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
threshold = st.sidebar.slider("이탈 판정 임계값 (확률)", 0.1, 0.9, 0.5, 0.05)

st.sidebar.subheader("의사결정트리")
max_depth = st.sidebar.slider("트리 최대 깊이", 2, 15, 5)
min_leaf = st.sidebar.slider("리프 노드 최소 샘플 수", 1, 200, 50)

st.sidebar.subheader("랜덤 포레스트")
rf_trees = st.sidebar.slider("트리 개수", 50, 300, 100, 50)
rf_depth = st.sidebar.slider("포레스트 트리 최대 깊이", 3, 20, 8)
rf_leaf = st.sidebar.slider("포레스트 리프 최소 샘플 수", 1, 200, 50)

# 원-핫 인코딩
X = pd.get_dummies(df[num_features + cat_features], columns=cat_features, drop_first=False)
y = df["이탈여부"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)

# 로지스틱 회귀는 스케일링 필요
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

cw = "balanced" if use_balance else None


@st.cache_resource(show_spinner="모델 학습 중...")
def train_models(_Xtr_sc, _Xtr, _ytr, cw, test_size, max_depth, min_leaf,
                 rf_trees, rf_depth, rf_leaf):
    lr = LogisticRegression(max_iter=1000, class_weight=cw)
    lr.fit(_Xtr_sc, _ytr)

    dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_leaf,
                                class_weight=cw, random_state=42)
    dt.fit(_Xtr, _ytr)

    rf = RandomForestClassifier(n_estimators=rf_trees, max_depth=rf_depth,
                                min_samples_leaf=rf_leaf, class_weight=cw,
                                random_state=42, n_jobs=-1)
    rf.fit(_Xtr, _ytr)
    return lr, dt, rf


lr, dt, rf = train_models(X_train_sc, X_train, y_train, cw, test_size,
                          max_depth, min_leaf, rf_trees, rf_depth, rf_leaf)

lr_proba = lr.predict_proba(X_test_sc)[:, 1]
dt_proba = dt.predict_proba(X_test)[:, 1]
rf_proba = rf.predict_proba(X_test)[:, 1]

preds = {
    "로지스틱 회귀": (lr_proba >= threshold).astype(int),
    "의사결정트리": (dt_proba >= threshold).astype(int),
    "랜덤 포레스트": (rf_proba >= threshold).astype(int),
}

# ---------------- 3. 성능 비교 ----------------
st.header("3. 모델 성능 비교 (재현율 · F1 중심)")


def metrics(y_true, y_pred):
    return {
        "재현율(Recall)": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "정밀도(Precision)": precision_score(y_true, y_pred, zero_division=0),
        "정확도(Accuracy)": accuracy_score(y_true, y_pred),
    }


result = pd.DataFrame({name: metrics(y_test, p) for name, p in preds.items()}).T.round(4)
st.dataframe(result)

best_f1 = result["F1"].idxmax()
best_recall = result["재현율(Recall)"].idxmax()
st.success(f"🏆 F1이 가장 높은 모델: **{best_f1}** | 재현율이 가장 높은 모델: **{best_recall}**")

res_long = result.reset_index().melt(id_vars="index", var_name="지표", value_name="점수")
res_long.columns = ["모델", "지표", "점수"]
fig = px.bar(res_long, x="지표", y="점수", color="모델", barmode="group",
             text_auto=".3f", title="모델별 성능 지표 비교")
fig.update_layout(yaxis_range=[0, 1])
st.plotly_chart(fig, use_container_width=True)

st.info(
    "💡 **재현율**은 실제 이탈 고객 중 모델이 찾아낸 비율, "
    "**F1**은 재현율과 정밀도의 조화평균입니다. "
    "이탈 예측에서는 이탈할 고객을 놓치지 않는 것이 중요해서 재현율을 중요하게 봅니다."
)

# 혼동행렬
st.subheader("혼동 행렬")
cols = st.columns(3)
for col, (name, pred) in zip(cols, preds.items()):
    cm = confusion_matrix(y_test, pred)
    fig = px.imshow(
        cm, text_auto=True, color_continuous_scale="Blues",
        x=["예측: 유지", "예측: 이탈"], y=["실제: 유지", "실제: 이탈"],
        title=f"{name}",
    )
    fig.update_layout(coloraxis_showscale=False)
    col.plotly_chart(fig, use_container_width=True)

# ---------------- 4. 이탈 요인 해석 ----------------
st.header("4. 어떤 데이터가 이탈로 이어지는가?")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "로지스틱 회귀 계수", "의사결정트리 중요도", "랜덤 포레스트 중요도",
    "모델별 중요도 비교", "트리 구조"
])

with tab1:
    coef = pd.DataFrame({"변수": X.columns, "계수": lr.coef_[0]})
    coef["영향"] = np.where(coef["계수"] > 0, "이탈 증가", "이탈 감소")
    coef = coef.reindex(coef["계수"].abs().sort_values().index)
    fig = px.bar(coef, x="계수", y="변수", orientation="h", color="영향",
                 color_discrete_map={"이탈 증가": "tomato", "이탈 감소": "steelblue"},
                 title="로지스틱 회귀 계수 (표준화 기준, 양수=이탈 증가)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("수치형 변수는 표준화되어 있어 계수 크기를 서로 비교할 수 있습니다.")

with tab2:
    imp = pd.DataFrame({"변수": X.columns, "중요도": dt.feature_importances_}).sort_values("중요도")
    fig = px.bar(imp, x="중요도", y="변수", orientation="h",
                 title="의사결정트리 변수 중요도", color_discrete_sequence=["seagreen"])
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    imp_rf = pd.DataFrame({"변수": X.columns, "중요도": rf.feature_importances_}).sort_values("중요도")
    fig = px.bar(imp_rf, x="중요도", y="변수", orientation="h",
                 title="랜덤 포레스트 변수 중요도", color_discrete_sequence=["darkorange"])
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    comp = pd.DataFrame({
        "변수": X.columns,
        "로지스틱 회귀(|계수| 비율)": np.abs(lr.coef_[0]) / np.abs(lr.coef_[0]).sum(),
        "의사결정트리": dt.feature_importances_,
        "랜덤 포레스트": rf.feature_importances_,
    })
    comp_long = comp.melt(id_vars="변수", var_name="모델", value_name="상대 중요도")
    fig = px.bar(comp_long, x="상대 중요도", y="변수", color="모델", barmode="group",
                 orientation="h", title="모델별 변수 중요도 비교 (합이 1이 되도록 정규화)",
                 height=700)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("세 모델이 공통으로 중요하게 보는 변수가 이탈의 핵심 요인일 가능성이 높습니다.")

with tab5:
    import matplotlib.pyplot as plt
    show_depth = st.slider("표시할 트리 깊이", 1, 4, 3)
    fig, ax = plt.subplots(figsize=(20, 9))
    plot_tree(dt, feature_names=list(X.columns), class_names=["유지", "이탈"],
              max_depth=show_depth, filled=True, fontsize=8, ax=ax)
    st.pyplot(fig)
    st.caption("트리 그림은 matplotlib을 사용하므로 한글이 깨질 수 있습니다. 깨지면 이 탭은 건너뛰어도 됩니다.")

# ---------------- 5. 직접 예측 ----------------
st.header("5. 가상 고객 이탈 예측해 보기")
c1, c2, c3 = st.columns(3)
age = c1.slider("나이", 18, 80, 40)
tenure = c1.slider("가입기간(개월)", 1, 72, 24)
monthly = c2.slider("월요금", 10.0, 150.0, 70.0)
gender = c2.selectbox("성별", sorted(df["성별"].unique()))
contract = c3.selectbox("계약형태", sorted(df["계약형태"].unique()))
payment = c3.selectbox("결제방식", sorted(df["결제방식"].unique()))

new = pd.DataFrame([{
    "나이": age, "가입기간(개월)": tenure, "월요금": monthly,
    "총요금": tenure * monthly, "성별": gender,
    "계약형태": contract, "결제방식": payment,
}])
new_enc = pd.get_dummies(new, columns=cat_features).reindex(columns=X.columns, fill_value=0)

p_lr = lr.predict_proba(scaler.transform(new_enc))[0, 1]
p_dt = dt.predict_proba(new_enc)[0, 1]
p_rf = rf.predict_proba(new_enc)[0, 1]
m1, m2, m3 = st.columns(3)
m1.metric("로지스틱 회귀 이탈 확률", f"{p_lr*100:.1f}%")
m2.metric("의사결정트리 이탈 확률", f"{p_dt*100:.1f}%")
m3.metric("랜덤 포레스트 이탈 확률", f"{p_rf*100:.1f}%")
