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
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    recall_score, f1_score, precision_score, accuracy_score,
    confusion_matrix
)

st.set_page_config(page_title="고객 이탈 분류 모델", layout="wide")
st.title("📡 어떤 데이터가 통신사 고객 이탈로 이어지는가?")
st.caption("분석 방식: 베이스라인 · 로지스틱 회귀 · 의사결정트리 · 랜덤 포레스트 | 평가 지표: 재현율(Recall), F1")

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
threshold = st.sidebar.slider("이탈 판정 임계값 (확률)", 0.1, 0.9, 0.5, 0.05)
st.sidebar.caption("베이스라인은 항상 최빈 클래스로 예측하므로 설정의 영향을 받지 않습니다.")

st.sidebar.subheader("로지스틱 회귀")
C_value = st.sidebar.select_slider(
    "C (규제 강도의 역수, 작을수록 규제 강함)",
    options=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    value=1.0,
)
penalty_kor = st.sidebar.selectbox("규제 방식 (penalty)", ["l2", "l1", "없음"])
lr_max_iter = st.sidebar.slider("최대 반복 횟수 (max_iter)", 100, 3000, 1000, 100)

st.sidebar.subheader("의사결정트리")
max_depth = st.sidebar.slider("트리 최대 깊이", 2, 15, 5)
min_leaf = st.sidebar.slider("리프 노드 최소 샘플 수", 1, 200, 50)

st.sidebar.subheader("랜덤 포레스트")
rf_trees = st.sidebar.slider("트리 개수", 50, 300, 100, 50)
rf_depth = st.sidebar.slider("포레스트 트리 최대 깊이", 3, 20, 8)
rf_leaf = st.sidebar.slider("포레스트 리프 최소 샘플 수", 1, 200, 50)

X = pd.get_dummies(df[num_features + cat_features], columns=cat_features, drop_first=False)
y = df["이탈여부"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

cw = "balanced" if use_balance else None


@st.cache_resource(show_spinner="모델 학습 중...")
def train_models(_Xtr_sc, _Xtr, _ytr, cw, test_size,
                 C_value, penalty_kor, lr_max_iter,
                 max_depth, min_leaf, rf_trees, rf_depth, rf_leaf):
    # 베이스라인: 훈련 데이터에서 가장 많은 클래스로 무조건 예측
    base = DummyClassifier(strategy="most_frequent")
    base.fit(_Xtr, _ytr)

    if penalty_kor == "l1":
        lr = LogisticRegression(C=C_value, penalty="l1", solver="saga",
                                max_iter=lr_max_iter, class_weight=cw)
    elif penalty_kor == "l2":
        lr = LogisticRegression(C=C_value, penalty="l2", solver="lbfgs",
                                max_iter=lr_max_iter, class_weight=cw)
    else:
        lr = LogisticRegression(penalty=None, solver="lbfgs",
                                max_iter=lr_max_iter, class_weight=cw)
    lr.fit(_Xtr_sc, _ytr)

    dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_leaf,
                                class_weight=cw, random_state=42)
    dt.fit(_Xtr, _ytr)

    rf = RandomForestClassifier(n_estimators=rf_trees, max_depth=rf_depth,
                                min_samples_leaf=rf_leaf, class_weight=cw,
                                random_state=42, n_jobs=-1)
    rf.fit(_Xtr, _ytr)
    return base, lr, dt, rf


base, lr, dt, rf = train_models(X_train_sc, X_train, y_train, cw, test_size,
                                C_value, penalty_kor, lr_max_iter,
                                max_depth, min_leaf, rf_trees, rf_depth, rf_leaf)

if penalty_kor == "없음":
    st.sidebar.caption("규제 없음일 때는 C 값이 사용되지 않습니다.")

base_pred = base.predict(X_test)  # 임계값과 무관하게 항상 최빈 클래스
lr_proba = lr.predict_proba(X_test_sc)[:, 1]
dt_proba = dt.predict_proba(X_test)[:, 1]
rf_proba = rf.predict_proba(X_test)[:, 1]

preds = {
    "베이스라인(최빈값)": base_pred,
    "로지스틱 회귀": (lr_proba >= threshold).astype(int),
    "의사결정트리": (dt_proba >= threshold).astype(int),
    "랜덤 포레스트": (rf_proba >= threshold).astype(int),
}

majority_label = "유지" if int(base.classes_[np.argmax(base.class_prior_)]) == 0 else "이탈"
st.info(f"🧪 베이스라인 모델은 모든 고객을 **'{majority_label}'** 으로 예측합니다 "
        f"(훈련 데이터에서 가장 많은 클래스).")

st.subheader("현재 로지스틱 회귀 하이퍼파라미터")
st.dataframe(pd.DataFrame({
    "하이퍼파라미터": ["C", "penalty", "solver", "max_iter", "class_weight"],
    "값": [
        C_value if penalty_kor != "없음" else "사용 안 함",
        penalty_kor,
        lr.solver,
        lr_max_iter,
        str(cw),
    ],
}), hide_index=True)

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

# 최고 모델은 베이스라인을 제외하고 선정
result_models = result.drop(index="베이스라인(최빈값)")
best_f1 = result_models["F1"].idxmax()
best_recall = result_models["재현율(Recall)"].idxmax()
st.success(f"🏆 F1이 가장 높은 모델: **{best_f1}** | 재현율이 가장 높은 모델: **{best_recall}** "
           f"(베이스라인 제외)")

# 베이스라인 대비 향상 정도
base_row = result.loc["베이스라인(최빈값)"]
gain = (result_models - base_row).round(4)
st.subheader("베이스라인 대비 향상 정도 (모델 점수 − 베이스라인 점수)")
st.dataframe(gain)

res_long = result.reset_index().melt(id_vars="index", var_name="지표", value_name="점수")
res_long.columns = ["모델", "지표", "점수"]
fig = px.bar(res_long, x="지표", y="점수", color="모델", barmode="group",
             text_auto=".3f", title="모델별 성능 지표 비교 (베이스라인 포함)")
fig.update_layout(yaxis_range=[0, 1])
st.plotly_chart(fig, use_container_width=True)

st.info(
    "💡 **재현율**은 실제 이탈 고객 중 모델이 찾아낸 비율, "
    "**F1**은 재현율과 정밀도의 조화평균입니다. "
    "이탈 예측에서는 이탈할 고객을 놓치지 않는 것이 중요해서 재현율을 중요하게 봅니다. "
    "베이스라인의 정확도와 비교해, 정확도만으로는 모델의 가치를 판단하기 어렵다는 점도 확인해 보세요."
)

st.subheader("혼동 행렬")
cols = st.columns(4)
for col, (name, pred) in zip(cols, preds.items()):
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
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
    "모델별 중요도 비교", "의사결정트리 구조"
])

with tab1:
    coef = pd.DataFrame({"변수": X.columns, "계수": lr.coef_[0]})
    coef["영향"] = np.where(coef["계수"] > 0, "이탈 증가", "이탈 감소")
    coef = coef.reindex(coef["계수"].abs().sort_values().index)
    fig = px.bar(coef, x="계수", y="변수", orientation="h", color="영향",
                 color_discrete_map={"이탈 증가": "tomato", "이탈 감소": "steelblue"},
                 title="로지스틱 회귀 계수 (표준화 기준, 양수=이탈 증가)")
    st.plotly_chart(fig, use_container_width=True)
    n_zero = int((lr.coef_[0] == 0).sum())
    st.caption(f"수치형 변수는 표준화되어 있어 계수 크기를 서로 비교할 수 있습니다. "
               f"현재 계수가 0인 변수: {n_zero}개 (l1 규제에서 변수가 제거되면 늘어납니다.)")

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
    abs_coef = np.abs(lr.coef_[0])
    coef_ratio = abs_coef / abs_coef.sum() if abs_coef.sum() > 0 else abs_coef
    comp = pd.DataFrame({
        "변수": X.columns,
        "로지스틱 회귀(|계수| 비율)": coef_ratio,
        "의사결정트리": dt.feature_importances_,
        "랜덤 포레스트": rf.feature_importances_,
    })
    comp_long = comp.melt(id_vars="변수", var_name="모델", value_name="상대 중요도")
    fig = px.bar(comp_long, x="상대 중요도", y="변수", color="모델", barmode="group",
                 orientation="h", title="모델별 변수 중요도 비교 (합이 1이 되도록 정규화)",
                 height=700)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("세 모델이 공통으로 중요하게 보는 변수가 이탈의 핵심 요인일 가능성이 높습니다. "
               "(베이스라인은 변수를 전혀 쓰지 않으므로 중요도가 없습니다.)")

with tab5:
    show_depth = st.slider("표시할 트리 깊이", 1, 4, 3)
    fig = plotly_tree(dt, list(X.columns), max_depth_show=show_depth)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("노드에 마우스를 올리면 분기 조건, 샘플 수, 이탈 확률이 표시됩니다. "
               "조건이 참이면 '예'(왼쪽), 거짓이면 '아니오'(오른쪽)로 갑니다. "
               "원-핫 인코딩된 변수(예: 계약형태_월 단위)는 0 또는 1 값이라 '≤ 0.50'은 '해당하지 않음'을 뜻합니다.")

    st.subheader("트리가 만든 규칙 표")
    rules = tree_rules_text(dt, list(X.columns))
    st.dataframe(rules.sort_values("이탈 확률(%)", ascending=False),
                 hide_index=True, use_container_width=True)

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

p_base = base.predict(new_enc)[0]
p_lr = lr.predict_proba(scaler.transform(new_enc))[0, 1]
p_dt = dt.predict_proba(new_enc)[0, 1]
p_rf = rf.predict_proba(new_enc)[0, 1]
m0, m1, m2, m3 = st.columns(4)
m0.metric("베이스라인 예측", "이탈" if p_base == 1 else "유지")
m1.metric("로지스틱 회귀 이탈 확률", f"{p_lr*100:.1f}%")
m2.metric("의사결정트리 이탈 확률", f"{p_dt*100:.1f}%")
m3.metric("랜덤 포레스트 이탈 확률", f"{p_rf*100:.1f}%")

# ---------------- 6. 로지스틱 회귀 최적 하이퍼파라미터 탐색 ----------------
st.header("6. 로지스틱 회귀 최적 하이퍼파라미터 탐색 (GridSearchCV)")
st.write("교차검증으로 여러 조합을 비교해 가장 좋은 값을 찾습니다. "
         "**훈련 데이터만** 사용하고, 테스트 데이터는 마지막 확인에만 씁니다.")

g1, g2, g3 = st.columns(3)
scoring_kor = g1.selectbox("최적화 기준", ["F1", "재현율(Recall)"])
cv_folds = g2.slider("교차검증 분할 수 (cv)", 3, 5, 3)
search_n = g3.slider("탐색에 사용할 훈련 샘플 수", 5000, len(X_train), min(20000, len(X_train)), 5000)

search_C = st.multiselect("탐색할 C 값", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                          default=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
search_penalty = st.multiselect("탐색할 규제 방식", ["l2", "l1"], default=["l2", "l1"])
search_cw = st.multiselect("탐색할 class_weight", ["balanced", "없음"], default=["balanced", "없음"])

if st.button("🔍 최적 하이퍼파라미터 찾기"):
    if not (search_C and search_penalty and search_cw):
        st.warning("각 항목에서 최소 1개 이상 선택해 주세요.")
    else:
        scoring = "f1" if scoring_kor == "F1" else "recall"

        if search_n < len(X_train):
            Xs, _, ys, _ = train_test_split(
                X_train_sc, y_train, train_size=search_n,
                random_state=42, stratify=y_train
            )
        else:
            Xs, ys = X_train_sc, y_train

        param_grid = {
            "C": search_C,
            "penalty": search_penalty,
            "class_weight": [None if c == "없음" else c for c in search_cw],
        }
        n_combo = len(search_C) * len(search_penalty) * len(search_cw)
        st.write(f"총 {n_combo}개 조합 × {cv_folds}-fold 교차검증 = {n_combo * cv_folds}번 학습")

        grid = GridSearchCV(
            LogisticRegression(solver="saga", max_iter=lr_max_iter),
            param_grid,
            scoring=scoring,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            n_jobs=-1,
            return_train_score=True,
        )
        with st.spinner("탐색 중입니다. 잠시만 기다려 주세요..."):
            grid.fit(Xs, ys)

        st.session_state["grid"] = grid
        st.session_state["scoring_kor"] = scoring_kor

if "grid" in st.session_state:
    grid = st.session_state["grid"]
    scoring_kor_done = st.session_state["scoring_kor"]
    best = grid.best_params_

    st.success(f"✅ 최적 파라미터: **{best}**")
    st.metric(f"교차검증 {scoring_kor_done} (평균)", f"{grid.best_score_:.4f}")

    cvr = pd.DataFrame(grid.cv_results_)
    table = pd.DataFrame({
        "C": cvr["param_C"],
        "penalty": cvr["param_penalty"],
        "class_weight": cvr["param_class_weight"].astype(str),
        "검증 점수(평균)": cvr["mean_test_score"].round(4),
        "검증 점수(표준편차)": cvr["std_test_score"].round(4),
        "훈련 점수(평균)": cvr["mean_train_score"].round(4),
        "순위": cvr["rank_test_score"],
    }).sort_values("순위")
    st.dataframe(table, hide_index=True, use_container_width=True)

    plot_df = table.copy()
    plot_df["C"] = plot_df["C"].astype(float)
    plot_df["설정"] = plot_df["penalty"] + " / " + plot_df["class_weight"]
    fig = px.line(plot_df.sort_values("C"), x="C", y="검증 점수(평균)", color="설정",
                  markers=True, log_x=True,
                  title=f"C 값에 따른 교차검증 {scoring_kor_done} 점수")
    fig.update_layout(xaxis_title="C (로그 눈금)", yaxis_title=f"{scoring_kor_done}")
    st.plotly_chart(fig, use_container_width=True)

    best_lr = LogisticRegression(
        C=best["C"], penalty=best["penalty"], solver="saga",
        class_weight=best["class_weight"], max_iter=lr_max_iter
    )
    best_lr.fit(X_train_sc, y_train)
    best_pred = (best_lr.predict_proba(X_test_sc)[:, 1] >= threshold).astype(int)
    cur_pred = preds["로지스틱 회귀"]

    compare = pd.DataFrame({
        "베이스라인(최빈값)": metrics(y_test, base_pred),
        "현재 설정": metrics(y_test, cur_pred),
        "탐색으로 찾은 최적 설정": metrics(y_test, best_pred),
    }).T.round(4)
    st.subheader("테스트 데이터 성능: 베이스라인 vs 현재 설정 vs 최적 설정")
    st.dataframe(compare)
    st.caption("최적 설정을 사이드바에 직접 입력해 보면 위 3번 성능 비교에도 반영됩니다. "
               "(class_weight는 사이드바의 '클래스 불균형 보정' 체크박스로 설정)")
