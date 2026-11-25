import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="통신사 고객 이탈 데이터 분석 학술 포스터",
    page_icon="📊",
    layout="wide"
)

# 사이드바: 실시간 수정 기능 (기본값은 보내주신 PDF 내용과 100% 동일)
st.sidebar.header("📝 포스터 텍스트 실시간 수정")
st.sidebar.info("기본값은 업로드하신 PDF의 내용과 동일하게 세팅되어 있습니다. 필요시 수정하시면 오른쪽 포스터에 즉시 반영됩니다.")

with st.sidebar.expander("1. 상단 헤더 및 요약", expanded=False):
    title = st.text_input("포스터 제목", "어떤 데이터가 통신사 고객 이탈로 이어지는가")
    team_info = st.text_input("팀원", "팀원: 20510 박주형 20518 이우빈 20523 정은솔")
    sub_info = st.text_input("부제목", "고등학교 데이터 과학 프로젝트 포스터 (A1 규격) | 분석 방법: 분류(Classification)")
    
    sum_q = st.text_area("연구 질문", "통신사 고객 이탈(Churn)을 유발하는 핵심 데이터 요인은 무엇이며, 이를 예측하는 최적의 머신러닝 분류 모델은 무엇인가?", height=70)
    sum_m = st.text_area("데이터와 방법", "Kaggle Telco Customer Churn 데이터셋 (총 100,000건 표본)을 바탕으로 탐색적 데이터 분석(EDA)을 수행하고, 로지스틱 회귀, 의사결정 트리, 랜덤 포레스트 분류 모델을 탐색.", height=70)
    sum_r = st.text_area("핵심 결과", "계약 형태 중 월 단위(Month-to-month) 계약 고객의 이탈률이 46.56%로 약정 계약(1년 16.75%, 2년 16.88%) 대비 이탈 위험이 압도적으로 높음을 확인.", height=70)
    sum_c = st.text_area("최종 결론", "통신사 이윤 극대화를 위해서는 실제 이탈할 고객을 미리 감지하여 방어 마케팅을 펼치는 것이 핵심이므로, 모델 중 재현율(Recall: 0.702)이 가장 뛰어난 로지스틱 회귀(Logistic Regression)를 최종 모델로 선정함.", height=80)

with st.sidebar.expander("2. 서론 및 데이터", expanded=False):
    sec1_p1 = st.text_area("1. 서론 (첫째 문단)", "통신 시장은 신규 고객 유치 비용이 기존 고객 유지 비용보다 약 5배 이상 높게 발생하는 대표적인 포화 시장이다. 따라서 고객 이탈(Churn)을 사전에 차단하고 장기 유지율을 높이는 것이 통신 기업의 수익 극대화 전략에서 가장 핵심적이다.", height=90)
    sec1_p2 = st.text_area("1. 서론 (둘째 문단)", "본 프로젝트에서는 방대한 통신사 고객 데이터를 분류(Classification) 알고리즘을 통해 분석하여, 고객의 이탈 여부를 효과적으로 예측하고 이탈에 결정적인 영향을 미치는 핵심 변수를 규명하고자 하였다.", height=90)
    
    sec2_p1 = st.text_area("2. 데이터 (첫째 문단)", "Kaggle의 Telco Customer Churn 데이터셋[1]을 선정하여 분석을 진행하였다. 전체 약 100,000건의 표본 중 이탈 고객은 33144명, 유지 고객은 66856명으로 구성되어 있다.", height=80)
    sec2_p2 = st.text_area("2. 데이터 (둘째 문단)", "범주형 변수 탐색 결과, '계약 형태(Contract)'가 이탈률을 가르는 가장 결정적인 요인으로 분석되었다. 단기 계약 형태일수록 서비스 이탈 장벽이 낮아 이탈 가능성이 급증함을 나타낸다.", height=80)

with st.sidebar.expander("3. 분석 방법 및 모델 선택 이유", expanded=False):
    sec3_p1 = st.text_area("3. 분석 방법 (첫째 문단)", "이탈 여부(0: 유지, 1: 이탈)를 예측하는 이진 분류(Binary Classification) 파이프라인을 구축하였다.", height=70)
    sec3_p2 = st.text_area("3. 분석 방법 (둘째 문단)", "고려한 후보 모델은 로지스틱 회귀(Logistic Regression), 의사결정 트리(Decision Tree), 랜덤 포레스트(Random Forest)이며, 가장 최빈값을 단순 예측하는 베이스라인 모델과 성능을 비교 평가하였다.", height=80)
    
    sec5_1 = st.text_area("5. 모델 선택 이유 - ① 재현율 최우선 고려", "재현율(실제 모델이 예측한 고객/실제 참인 고객)이 가장 이탈율을 잘보여주는 고객이기 때문에 재현율을 최우선적으로 고려하였다.", height=70)
    sec5_2 = st.text_area("5. 모델 선택 이유 - ② 정밀도 및 정확도", "재현율이 낮고 정밀도 및 정확도가 높은 모델은 가격이 높은 상품의 판매량을 예측할 때 좋은 효과를 보이고 재현율이 높고 정밀도 및 정확도가 낮은 모델은 비교적 가격이 낮고 장기적인 계약 형태의 상품을 예측할 때 좋은 효과를 보이기에 정밀도 및 정확도는 비록 중요하긴 하지만 그다지 큰 비중을 두지 않았다.", height=110)
    sec5_3 = st.text_area("5. 모델 선택 이유 - ③ 선제적 방어 마케팅 가능", "로지스틱 회귀를 통해 총 33144명의 이탈 위험 고객을 선제적으로 감지하여 타깃 방어를 적용할 수 있게 되었다.", height=70)

with st.sidebar.expander("4. 한계 및 분석 결과", expanded=False):
    sec6_limit = st.text_area("프로젝트의 한계 및 향후 과제", "월 요금, 결제 방식 등 복합 변수와의 교차 작용을 해석하기 위한 하이퍼파라미터 튜닝이 추가로 필요하다. 분류만이 이를 효과적으로 보여줄 수 있는지에 대한 고민과 로지스틱 회귀가 정말로 이 탐구 주제에 가장 적합한지에 대한 의문도 해결해보아야 할 과제인 것 같다. 그리고 통신 시장에만 국한되는 것이 아닌 여러가지 시장에서 기업의 이윤을 위해서는 어떤 모델이 최선의 분석으로 이윤을 극대화 할 수 있을 지를 시장별로 비교, 분석해보고 싶다. 마지막으로 이번 모델에서는 고려되지 않은 고객 당 달라지는 이윤을 고려하지 못한 부분이 마음에 걸려 다음 번 분석에서는 이도 고려하여 기업의 이윤을 최대한 극대화 할 수 있는 모델을 만들어보고 싶다.", height=160)
    sec6_result = st.text_area("분석 결과", "단순히 정확도가 높은 모델을 선택하는 것이 아니라, \"통신사의 이윤을 극대화한다\"는 비즈니스 목적에 맞춰 재현율(Recall)을 평가 기준으로 삼는 시각의 중요성을 배웠다. 또한 기업의 이윤을 극대화하기 위해 ai 시대에서 ai를 활용해보니 어떠한 도구보다 ai가 가장 도움이 되는 것 같고 여러가지 모델들도 각각의 분야에서 적재적소에 사용한다면 엄청난 이익을 가져다줄 것 같다. 이번 분석결과를 통해서 통신 시장에서는 장기계약을 채결하는 것이 가장 중요하고 월요금제는 기업의 이윤에 별 도움이 되지 않음을 알게 되었다.", height=150)

# A1 포스터 HTML/CSS 템플릿
poster_html = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<style>
  @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
