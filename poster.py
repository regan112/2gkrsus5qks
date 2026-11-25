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
  
  * {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
    font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
  }}

  body {{
    background-color: #e9ecef;
    display: flex;
    justify-content: center;
    padding: 20px;
  }}

  /* A1 비율 컨테이너 (594mm x 841mm 비율 유지) */
  .poster-a1 {{
    width: 1188px;
    min-height: 1682px;
    background: #ffffff;
    padding: 44px 50px;
    box-shadow: 0 12px 36px rgba(0,0,0,0.15);
    color: #1a202c;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }}

  /* 헤더 */
  .header {{
    border-bottom: 4px solid #1e3a8a;
    padding-bottom: 16px;
    margin-bottom: 20px;
  }}
  .header h1 {{
    font-size: 42px;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.02em;
    margin-bottom: 8px;
  }}
  .header .team {{
    font-size: 19px;
    font-weight: 700;
    color: #1e40af;
    margin-bottom: 4px;
  }}
  .header .sub {{
    font-size: 15.5px;
    font-weight: 500;
    color: #475569;
  }}

  /* 요약 박스 */
  .summary-box {{
    background-color: #eff6ff;
    border-left: 7px solid #1d4ed8;
    padding: 18px 24px;
    margin-bottom: 24px;
    border-radius: 0 8px 8px 0;
  }}
  .summary-box h2 {{
    font-size: 20px;
    font-weight: 800;
    color: #1e3a8a;
    margin-bottom: 8px;
  }}
  .summary-box p {{
    font-size: 15.5px;
    line-height: 1.52;
    margin-bottom: 5px;
  }}
  .summary-box strong {{
    color: #1e3a8a;
  }}

  /* 2단 메인 그리드 */
  .main-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 34px;
    flex-grow: 1;
  }}

  .section {{
    margin-bottom: 22px;
  }}
  .section-title {{
    font-size: 22px;
    font-weight: 800;
    color: #0f172a;
    border-bottom: 2.5px solid #1e3a8a;
    padding-bottom: 5px;
    margin-bottom: 12px;
  }}
  .body-text {{
    font-size: 15.2px;
    line-height: 1.58;
    color: #2d3748;
    text-align: justify;
    margin-bottom: 8px;
  }}

  /* 그래프 카드 공통 */
  .chart-card {{
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 14px;
    margin: 10px 0;
  }}
  .chart-caption {{
    font-size: 13.5px;
    color: #475569;
    margin-top: 8px;
    font-weight: 600;
  }}

  /* 그래프 1: 계약형태별 이탈률 */
  .bar-group-contract {{
    display: flex;
    align-items: flex-end;
    justify-content: space-around;
    height: 135px;
    padding-top: 20px;
    border-bottom: 2px solid #cbd5e1;
  }}
  .c-bar-wrap {{
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 24%;
  }}
  .c-bar {{
    width: 100%;
    background: #f97316;
    border-radius: 4px 4px 0 0;
    display: flex;
    justify-content: center;
    align-items: flex-start;
    padding-top: 5px;
    color: white;
    font-weight: 700;
    font-size: 13.5px;
  }}
  .c-bar.highlight {{
    background: #ef4444;
  }}
  .c-label {{
    margin-top: 6px;
    font-size: 13.5px;
    font-weight: 700;
    color: #334155;
  }}

  /* 파이프라인 플로우차트 */
  .pipeline {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 12px 0;
    gap: 4px;
  }}
  .pipe-step {{
    border: 1.5px solid #93c5fd;
    background: #eff6ff;
    border-radius: 6px;
    padding: 7px 4px;
    text-align: center;
    font-size: 12.2px;
    font-weight: 700;
    color: #1e3a8a;
    flex: 1;
    line-height: 1.3;
  }}
  .pipe-step.active {{
    background: #1e3a8a;
    color: #ffffff;
    border-color: #1e3a8a;
  }}
  .pipe-arrow {{
    color: #64748b;
    font-weight: bold;
    font-size: 13px;
  }}

  /* 모델 비교 표 */
  table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 8px;
    font-size: 14px;
  }}
  th {{
    background: #f1f5f9;
    color: #0f172a;
    font-weight: 700;
    padding: 8px 6px;
    border-top: 2px solid #334155;
    border-bottom: 1.5px solid #94a3b8;
    text-align: center;
  }}
  th:first-child, td:first-child {{
    text-align: left;
    padding-left: 10px;
  }}
  td {{
    padding: 7px 6px;
    border-bottom: 1px solid #e2e8f0;
    text-align: center;
  }}
  tr.selected-row {{
    background-color: #dbeafe;
    font-weight: 700;
    color: #1e3a8a;
  }}

  /* 그림 2: 수평 막대 비교 차트 */
  .metric-block {{
    margin-bottom: 10px;
  }}
  .metric-header {{
    font-size: 13.5px;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 4px;
  }}
  .h-bar-row {{
    display: flex;
    align-items: center;
    margin-bottom: 4px;
    font-size: 12.5px;
  }}
  .h-bar-label {{
    width: 95px;
    font-weight: 600;
    color: #334155;
  }}
  .h-bar-track {{
    flex-grow: 1;
    background: #e2e8f0;
    height: 14px;
    border-radius: 3px;
    overflow: hidden;
    margin-right: 8px;
  }}
  .h-bar-fill {{
    height: 100%;
    background: #64748b;
  }}
  .h-bar-fill.primary {{
    background: #1d4ed8;
  }}
  .h-bar-val {{
    width: 42px;
    font-weight: 700;
    color: #0f172a;
    text-align: right;
  }}

  /* 그림 3: 혼동 행렬 2x2 그리드 */
  .cm-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-top: 6px;
  }}
  .cm-box {{
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 8px;
  }}
  .cm-box.selected {{
    border: 2px solid #1d4ed8;
    background: #eff6ff;
  }}
  .cm-title {{
    font-size: 13.5px;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 5px;
    text-align: center;
  }}
  .cm-cells {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 4px;
    text-align: center;
    font-size: 12.5px;
    font-weight: 700;
  }}
  .cell {{
    padding: 8px 4px;
    border-radius: 4px;
  }}
  .c-dark {{ background: #0f2952; color: #fff; }}
  .c-med {{ background: #7da6c9; color: #0f172a; }}
  .c-light {{ background: #cbdbe9; color: #0f172a; }}
  .c-pale {{ background: #f1f5f9; color: #475569; }}

  /* 섹션 5 & 6 본문 스타일 */
  .reason-item {{
    font-size: 14.8px;
    line-height: 1.55;
    color: #2d3748;
    margin-bottom: 8px;
    text-align: justify;
  }}
  .reason-item strong {{
    color: #1e3a8a;
  }}

  /* 하단 참고문헌 */
  .footer {{
    border-top: 2px solid #94a3b8;
    padding-top: 10px;
    margin-top: 12px;
    font-size: 12.5px;
    color: #475569;
    line-height: 1.45;
  }}

  /* 인쇄(PDF 저장) 시 A1 용지 최적화 */
  @media print {{
    body {{
      background: none;
      padding: 0;
    }}
    .poster-a1 {{
      width: 594mm;
      height: 841mm;
      box-shadow: none;
      padding: 22mm 24mm;
    }}
  }}
</style>
</head>
<body>
  <div class="poster-a1">
    <div>
      <!-- 헤더 -->
      <div class="header">
        <h1>{title}</h1>
        <div class="team">{team_info}</div>
        <div class="sub">{sub_info}</div>
      </div>

      <!-- 요약 박스 -->
      <div class="summary-box">
        <h2>요약 (Executive Summary)</h2>
        <p><strong>연구 질문:</strong> {sum_q}</p>
        <p><strong>데이터와 방법:</strong> {sum_m}</p>
        <p><strong>핵심 결과:</strong> {sum_r}</p>
        <p><strong>최종 결론:</strong> {sum_c}</p>
      </div>

      <!-- 본문 2단 그리드 -->
      <div class="main-grid">
        <!-- 왼쪽 컬럼 -->
        <div>
          <div class="section">
            <div class="section-title">1. 서론</div>
            <p class="body-text">{sec1_p1}</p>
            <p class="body-text">{sec1_p2}</p>
          </div>

          <div class="section">
            <div class="section-title">2. 데이터 및 탐색적 분석 (EDA)</div>
            <p class="body-text">{sec2_p1}</p>
            <div class="chart-card">
              <div class="bar-group-contract">
                <div class="c-bar-wrap">
                  <div class="c-bar" style="height: 42px;">16.75%</div>
                  <div class="c-label">1년</div>
                </div>
                <div class="c-bar-wrap">
                  <div class="c-bar" style="height: 43px;">16.88%</div>
                  <div class="c-label">2년</div>
                </div>
                <div class="c-bar-wrap">
                  <div class="c-bar highlight" style="height: 112px;">46.56%</div>
                  <div class="c-label">월 단위</div>
                </div>
              </div>
              <div class="chart-caption">그림 1. 계약 형태별 고객 이탈률 비교 (고해상도 벡터 차트)<br>월 단위 계약의 이탈률(46.56%)이 1년(16.75%), 2년(16.88%) 약정 계약에 비해 현저하게 높음.</div>
            </div>
            <p class="body-text">{sec2_p2}</p>
          </div>

          <div class="section">
            <div class="section-title">3. 분석 방법</div>
            <p class="body-text">{sec3_p1}</p>
            <div class="pipeline">
              <div class="pipe-step">데이터 수집<br>Kaggle 100k</div>
              <div class="pipe-arrow">→</div>
              <div class="pipe-step">전처리<br>인코딩·분할</div>
              <div class="pipe-arrow">→</div>
              <div class="pipe-step">후보 모델<br>3종 알고리즘</div>
              <div class="pipe-arrow">→</div>
              <div class="pipe-step active">선택: 로지스틱 회귀<br>재현율 0.702</div>
              <div class="pipe-arrow">→</div>
              <div class="pipe-step">비즈니스 평가<br>혼동행렬 분석</div>
            </div>
            <p class="body-text">{sec3_p2}</p>
          </div>
        </div>

        <!-- 오른쪽 컬럼 -->
        <div>
          <div class="section">
            <div class="section-title">4. 모델 평가 결과</div>
            <table>
              <thead>
                <tr>
                  <th>모델명</th>
                  <th>재현율(Recall)</th>
                  <th>F1-Score</th>
                  <th>정밀도(Precision)</th>
                  <th>정확도(Accuracy)</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>베이스라인(최빈값)</td>
                  <td>0.000</td>
                  <td>0.000</td>
                  <td>0.000</td>
                  <td>0.669</td>
                </tr>
                <tr class="selected-row">
                  <td>로지스틱 회귀 (최종 선택)</td>
                  <td>0.702</td>
                  <td>0.599</td>
                  <td>0.522</td>
                  <td>0.688</td>
                </tr>
                <tr>
                  <td>의사결정 트리</td>
                  <td>0.618</td>
                  <td>0.625</td>
                  <td>0.631</td>
                  <td>0.754</td>
                </tr>
                <tr>
                  <td>랜덤 포레스트</td>
                  <td>0.629</td>
                  <td>0.625</td>
                  <td>0.621</td>
                  <td>0.750</td>
                </tr>
              </tbody>
            </table>

            <div class="chart-card">
              <div class="metric-block">
                <div class="metric-header">재현율 (Recall)</div>
                <div class="h-bar-row"><div class="h-bar-label">로지스틱 회귀</div><div class="h-bar-track"><div class="h-bar-fill primary" style="width:70.2%"></div></div><div class="h-bar-val">0.702</div></div>
                <div class="h-bar-row"><div class="h-bar-label">의사결정트리</div><div class="h-bar-track"><div class="h-bar-fill" style="width:61.8%"></div></div><div class="h-bar-val">0.618</div></div>
                <div class="h-bar-row"><div class="h-bar-label">랜덤포레스트</div><div class="h-bar-track"><div class="h-bar-fill" style="width:62.9%"></div></div><div class="h-bar-val">0.629</div></div>
              </div>
              <div class="metric-block" style="margin-bottom:0;">
                <div class="metric-header">정확도 (Accuracy)</div>
                <div class="h-bar-row"><div class="h-bar-label">베이스라인</div><div class="h-bar-track"><div class="h-bar-fill" style="width:66.9%"></div></div><div class="h-bar-val">0.669</div></div>
                <div class="h-bar-row"><div class="h-bar-label">로지스틱 회귀</div><div class="h-bar-track"><div class="h-bar-fill primary" style="width:68.8%"></div></div><div class="h-bar-val">0.688</div></div>
                <div class="h-bar-row"><div class="h-bar-label">의사결정트리</div><div class="h-bar-track"><div class="h-bar-fill" style="width:75.4%"></div></div><div class="h-bar-val">0.754</div></div>
                <div class="h-bar-row"><div class="h-bar-label">랜덤포레스트</div><div class="h-bar-track"><div class="h-bar-fill" style="width:75.0%"></div></div><div class="h-bar-val">0.750</div></div>
              </div>
              <div class="chart-caption">그림 2. 주요 성능 지표 비교 (선명한 벡터 그래픽) - 로지스틱 회귀가 이탈자 감지의 핵심인 재현율(0.702)에서 최우수 성과 기록.</div>
            </div>

            <div class="cm-grid">
              <div class="cm-box">
                <div class="cm-title">베이스라인</div>
                <div class="cm-cells">
                  <div class="cell c-dark">13.37k</div>
                  <div class="cell c-pale">0</div>
                  <div class="cell c-med">6,629</div>
                  <div class="cell c-pale">0</div>
                </div>
              </div>
              <div class="cm-box selected">
                <div class="cm-title">로지스틱 회귀</div>
                <div class="cm-cells">
                  <div class="cell c-dark">9,115</div>
                  <div class="cell c-light">4,256</div>
                  <div class="cell c-pale">1,977</div>
                  <div class="cell c-med">4,652</div>
                </div>
              </div>
              <div class="cm-box">
                <div class="cm-title">의사결정트리</div>
                <div class="cm-cells">
                  <div class="cell c-dark">10.98k</div>
                  <div class="cell c-pale">2,393</div>
                  <div class="cell c-pale">2,530</div>
                  <div class="cell c-light">4,099</div>
                </div>
              </div>
              <div class="cm-box">
                <div class="cm-title">랜덤 포레스트</div>
                <div class="cm-cells">
                  <div class="cell c-dark">10.83k</div>
                  <div class="cell c-pale">2,544</div>
                  <div class="cell c-pale">2,461</div>
                  <div class="cell c-light">4,168</div>
                </div>
              </div>
            </div>
            <div class="chart-caption">그림 3. 모델별 혼동 행렬 (Confusion Matrix) 비교 (선명한 벡터 그리드)<br>로지스틱 회귀가 실제 이탈 고객 6,629명 중 가장 많은 4,652명(TP)을 올바르게 포착함.</div>
          </div>

          <div class="section">
            <div class="section-title">5. 모델 선택 이유 및 논의</div>
            <div class="reason-item"><strong>① 재현율 최우선 고려:</strong> {sec5_1}</div>
            <div class="reason-item"><strong>② 정밀도 및 정확도:</strong> {sec5_2}</div>
            <div class="reason-item"><strong>③ 선제적 방어 마케팅 가능:</strong> {sec5_3}</div>
          </div>

          <div class="section">
            <div class="section-title">6. 한계 및 분석 결과</div>
            <div class="reason-item"><strong>프로젝트의 한계 및 향후 과제:</strong> {sec6_limit}</div>
            <div class="reason-item"><strong>분석 결과:</strong> {sec6_result}</div>
          </div>
        </div>
      </div>
    </div>

    <!-- 하단 참고문헌 -->
    <div class="footer">
      <strong>참고문헌 및 출처:</strong> Kaggle Telco Customer Churn Dataset (https://www.kaggle.com/datasets/dhrubangtalukdar/telco-customer-churn-data) | 통신사 고객 이탈 데이터 분석 학술 포스터
    </div>
  </div>
</body>
</html>
"""

components.html(poster_html, height=1750, scrolling=True)
