import streamlit as st
import streamlit.components.v1 as components

# 페이지 기본 설정 (넓은 화면)
st.set_page_config(
    page_title="통신사 고객 이탈 예측 A1 학술 포스터",
    page_icon="📊",
    layout="wide"
)

# 사이드바: 실시간 텍스트 수정 기능
st.sidebar.header("📝 포스터 텍스트 실시간 수정")
st.sidebar.info("아래 내용을 수정하면 오른쪽 A1 포스터에 즉시 반영됩니다. 인쇄(Ctrl+P) 시 A1 규격에 맞춰 깔끔하게 저장됩니다.")

with st.sidebar.expander("1. 기본 정보 (제목/팀원)", expanded=True):
    title = st.text_input("포스터 제목", "어떤 데이터가 통신사 고객 이탈로 이어지는가")
    team_info = st.text_input("팀원 및 소속", "20510 박주형 · 20518 이우빈 · 20523 정은솔 | 데이터 과학 팀 프로젝트 · 2026")

with st.sidebar.expander("2. 상단 요약 박스", expanded=False):
    sum_q = st.text_input("연구 질문", "어떤 데이터가 통신사 고객 이탈로 이어지는가")
    sum_m = st.text_input("데이터와 방법", "통신사 고객 데이터, 로지스틱 회귀·의사결정트리·랜덤 포레스트 분류 모델 비교")
    sum_r = st.text_input("핵심 결과", "로지스틱 회귀 최종 채택(재현율 0.702), 월 단위 약정 고객 이탈률 46.56%")
    sum_c = st.text_input("결론", "약정 계약 형태가 고객 이탈의 핵심 동인이며, 이탈 방어 전략 수립에 재현율 중심 모델이 가장 적합함")

with st.sidebar.expander("3. 본문 섹션 수정", expanded=False):
    sec1_intro = st.text_area(
        "1. 서론 (데이터 선정 이유)",
        "통신 산업은 방대한 고객 가입 정보와 이용 패턴 데이터가 축적되어 있어 데이터 기반 의사결정에 최적화된 분야이다. 특히 신규 고객 유치 비용(CAC)이 기존 고객 유지 비용보다 훨씬 크기 때문에, 이탈 징후를 사전에 포착하는 것이 기업의 이윤 극대화와 직결된다. 본 연구는 통신사 고객 데이터를 활용하여 고객 이탈을 유발하는 핵심 변수를 규명하고, 3종의 분류 모델과 기준선을 비교해 최적의 이탈 예측 모델을 도출하고자 한다.",
        height=130
    )
    sec5_reason = st.text_area(
        "5. 모델 선택 이유 및 논의 (경영학적 관점)",
        "• 로지스틱 회귀 최종 선택 (재현율 0.702): 전체 정확도는 의사결정트리(0.754)가 높았으나, 실제 이탈 고객 6,629명 중 4,652명을 정확히 포착해 재현율이 가장 높은 로지스틱 회귀를 최종 선택하였다.\n• 기회비용 최소화 전략: 통신업 특성상 실제 이탈 고객을 놓쳐 발생하는 매출 손실(FN)이 유지 고객에게 방어 쿠폰을 지급하는 마케팅 비용(FP)보다 훨씬 크므로 재현율 극대화가 기업 이윤 방어에 유리하다.\n• 약정 전환 CRM 도출: 월 단위 계약 고객의 이탈률(46.56%)이 장기 약정 대비 약 2.8배 높으므로, 가입 초기 장기 약정 전환 프로모션을 집중해야 한다.",
        height=150
    )
    sec6_limit = st.text_area(
        "6. 한계 및 배운 점",
        "• 비용-편익(Cost-Benefit) 가중치 미반영: 모든 이탈 고객을 동일하게 취급했으나, 향후 고객별 월평균 매출(ARPU)을 결합해 '기대 손실액' 기준의 VIP 타겟팅 모델로 고도화할 필요가 있다.\n• 데이터 불균형 및 외부 변수: 유지 고객(13,371명)이 이탈 고객(6,629명)보다 약 2배 많아 향후 SMOTE 오버샘플링 적용 및 경쟁사 보조금 등 외부 시장 변수 반영이 필요하다.\n• 배운 점: 머신러닝 평가지표(정확도 vs 재현율)의 트레이드오프가 실제 기업의 수익 구조 및 마케팅 ROI 최적화와 직결됨을 깨달았다.",
        height=150
    )

# 줄바꿈 및 불릿 포맷 변환 도우미 함수
def format_bullets(text):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    html_items = ""
    for line in lines:
        clean_line = line.lstrip("•-* ").strip()
        if ":" in clean_line:
            head, body = clean_line.split(":", 1)
            html_items += f"<li><strong>{head}:</strong>{body}</li>"
        else:
            html_items += f"<li>{clean_line}</li>"
    return f"<ul class='bullet-list'>{html_items}</ul>"

sec5_html = format_bullets(sec5_reason)
sec6_html = format_bullets(sec6_limit)

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
    padding: 48px 54px;
    box-shadow: 0 12px 36px rgba(0,0,0,0.15);
    color: #1a202c;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }}

  /* 헤더 */
  .header {{
    border-bottom: 4px solid #1e3a8a;
    padding-bottom: 18px;
    margin-bottom: 22px;
  }}
  .header h1 {{
    font-size: 44px;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.02em;
    margin-bottom: 10px;
  }}
  .header .team {{
    font-size: 19px;
    font-weight: 600;
    color: #1e40af;
  }}

  /* 요약 박스 */
  .summary-box {{
    background-color: #eff6ff;
    border-left: 7px solid #1d4ed8;
    padding: 20px 26px;
    margin-bottom: 28px;
    border-radius: 0 8px 8px 0;
  }}
  .summary-box h2 {{
    font-size: 21px;
    font-weight: 800;
    color: #1e3a8a;
    margin-bottom: 10px;
  }}
  .summary-box p {{
    font-size: 16.5px;
    line-height: 1.55;
    margin-bottom: 5px;
  }}
  .summary-box strong {{
    color: #1e3a8a;
  }}

  /* 2단 메인 그리드 */
  .main-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 38px;
    flex-grow: 1;
  }}

  .section {{
    margin-bottom: 26px;
  }}
  .section-title {{
    font-size: 24px;
    font-weight: 800;
    color: #0f172a;
    border-bottom: 2.5px solid #1e3a8a;
    padding-bottom: 6px;
    margin-bottom: 14px;
  }}
  .body-text {{
    font-size: 16px;
    line-height: 1.62;
    color: #2d3748;
    text-align: justify;
  }}

  .bullet-list {{
    padding-left: 20px;
    font-size: 15.8px;
    line-height: 1.6;
    color: #2d3748;
  }}
  .bullet-list li {{
    margin-bottom: 8px;
  }}
  .bullet-list strong {{
    color: #1e3a8a;
  }}

  /* 그래프 1: 계약형태별 이탈률 */
  .chart-card {{
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 16px;
    margin-top: 12px;
  }}
  .chart-caption {{
    font-size: 14px;
    color: #475569;
    margin-top: 8px;
    font-weight: 500;
  }}
  .bar-group-contract {{
    display: flex;
    align-items: flex-end;
    justify-content: space-around;
    height: 150px;
    padding-top: 25px;
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
    padding-top: 6px;
    color: white;
    font-weight: 700;
    font-size: 14px;
  }}
  .c-bar.highlight {{
    background: #ef4444;
  }}
  .c-label {{
    margin-top: 8px;
    font-size: 14px;
    font-weight: 700;
    color: #334155;
  }}

  /* 파이프라인 플로우차트 */
  .pipeline {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 14px 0;
    gap: 4px;
  }}
  .pipe-step {{
    border: 1.5px solid #93c5fd;
    background: #eff6ff;
    border-radius: 6px;
    padding: 8px 6px;
    text-align: center;
    font-size: 13px;
    font-weight: 700;
    color: #1e3a8a;
    flex: 1;
  }}
  .pipe-step.active {{
    background: #1e3a8a;
    color: #ffffff;
    border-color: #1e3a8a;
  }}
  .pipe-arrow {{
    color: #64748b;
    font-weight: bold;
    font-size: 14px;
  }}

  /* 모델 비교 표 */
  table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 10px;
    font-size: 14.5px;
  }}
  th {{
    background: #f1f5f9;
    color: #0f172a;
    font-weight: 700;
    padding: 9px 8px;
    border-top: 2px solid #334155;
    border-bottom: 1.5px solid #94a3b8;
    text-align: center;
  }}
  th:first-child, td:first-child {{
    text-align: left;
    padding-left: 12px;
  }}
  td {{
    padding: 8px;
    border-bottom: 1px solid #e2e8f0;
    text-align: center;
  }}
  tr.selected-row {{
    background-color: #dbeafe;
    font-weight: 700;
    color: #1e3a8a;
  }}

  /* 혼동 행렬 2x2 그리드 */
  .cm-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-top: 8px;
  }}
  .cm-box {{
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 10px;
  }}
  .cm-box.selected {{
    border: 2px solid #1d4ed8;
    background: #eff6ff;
  }}
  .cm-title {{
    font-size: 14px;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 6px;
    text-align: center;
  }}
  .cm-cells {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 4px;
    text-align: center;
    font-size: 13px;
    font-weight: 700;
  }}
  .cell {{
    padding: 10px 4px;
    border-radius: 4px;
  }}
  .c-dark {{ background: #0f2952; color: #fff; }}
  .c-med {{ background: #7da6c9; color: #0f172a; }}
  .c-light {{ background: #cbdbe9; color: #0f172a; }}
  .c-pale {{ background: #f1f5f9; color: #475569; }}

  /* 핵심 수치 하이라이트 */
  .kpi-row {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-top: 14px;
    padding-top: 12px;
    border-top: 1.5px solid #cbd5e1;
  }}
  .kpi-card {{
    text-align: center;
  }}
  .kpi-num {{
    font-size: 30px;
    font-weight: 800;
    color: #1e3a8a;
  }}
  .kpi-desc {{
    font-size: 13.5px;
    color: #475569;
    font-weight: 600;
  }}

  /* 하단 참고문헌 */
  .footer {{
    border-top: 2px solid #94a3b8;
    padding-top: 12px;
    margin-top: 16px;
    font-size: 13px;
    color: #475569;
    line-height: 1.5;
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
      padding: 24mm 26mm;
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
      </div>

      <!-- 요약 박스 -->
      <div class="summary-box">
        <h2>요약 (Executive Summary)</h2>
        <p><strong>연구 질문:</strong> {sum_q}</p>
        <p><strong>데이터와 방법:</strong> {sum_m}</p>
        <p><strong>핵심 결과:</strong> {sum_r}</p>
        <p><strong>결론:</strong> {sum_c}</p>
      </div>

      <!-- 본문 2단 그리드 -->
      <div class="main-grid">
        <!-- 왼쪽 컬럼 -->
        <div>
          <div class="section">
            <div class="section-title">1. 서론 (연구 배경 및 목적)</div>
            <p class="body-text">{sec1_intro}</p>
          </div>

          <div class="section">
            <div class="section-title">2. 데이터 및 탐색적 분석 (EDA)</div>
            <ul class="bullet-list">
              <li><strong>데이터 출처:</strong> Kaggle Telco Customer Churn 데이터셋 (시험 평가 표본 20,000건 기준)</li>
              <li><strong>타겟 분포:</strong> 전체 고객 중 유지 13,371명(66.9%), 이탈 6,629명(33.1%)</li>
              <li><strong>핵심 변수 발견:</strong> 계약 형태(Contract)에 따라 고객 이탈률에서 극명한 격차 확인</li>
            </ul>
            <div class="chart-card">
              <div style="font-weight:700; font-size:15px; color:#0f172a;">계약 형태별 고객 이탈률 (%)</div>
              <div class="bar-group-contract">
                <div class="c-bar-wrap">
                  <div class="c-bar" style="height: 45px;">16.75%</div>
                  <div class="c-label">1년 약정</div>
                </div>
                <div class="c-bar-wrap">
                  <div class="c-bar" style="height: 46px;">16.88%</div>
                  <div class="c-label">2년 약정</div>
                </div>
                <div class="c-bar-wrap">
                  <div class="c-bar highlight" style="height: 122px;">46.56%</div>
                  <div class="c-label">월 단위 계약</div>
                </div>
              </div>
              <div class="chart-caption">그림 1. 월 단위 계약 고객의 이탈률(46.56%)이 1·2년 약정 고객 대비 약 2.8배 높게 나타남</div>
            </div>
          </div>

          <div class="section">
            <div class="section-title">3. 분석 방법 및 모델 비교</div>
            <div class="pipeline">
              <div class="pipe-step">데이터 수집<br>Telco Churn</div>
              <div class="pipe-arrow">→</div>
              <div class="pipe-step">전처리·분할<br>범주형 인코딩</div>
              <div class="pipe-arrow">→</div>
              <div class="pipe-step">후보 모델 비교<br>기준선·3종 모델</div>
              <div class="pipe-arrow">→</div>
              <div class="pipe-step active">최종 선택<br>로지스틱 회귀</div>
              <div class="pipe-arrow">→</div>
              <div class="pipe-step">경영 전략 도출<br>혼동 행렬 해석</div>
            </div>
            <table>
              <thead>
                <tr>
                  <th>분류 모델</th>
                  <th>재현율(Recall)</th>
                  <th>정밀도(Precision)</th>
                  <th>F1 점수</th>
                  <th>정확도(Accuracy)</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>베이스라인 (최빈값)</td>
                  <td>-</td>
                  <td>-</td>
                  <td>-</td>
                  <td>0.669</td>
                </tr>
                <tr class="selected-row">
                  <td>로지스틱 회귀 (최종 선택)</td>
                  <td>0.702</td>
                  <td>0.522</td>
                  <td>0.599</td>
                  <td>0.688</td>
                </tr>
                <tr>
                  <td>의사결정트리</td>
                  <td>0.618</td>
                  <td>0.631</td>
                  <td>0.625</td>
                  <td>0.754</td>
                </tr>
                <tr>
                  <td>랜덤 포레스트</td>
                  <td>0.629</td>
                  <td>0.621</td>
                  <td>0.625</td>
                  <td>0.750</td>
                </tr>
              </tbody>
            </table>
            <div class="chart-caption">표 1. 모델별 분류 성능 지표 비교 (이탈 고객 탐지력 기준 재현율 최우선 평가)</div>
          </div>
        </div>

        <!-- 오른쪽 컬럼 -->
        <div>
          <div class="section">
            <div class="section-title">4. 결과 (혼동 행렬 및 핵심 지표)</div>
            <div class="cm-grid">
              <div class="cm-box">
                <div class="cm-title">베이스라인 (최빈값)</div>
                <div class="cm-cells">
                  <div class="cell c-dark">실제유지·예측유지<br>13,371</div>
                  <div class="cell c-pale">실제유지·예측이탈<br>0</div>
                  <div class="cell c-med">실제이탈·예측유지<br>6,629</div>
                  <div class="cell c-pale">실제이탈·예측이탈<br>0</div>
                </div>
              </div>
              <div class="cm-box selected">
                <div class="cm-title">★ 로지스틱 회귀 (최종 선택)</div>
                <div class="cm-cells">
                  <div class="cell c-dark">실제유지·예측유지<br>9,115</div>
                  <div class="cell c-light">실제유지·예측이탈<br>4,256</div>
                  <div class="cell c-pale">실제이탈·예측유지<br>1,977</div>
                  <div class="cell c-med">실제이탈·예측이탈<br>4,652</div>
                </div>
              </div>
              <div class="cm-box">
                <div class="cm-title">의사결정트리</div>
                <div class="cm-cells">
                  <div class="cell c-dark">실제유지·예측유지<br>10,978</div>
                  <div class="cell c-pale">실제유지·예측이탈<br>2,393</div>
                  <div class="cell c-pale">실제이탈·예측유지<br>2,530</div>
                  <div class="cell c-light">실제이탈·예측이탈<br>4,099</div>
                </div>
              </div>
              <div class="cm-box">
                <div class="cm-title">랜덤 포레스트</div>
                <div class="cm-cells">
                  <div class="cell c-dark">실제유지·예측유지<br>10,827</div>
                  <div class="cell c-pale">실제유지·예측이탈<br>2,544</div>
                  <div class="cell c-pale">실제이탈·예측유지<br>2,461</div>
                  <div class="cell c-light">실제이탈·예측이탈<br>4,168</div>
                </div>
              </div>
            </div>
            <div class="chart-caption">그림 2. 모델별 혼동 행렬 비교 (로지스틱 회귀가 실제 이탈자 4,652명을 포착해 최다 탐지)</div>

            <div class="kpi-row">
              <div class="kpi-card">
                <div class="kpi-num">0.702</div>
                <div class="kpi-desc">로지스틱 회귀 재현율(1위)</div>
              </div>
              <div class="kpi-card">
                <div class="kpi-num">4,652명</div>
                <div class="kpi-desc">실제 이탈 고객 사전 탐지</div>
              </div>
              <div class="kpi-card">
                <div class="kpi-num">46.56%</div>
                <div class="kpi-desc">월 단위 계약 고객 이탈률</div>
              </div>
            </div>
          </div>

          <div class="section">
            <div class="section-title">5. 모델 선택 이유 및 논의 (경영학적 관점)</div>
            {sec5_html}
          </div>

          <div class="section">
            <div class="section-title">6. 한계 및 배운 점</div>
            {sec6_html}
          </div>
        </div>
      </div>
    </div>

    <!-- 하단 참고문헌 -->
    <div class="footer">
      <strong>참고문헌 및 데이터 출처:</strong> [1] Kaggle Telco Customer Churn Data (https://www.kaggle.com/datasets/dhrubangtalukdar/telco-customer-churn-data)<br>
      <strong>분석 환경 및 대시보드:</strong> Python, scikit-learn, Streamlit | Interactive App: https://2gkrsus5qks-ygiaeql3mnvc2fyjxebh8m.streamlit.app/
    </div>
  </div>
</body>
</html>
"""

# Streamlit 화면에 포스터 렌더링
components.html(poster_html, height=1750, scrolling=True)
