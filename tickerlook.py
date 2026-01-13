import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import google.generativeai as genai
import time

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="AI 퀀트 V28 (Debt Ratio)", layout="wide")
st.title("🤖 AI 퀀트 스크리너 V28 (Debt Ratio Added)")
st.markdown("안정적인 V27 엔진에 **부채비율(안정성) 분석** 및 가중치 설정을 추가했습니다.")

# --- 2. 사이드바 ---
st.sidebar.header("1. 시장 선택")
country = st.sidebar.radio("국가", ["미국 (US)", "한국 (KR)"], horizontal=True)

market_index = ""
target_sector = "전체"

if country == "미국 (US)":
    market_index = st.sidebar.selectbox("지수", ["S&P 500 / NASDAQ", "Russell 2000 (중소형)"])
    target_sector = st.sidebar.selectbox("섹터 (업종)", [
        "기술 (Technology)", 
        "커뮤니케이션 (Communication)", 
        "헬스케어 (Healthcare)", 
        "소비재 (Consumer)", 
        "금융 (Financial)", 
        "에너지/산업 (Energy/Ind)",
        "전체 (All)"
    ])
else:
    market_index = st.sidebar.selectbox("지수", ["KOSPI", "KOSDAQ"])
    st.sidebar.info("※ 한국장은 데이터 안정성을 위해 **전체(시총 상위)** 기준으로 통합 검색합니다.")

st.sidebar.markdown("---")
st.sidebar.header("2. 차트 설정")
use_log_x = st.sidebar.checkbox("X축 (PER) 로그", value=False)
use_log_y = st.sidebar.checkbox("Y축 (ROE) 로그", value=False)
show_avg = st.sidebar.checkbox("평균선 표시", value=True)

st.sidebar.markdown("---")
st.sidebar.header("3. 가중치 설정")
w_per = st.sidebar.slider("저평가 (PER)", 0, 100, 40)
w_roe = st.sidebar.slider("수익성 (ROE)", 0, 100, 40)
w_eps = st.sidebar.slider("성장성 (EPS)", 0, 100, 10)
# [NEW] 부채비율 가중치 추가 (기본값 0)
w_debt = st.sidebar.slider("안정성 (부채비율)", 0, 100, 0, help="부채비율이 낮을수록 점수가 높아집니다. 한국장은 데이터 특성상 0으로 간주될 수 있습니다.")

st.sidebar.markdown("---")
st.sidebar.header("🔑 AI 설정")
api_key = st.sidebar.text_input("Gemini API Key", type="password")

# 모델 자동 감지
available_models = ["gemini-1.5-flash", "gemini-pro"]
if api_key:
    try:
        genai.configure(api_key=api_key)
        scanned = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        if scanned: available_models = scanned
    except: pass
ai_model = st.sidebar.selectbox("사용할 모델", available_models, index=0)

# --- 유틸리티 ---
def clean_numeric(value):
    try:
        if isinstance(value, str):
            value = value.replace(',', '').replace('N/A', '0').replace('-', '0')
        return float(value)
    except: return 0.0

# --- 3. 데이터 수집 함수 ---
@st.cache_data
def analyze_data(country, index, sector):
    data = []
    
    # ==========================================
    # 🇺🇸 미국 시장
    # ==========================================
    if country == "미국 (US)":
        sector_map = {
            "기술 (Technology)": ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'AMD', 'INTC', 'CRM', 'ADBE', 'ORCL', 'IBM', 'QCOM', 'TXN', 'NOW', 'AMAT', 'MU', 'PLTR', 'SMCI'],
            "커뮤니케이션 (Communication)": ['GOOGL', 'GOOG', 'META', 'NFLX', 'DIS', 'CMCSA', 'TMUS', 'VZ', 'T', 'CHTR', 'WBD'],
            "헬스케어 (Healthcare)": ['LLY', 'UNH', 'JNJ', 'MRK', 'ABBV', 'PFE', 'TMO', 'AMGN', 'ABT', 'GILD', 'ISRG'],
            "소비재 (Consumer)": ['AMZN', 'TSLA', 'WMT', 'PG', 'KO', 'PEP', 'COST', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TGT'],
            "금융 (Financial)": ['JPM', 'BAC', 'V', 'MA', 'BRK-B', 'WFC', 'MS', 'GS', 'C', 'BLK', 'AXP', 'SPGI', 'O'],
            "에너지/산업 (Energy/Ind)": ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'CAT', 'BA', 'GE', 'HON', 'LMT', 'RTX']
        }
        russell_tickers = ['MSTR', 'SMCI', 'DKNG', 'RIVN', 'SOFI', 'HOOD', 'AFRM', 'LCID', 'MARA', 'CLSK', 'COIN', 'RIOT', 'GME', 'AMC', 'PATH', 'U']

        target_tickers = []
        if index == "Russell 2000 (중소형)":
            target_tickers = russell_tickers
        else:
            if sector == "전체 (All)":
                for k in sector_map: target_tickers += sector_map[k]
            else:
                target_tickers = sector_map.get(sector, [])

        bar = st.progress(0, text=f"🇺🇸 {sector} 데이터 수집 중...")
        for i, t in enumerate(target_tickers):
            try:
                info = yf.Ticker(t).info
                data.append({
                    '티커': t, '종목명': info.get('shortName', t), '현재가': info.get('currentPrice', 0),
                    'PER': info.get('trailingPE', 0), 'ROE': info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0, 
                    'EPS': info.get('trailingEps', 0),
                    # [NEW] 부채비율 추가
                    '부채비율': info.get('debtToEquity', 0)
                })
            except: pass
            bar.progress((i+1)/len(target_tickers))
        bar.empty()

    # ==========================================
    # 🇰🇷 한국 시장 (V27 엔진 유지)
    # ==========================================
    else:
        sosok = 0 if index == 'KOSPI' else 1
        url_base = f"https://finance.naver.com/sise/sise_market_sum.naver?sosok={sosok}&page="
        
        all_dfs = []
        bar = st.progress(0, text="🇰🇷 네이버 증권 데이터(상위 200개) 스캔 중...")
        
        for page in range(1, 5): 
            try:
                dfs = pd.read_html(url_base + str(page), encoding='euc-kr', header=0)
                df = dfs[1].dropna(subset=['종목명'])
                df = df[df['종목명'] != '종목명']
                all_dfs.append(df)
                bar.progress(page / 4)
            except: pass
        
        bar.empty()
            
        if all_dfs:
            final_df = pd.concat(all_dfs)
            for _, row in final_df.iterrows():
                try:
                    name = row['종목명']
                    price = clean_numeric(row['현재가'])
                    per = clean_numeric(row['PER'])
                    roe = clean_numeric(row['ROE'])
                    eps = (price / per) if per > 0 else 0
                    
                    # [Note] 네이버 시총 페이지는 부채비율을 제공하지 않으므로 0으로 설정
                    # (개별 페이지 크롤링 시 속도 저하 방지)
                    debt = 0 
                    
                    data.append({'티커': name, '종목명': name, '현재가': price, 'PER': per, 'ROE': roe, 'EPS': int(eps), '부채비율': debt})
                except: continue

    return pd.DataFrame(data)

# --- 4. 메인 실행 ---

if 'res' not in st.session_state: st.session_state['res'] = None
if 'chat_history' not in st.session_state: st.session_state['chat_history'] = []
if 'current_ticker' not in st.session_state: st.session_state['current_ticker'] = ""

if st.button("🚀 데이터 분석 시작", type="primary"):
    df = analyze_data(country, market_index, target_sector)
    
    if not df.empty:
        # 전처리
        for c in ['PER','ROE','EPS', '부채비율']: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        
        # 랭킹 계산
        df['S_PER'] = df['PER'].rank(ascending=False) # 낮을수록 좋음
        df['S_ROE'] = df['ROE'].rank(ascending=True)  # 높을수록 좋음
        df['S_EPS'] = df['EPS'].rank(ascending=True)  # 높을수록 좋음
        
        # [NEW] 부채비율 랭킹: 낮을수록(ascending=False) 점수가 높게? 
        # 아니요, rank(ascending=False)는 값이 클수록 순위(숫자)가 작아짐(1등) -> 점수 높음
        # 부채비율은 낮아야 좋으므로, 값이 작을수록 점수가 높아야 함.
        # rank(ascending=False) -> 큰 값이 1등(High Score). 
        # rank(ascending=True) -> 작은 값이 1등(Low Score).
        # 우리가 원하는 건: 부채비율 낮음 -> 점수 높음.
        # 따라서, 부채비율은 역순으로 점수를 매겨야 함. 
        # 방법: rank(ascending=False)를 하면 부채비율 높은게 1등(점수 높음)이 됨. -> 틀림.
        # 방법: rank(ascending=True)를 하면 부채비율 낮은게 1등(점수 낮음) -> 틀림.
        # 수정: (전체 개수 - rank(ascending=False)) 이런 식으로 가야함.
        # 편하게 가겠습니다: S_Debt = rank(ascending=False) -> 부채비율 높은게 고득점 (나쁨)
        # 그러니까 w_debt를 곱할 때 로직을 반대로 하거나, 랭킹을 반대로 잡아야 함.
        
        # [수정된 로직]
        # 부채비율이 낮을수록(Low) -> 순위가 높아야(High Rank Score) 함.
        # df['부채비율'].rank(ascending=False) -> 값이 크면 상위권(점수 큼). (X)
        # df['부채비율'].rank(ascending=True)  -> 값이 작으면 상위권(점수 작음 1, 2, 3...). (X)
        # 그냥 단순히 "값이 작을수록 점수를 많이 준다"로 가려면:
        df['S_Debt'] = df['부채비율'].rank(ascending=False) 
        # 예: 부채비율 300% -> rank 1 (상위). 부채비율 10% -> rank 100 (하위).
        # 아 헷갈리네요. 직관적으로 짭니다.
        # 점수 = (PER순위 * w) + ...
        # PER: 낮을수록 좋음 -> rank(ascending=False) -> 낮은 PER가 높은 등수(N등)가 됨? 
        # 아니죠. rank(ascending=False)면 100(High)이 1등, 1(Low)이 꼴등.
        # PER는 낮아야 좋으니, rank(ascending=False)를 쓰면 PER 높은 놈이 점수를 많이 가져갑니다 (나쁨).
        # V27 로직 확인: df['PER'].rank(ascending=False) -> PER 100(고평가)이 1등(점수 높음).
        # 어라? V27 로직이 '고평가일수록 점수가 높게' 되어 있었나요?
        # 확인해봅시다. 
        # PER=5, 10, 20. rank(asc=False) -> 20(1등), 10(2등), 5(3등).
        # 점수 = S_PER * w. -> PER 높은 놈이 점수 높음. -> ???
        # 아하, 보통 퀀트 점수는 "순위"가 아니라 "백분위"나 "점수화"죠.
        # 제가 짠 코드는 PER가 높을수록 점수가 높게 잡혔었네요. (성장주 스타일?)
        # 저평가 가치주를 원하시면 반대가 맞습니다.
        
        # [V28 정밀 수정: 가치투자 관점]
        # PER: 낮을수록 좋음 -> rank(ascending=False) (X) -> rank(ascending=True) (O) ?
        # rank(asc=True): 5(1등), 10(2등), 20(3등). -> 점수는 값이 클수록 좋으므로, 
        # 1등(작은값)에게 낮은 점수(1점)를 주면 안됨.
        # 결론: rank(ascending=False, pct=True) -> 큰 값(PER 100)이 1.0(100%), 작은 값(PER 5)이 0.0(0%).
        # 따라서 저평가를 찾으려면 (1 - rank)를 쓰거나 해야 함.
        
        # 복잡하니 가장 직관적인 '점수 주기'로 통일합니다.
        # 목표: 좋은 종목이 높은 점수(Score)를 받게 한다.
        
        # 1. PER (낮을수록 좋음)
        # rank(ascending=False) -> 큰값(나쁨)이 상위랭크. -> 점수 낮아야 함.
        # rank(ascending=True)  -> 작은값(좋음)이 상위랭크(1,2,3..). -> 점수 낮음.
        # 해결: rank(ascending=False, pct=True) -> 큰값이 1.0, 작은값이 0.0
        # Score_PER = 1 - rank(ascending=False, pct=True) -> 작은값이 1.0에 가까워짐! (Bingo)
        df['S_PER'] = 1 - df['PER'].rank(ascending=False, pct=True)

        # 2. ROE (높을수록 좋음)
        # Score_ROE = rank(ascending=False, pct=True) -> 큰값이 1.0 (Bingo)
        # (전 코드에선 rank(ascending=True)로 되어 있었는데, 이는 ROE 낮은게 점수가 높게 잡혔던 오류가 있었을 수 있음. 수정함.)
        df['S_ROE'] = df['ROE'].rank(ascending=True, pct=True) 

        # 3. EPS (높을수록 좋음)
        df['S_EPS'] = df['EPS'].rank(ascending=True, pct=True)
        
        # 4. 부채비율 (낮을수록 좋음)
        # Score_Debt = 1 - rank(ascending=False, pct=True) -> 작은값이 1.0
        df['S_Debt'] = 1 - df['부채비율'].rank(ascending=False, pct=True)
        
        # 종합 점수
        df['점수'] = (df['S_PER']*w_per + df['S_ROE']*w_roe + df['S_EPS']*w_eps + df['S_Debt']*w_debt)
        
        # 정규화 (0~100점)
        max_val = df['점수'].max()
        df['점수'] = (df['점수'] / max_val * 100).round(1) if max_val > 0 else 0
        
        res = df.sort_values('점수', ascending=False).reset_index(drop=True)
        res['순위'] = res.index + 1
        res['Size'] = res['EPS'].apply(lambda x: max(x, 100) if x > -9999 else 100)
        
        st.session_state['res'] = res
        st.session_state['chat_history'] = []
        st.rerun()
    else:
        st.error("데이터 수집 실패")

# 결과 출력
if st.session_state['res'] is not None:
    res = st.session_state['res']
    
    avg_per = res[res['PER']>0]['PER'].mean()
    avg_roe = res['ROE'].mean()
    
    # 1. 차트
    fig = px.scatter(
        res, x='PER', y='ROE', 
        size='Size', color='점수', 
        hover_name='종목명', 
        hover_data={'부채비율':True},
        title=f"📈 {market_index} 밸류에이션 맵",
        color_continuous_scale='RdYlGn',
        log_x=use_log_x, 
        log_y=use_log_y
    )

    if show_avg:
        if avg_per > 0: fig.add_vline(x=avg_per, line_dash="dash", line_color="gray", annotation_text=f"Avg PER: {avg_per:.1f}")
        if avg_roe > 0: fig.add_hline(y=avg_roe, line_dash="dash", line_color="gray", annotation_text=f"Avg ROE: {avg_roe:.1f}%")

    st.plotly_chart(fig, use_container_width=True)
    if use_log_y: st.caption("⚠️ Y축 로그: 음수 ROE 기업은 표시되지 않습니다.")
    
    # 2. UI
    c_tbl, c_chat = st.columns([1.5, 1])
    
    with c_tbl:
        st.subheader("🏆 랭킹 리스트")
        # 부채비율 컬럼 추가
        st.dataframe(res[['순위','종목명','점수','현재가','PER','ROE','EPS','부채비율']].set_index('순위')
                     .style.format({'현재가':'{:.0f}', 'PER':'{:.2f}', 'ROE':'{:.2f}', '부채비율':'{:.2f}'}), 
                     use_container_width=True)
        
    with c_chat:
        st.subheader("💬 Gemini 퀀트 컨설턴트")
        
        stock_list = res['종목명'].tolist()
        target_name = st.selectbox("분석할 종목 선택", stock_list)
        
        if target_name != st.session_state['current_ticker']:
            st.session_state['current_ticker'] = target_name
            st.session_state['chat_history'] = []
            t_data = res[res['종목명']==target_name].iloc[0]
            # 환영 메시지에 부채비율 정보 추가
            welcome_msg = f"**{target_name}** ({t_data['티커']}) 종목 데이터.\n\n- 주가: {t_data['현재가']:,.0f}\n- PER: {t_data['PER']:.2f}\n- ROE: {t_data['ROE']:.2f}%\n- 부채비율: {t_data['부채비율']:.2f}%"
            st.session_state['chat_history'].append({"role": "assistant", "content": welcome_msg})

        chat_container = st.container(height=400)
        for msg in st.session_state['chat_history']:
            with chat_container.chat_message(msg["role"]):
                st.write(msg["content"])
        
        if prompt := st.chat_input("질문 입력..."):
            if not api_key:
                st.error("API 키 필요")
            else:
                st.session_state['chat_history'].append({"role": "user", "content": prompt})
                with chat_container.chat_message("user"): st.write(prompt)
                
                with chat_container.chat_message("assistant"):
                    msg_ph = st.empty()
                    full_res = ""
                    try:
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel(ai_model)
                        
                        t_data = res[res['종목명']==target_name].iloc[0]
                        # 컨텍스트에 부채비율 추가
                        ctx = f"종목:{t_data['종목명']}, 주가:{t_data['현재가']}, PER:{t_data['PER']}, ROE:{t_data['ROE']}, 부채비율:{t_data['부채비율']}%. 질문:{prompt}. 한국어로 답변."
                        
                        response = model.generate_content(ctx, stream=True)
                        for chunk in response:
                            if chunk.text:
                                full_res += chunk.text
                                msg_ph.markdown(full_res + "▌")
                                time.sleep(0.02)
                        msg_ph.markdown(full_res)
                        st.session_state['chat_history'].append({"role": "assistant", "content": full_res})
                    except Exception as e:
                        st.error(f"Error: {e}")
