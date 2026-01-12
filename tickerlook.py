import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import google.generativeai as genai
import time
import requests

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="AI 퀀트 V30 (Global Fix)", layout="wide")
st.title("🤖 AI 퀀트 스크리너 V30 (Yahoo & Naver Fix)")
st.markdown("클라우드 환경에서 **Yahoo Finance(미국)**와 **네이버(한국)**의 차단을 모두 우회합니다.")

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
w_debt = st.sidebar.slider("안정성 (부채비율)", 0, 100, 0)

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

# [핵심] 차단 우회용 세션 생성 (Yahoo & Naver 공용)
def get_session():
    session = requests.Session()
    # 진짜 브라우저처럼 보이는 헤더
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.google.com/'
    })
    return session

# --- 3. 데이터 수집 함수 ---
@st.cache_data
def analyze_data(country, index, sector):
    data = []
    session = get_session() # 세션 가져오기
    
    # ==========================================
    # 🇺🇸 미국 시장 (Yahoo Finance Fix)
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
                # [핵심] yfinance에 커스텀 세션 주입
                ticker = yf.Ticker(t, session=session)
                
                # 1. fast_info 시도 (빠르고 차단 덜 됨)
                try:
                    price = ticker.fast_info['last_price']
                except:
                    price = 0
                
                # 2. info 시도 (재무정보)
                try:
                    info = ticker.info
                    name = info.get('shortName', t)
                    per = info.get('trailingPE', 0)
                    roe = info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0
                    eps = info.get('trailingEps', 0)
                    debt = info.get('debtToEquity', 0)
                    if price == 0: price = info.get('currentPrice', 0)
                except:
                    # 실패 시 기본값
                    name = t
                    per, roe, eps, debt = 0, 0, 0, 0
                
                if price > 0: # 가격이 있을 때만 추가
                    data.append({
                        '티커': t, '종목명': name, '현재가': price,
                        'PER': per, 'ROE': roe, 'EPS': eps, '부채비율': debt
                    })
            except: 
                pass
            bar.progress((i+1)/len(target_tickers))
        bar.empty()

    # ==========================================
    # 🇰🇷 한국 시장 (Naver Fix)
    # ==========================================
    else:
        sosok = 0 if index == 'KOSPI' else 1
        url_base = f"https://finance.naver.com/sise/sise_market_sum.naver?sosok={sosok}&page="
        
        all_dfs = []
        bar = st.progress(0, text="🇰🇷 네이버 증권 데이터(상위 200개) 스캔 중...")
        
        for page in range(1, 5): 
            try:
                # [핵심] requests로 먼저 html 가져옴 (헤더 포함)
                res_html = session.get(url_base + str(page))
                dfs = pd.read_html(res_html.text, encoding='euc-kr', header=0)
                
                df = dfs[1].dropna(subset=['종목명'])
                df = df[df['종목명'] != '종목명']
                all_dfs.append(df)
                bar.progress(page / 4)
                time.sleep(0.2) # 약간의 딜레이
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
        
        df['S_PER'] = 1 - df['PER'].rank(ascending=False, pct=True) 
        df['S_ROE'] = df['ROE'].rank(ascending=True, pct=True)
        df['S_EPS'] = df['EPS'].rank(ascending=True, pct=True)
        df['S_Debt'] = 1 - df['부채비율'].rank(ascending=False, pct=True)
        
        df['점수'] = (df['S_PER']*w_per + df['S_ROE']*w_roe + df['S_EPS']*w_eps + df['S_Debt']*w_debt)
        
        max_val = df['점수'].max()
        df['점수'] = (df['점수'] / max_val * 100).round(1) if max_val > 0 else 0
        
        res = df.sort_values('점수', ascending=False).reset_index(drop=True)
        res['순위'] = res.index + 1
        res['Size'] = res['EPS'].apply(lambda x: max(x, 100) if x > -9999 else 100)
        
        st.session_state['res'] = res
        st.session_state['chat_history'] = []
        st.rerun()
    else:
        st.error("데이터 수집 실패. (잠시 후 다시 시도하거나 서버 상태를 확인하세요)")

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
            welcome_msg = f"**{target_name}** ({t_data['티커']})\n- PER: {t_data['PER']:.2f}\n- ROE: {t_data['ROE']:.2f}%\n- 부채비율: {t_data['부채비율']:.2f}%"
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
