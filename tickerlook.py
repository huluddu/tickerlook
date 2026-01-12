import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import google.generativeai as genai
import time
import requests

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="AI 퀀트 V31 (Debug)", layout="wide")
st.title("🤖 AI 퀀트 스크리너 V31 (Debug Mode)")
st.warning("⚠️ 디버그 모드: 에러 발생 시 상세 내용을 화면에 출력합니다.")

# --- 2. 사이드바 ---
st.sidebar.header("1. 시장 선택")
country = st.sidebar.radio("국가", ["미국 (US)", "한국 (KR)"], horizontal=True)

market_index = ""
target_sector = "전체"

if country == "미국 (US)":
    market_index = st.sidebar.selectbox("지수", ["S&P 500 / NASDAQ", "Russell 2000"])
    target_sector = st.sidebar.selectbox("섹터", ["전체", "기술", "금융", "헬스케어", "소비재", "에너지/산업"])
else:
    market_index = st.sidebar.selectbox("지수", ["KOSPI", "KOSDAQ"])
    st.sidebar.caption("한국: 시총 상위 통합 검색")

st.sidebar.markdown("---")
w_per = st.sidebar.slider("저평가 (PER)", 0, 100, 40)
w_roe = st.sidebar.slider("수익성 (ROE)", 0, 100, 40)
w_eps = st.sidebar.slider("성장성 (EPS)", 0, 100, 10)
w_debt = st.sidebar.slider("안정성 (부채비율)", 0, 100, 0)

st.sidebar.markdown("---")
api_key = st.sidebar.text_input("Gemini API Key", type="password")

available_models = ["gemini-1.5-flash", "gemini-pro"]
if api_key:
    try:
        genai.configure(api_key=api_key)
        scanned = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        if scanned: available_models = scanned
    except: pass
ai_model = st.sidebar.selectbox("AI 모델", available_models)

# --- 유틸리티 ---
def clean_numeric(value):
    try:
        if isinstance(value, str):
            value = value.replace(',', '').replace('N/A', '0').replace('-', '0')
        return float(value)
    except: return 0.0

def get_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
    })
    return session

# --- 3. 데이터 수집 (디버깅용) ---
@st.cache_data
def analyze_data(country, index, sector):
    data = []
    error_logs = [] # 에러 기록용
    session = get_session()
    
    # ----------------------------------
    # 🇺🇸 미국
    # ----------------------------------
    if country == "미국 (US)":
        sector_map = {
            "기술": ['AAPL', 'MSFT', 'NVDA', 'AMD', 'INTC', 'PLTR'],
            "금융": ['JPM', 'BAC', 'V', 'MA'],
            "헬스케어": ['LLY', 'UNH', 'JNJ', 'PFE'],
            "소비재": ['TSLA', 'AMZN', 'KO', 'PEP'],
            "에너지/산업": ['XOM', 'CAT', 'BA']
        }
        # (샘플을 줄여서 테스트 속도 향상)
        if sector == "전체": 
            targets = ['AAPL', 'MSFT', 'TSLA', 'GOOGL', 'NVDA', 'AMZN', 'META']
        else:
            targets = sector_map.get(sector, ['AAPL', 'TSLA'])
            
        bar = st.progress(0, text="미국 데이터 접속 중...")
        for i, t in enumerate(targets):
            try:
                # yfinance 디버깅
                ticker = yf.Ticker(t, session=session)
                info = ticker.info # 여기서 에러가 나는지 확인
                
                # 데이터가 비어있으면 에러로 간주
                if not info or 'regularMarketPrice' not in info and 'currentPrice' not in info:
                    raise ValueError(f"Empty info for {t}")

                data.append({
                    '티커': t, '종목명': info.get('shortName', t), 
                    '현재가': info.get('currentPrice', 0),
                    'PER': info.get('trailingPE', 0), 
                    'ROE': info.get('returnOnEquity', 0)*100 if info.get('returnOnEquity') else 0,
                    'EPS': info.get('trailingEps', 0),
                    '부채비율': info.get('debtToEquity', 0)
                })
            except Exception as e:
                error_logs.append(f"🇺🇸 {t} 실패: {str(e)}")
            bar.progress((i+1)/len(targets))
        bar.empty()

    # ----------------------------------
    # 🇰🇷 한국
    # ----------------------------------
    else:
        sosok = 0 if index == 'KOSPI' else 1
        url_base = f"https://finance.naver.com/sise/sise_market_sum.naver?sosok={sosok}&page="
        
        all_dfs = []
        bar = st.progress(0, text="한국 데이터 접속 중...")
        
        for page in range(1, 3): # 테스트용 2페이지만
            try:
                res = session.get(url_base + str(page))
                # HTTP 상태 코드 확인
                if res.status_code != 200:
                    error_logs.append(f"🇰🇷 페이지 {page} 접속 실패 (Status: {res.status_code})")
                    continue
                
                dfs = pd.read_html(res.text, encoding='euc-kr', header=0)
                if len(dfs) < 2:
                    error_logs.append(f"🇰🇷 페이지 {page} 표를 못 찾음")
                    continue
                    
                df = dfs[1].dropna(subset=['종목명'])
                df = df[df['종목명'] != '종목명']
                all_dfs.append(df)
            except Exception as e:
                error_logs.append(f"🇰🇷 페이지 {page} 파싱 에러: {str(e)}")
            
            bar.progress(page/2)
        
        bar.empty()
        
        if all_dfs:
            final_df = pd.concat(all_dfs)
            for _, row in final_df.iterrows():
                try:
                    name = row['종목명']
                    price = clean_numeric(row['현재가'])
                    per = clean_numeric(row['PER'])
                    roe = clean_numeric(row['ROE'])
                    eps = (price/per) if per>0 else 0
                    data.append({'티커':name, '종목명':name, '현재가':price, 'PER':per, 'ROE':roe, 'EPS':int(eps), '부채비율':0})
                except: continue

    return pd.DataFrame(data), error_logs

# --- 4. 메인 실행 ---
if 'res' not in st.session_state: st.session_state['res'] = None

if st.button("🚀 분석 시작 (디버그)", type="primary"):
    df, errors = analyze_data(country, market_index, target_sector)
    
    # [핵심] 에러가 있으면 화면에 빨간색으로 토해냄
    if errors:
        st.error("⚠️ 데이터 수집 중 문제가 발생했습니다:")
        for err in errors[:5]: # 너무 많으면 5개만
            st.error(err)
    
    if not df.empty:
        for c in ['PER','ROE','EPS','부채비율']: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        
        df['S_PER'] = 1 - df['PER'].rank(ascending=False, pct=True)
        df['S_ROE'] = df['ROE'].rank(ascending=True, pct=True)
        df['S_EPS'] = df['EPS'].rank(ascending=True, pct=True)
        df['S_Debt'] = 1 - df['부채비율'].rank(ascending=False, pct=True)
        
        df['점수'] = (df['S_PER']*w_per + df['S_ROE']*w_roe + df['S_EPS']*w_eps + df['S_Debt']*w_debt)
        max_val = df['점수'].max()
        df['점수'] = (df['점수']/max_val*100).round(1) if max_val > 0 else 0
        
        res = df.sort_values('점수', ascending=False).reset_index(drop=True)
        res['Size'] = res['EPS'].apply(lambda x: max(x, 100) if x > -9999 else 100)
        st.session_state['res'] = res
        st.rerun()
    else:
        st.warning("데이터가 하나도 없습니다. 위 에러 메시지를 확인해주세요.")

# 결과 출력
if st.session_state['res'] is not None:
    res = st.session_state['res']
    st.success(f"✅ {len(res)}개 종목 분석 성공")
    
    fig = px.scatter(res, x='PER', y='ROE', size='Size', color='점수', hover_name='종목명', title="Map")
    st.plotly_chart(fig, use_container_width=True)
    
    c1, c2 = st.columns([1.5, 1])
    with c1: st.dataframe(res)
    with c2:
        st.write("AI 분석")
        target = st.selectbox("종목", res['종목명'].unique())
        if st.button("분석"):
            row = res[res['종목명']==target].iloc[0]
            genai.configure(api_key=api_key)
            try:
                m = genai.GenerativeModel(ai_model)
                r = m.generate_content(f"{target} 분석해줘. 주가 {row['현재가']}")
                st.write(r.text)
            except Exception as e: st.error(f"AI Error: {e}")
