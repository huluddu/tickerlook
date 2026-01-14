import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import time
import requests
import numpy as np
from pykrx import stock
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="AI 퀀트 V41 (Hybrid Fetch)", layout="wide")
st.title("🤖 AI 퀀트 스크리너 V41 (Hybrid Fetching)")
st.markdown("주가 데이터(Bulk)와 재무 데이터(Individual) 수집 방식을 분리하여 **데이터 수집 성공률을 극대화**했습니다.")

# --- 2. 사이드바 ---
st.sidebar.header("1. 시장 선택")
country = st.sidebar.radio("국가", ["미국 (US)", "한국 (KR)"], horizontal=True)

market_index = ""
target_sector = "전체"

if country == "미국 (US)":
    market_index = st.sidebar.selectbox("지수", ["S&P 500 / NASDAQ", "Russell 2000 (중소형)"])
    target_sector = st.sidebar.selectbox("섹터 (업종)", [
        "전체 (All)",
        "기술 (Technology)", "커뮤니케이션 (Communication)", "헬스케어 (Healthcare)", 
        "소비재 (Consumer)", "금융 (Financial)", "에너지/산업 (Energy/Ind)"
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
w_debt = st.sidebar.slider("안정성 (부채비율)", 0, 100, 10)

st.sidebar.markdown("---")
st.sidebar.header("🔑 AI 설정")
api_key = st.sidebar.text_input("Gemini API Key", type="password")

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

def get_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
    })
    return session

# --- 차트 그리기 ---
def draw_chart(ticker_code, country_code, market_index=""):
    try:
        df = pd.DataFrame()
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=365*3)
        
        if country_code == "한국 (KR)":
            s_str = start_dt.strftime("%Y%m%d")
            e_str = end_dt.strftime("%Y%m%d")
            try:
                df = stock.get_market_ohlcv(s_str, e_str, ticker_code)
                if df.empty: return None
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount', 'Rate'][:len(df.columns)]
                df = df[['Open', 'High', 'Low', 'Close']]
            except:
                suffix = ".KQ" if "KOSDAQ" in market_index else ".KS"
                df = yf.download(f"{ticker_code}{suffix}", start=start_dt, end=end_dt, progress=False)
        else:
            df = yf.download(ticker_code, start=start_dt, end=end_dt, progress=False)
            
        if len(df) < 10: return None

        if isinstance(df['Close'], pd.DataFrame):
            df_new = pd.DataFrame()
            df_new['Close'] = df['Close'].iloc[:, 0]
            df_new['Open'] = df['Open'].iloc[:, 0]
            df_new['High'] = df['High'].iloc[:, 0]
            df_new['Low'] = df['Low'].iloc[:, 0]
            df = df_new

        df['MA120'] = df['Close'].rolling(window=120).mean()

        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='주가'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA120'], mode='lines', name='120일선', line=dict(color='orange', width=2)))

        fig.update_layout(
            title="📊 3년 주가 흐름 & 120일선",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            height=500,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=50, b=20)
        )
        return fig
    except: return None

# --- 기술적 지표 계산 ---
def calculate_technicals(ticker_code, country_code, market_index=""):
    df = pd.DataFrame()
    try:
        if country_code == "한국 (KR)":
            end_dt = datetime.now().strftime("%Y%m%d")
            start_dt = (datetime.now() - timedelta(days=200)).strftime("%Y%m%d")
            try:
                df = stock.get_market_ohlcv(start_dt, end_dt, ticker_code)
                if df.empty: raise Exception
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount', 'Rate'][:len(df.columns)]
                df = df[['Open', 'High', 'Low', 'Close']]
            except:
                suffix = ".KQ" if "KOSDAQ" in market_index else ".KS"
                df = yf.download(f"{ticker_code}{suffix}", period="6mo", progress=False)
        else:
            df = yf.download(ticker_code, period="6mo", progress=False)
            
        if len(df) < 20: return None

        close = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
        high = df['High'].iloc[:, 0] if isinstance(df['High'], pd.DataFrame) else df['High']
        low = df['Low'].iloc[:, 0] if isinstance(df['Low'], pd.DataFrame) else df['Low']
        
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        lowest_low = low.rolling(14).min()
        highest_high = high.rolling(14).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(20).mean()
        mean_dev = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (tp - sma_tp) / (0.015 * mean_dev)
        
        w_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        
        return {"RSI": rsi.iloc[-1], "Stochastic_K": k_percent.iloc[-1], "CCI": cci.iloc[-1], "Williams_R": w_r.iloc[-1]}
    except: return None

# --- 3. 데이터 수집 함수 (Hybrid V41) ---
@st.cache_data
def analyze_data(country, index, sector):
    data = []
    
    # 🇺🇸 미국 (Hybrid: Bulk Price + Gentle Info)
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
        if index == "Russell 2000 (중소형)": target_tickers = russell_tickers
        else:
            if "전체" in sector:
                for k in sector_map: target_tickers += sector_map[k]
            else:
                target_tickers = sector_map.get(sector, [])

        bar = st.progress(0, text=f"🇺🇸 {sector} : 주가 데이터 일괄 다운로드 중... (1/2)")
        
        # [Step 1] 주가(Price)는 한 방에 가져오기 (성공률 99%)
        try:
            bulk_data = yf.download(target_tickers, period="1d", progress=False)
            # 멀티인덱스 처리 ('Close', 'AAPL')
            has_bulk = not bulk_data.empty
        except:
            has_bulk = False

        bar.progress(0.3, text=f"🇺🇸 {sector} : 재무 데이터 상세 조회 중... (2/2)")
        
        # [Step 2] 재무(Info)는 살살 가져오기
        for i, t in enumerate(target_tickers):
            try:
                # 1. Price 확보
                price = 0
                if has_bulk:
                    try:
                        # yfinance 버전에 따라 컬럼 구조가 다를 수 있음
                        if isinstance(bulk_data['Close'], pd.DataFrame):
                            price = bulk_data['Close'][t].iloc[-1]
                        else:
                            price = bulk_data['Close'].iloc[-1]
                    except: price = 0
                
                # 2. Fundamentals 확보 (재시도 로직 포함)
                ticker = yf.Ticker(t)
                info = {}
                
                # 시도 1
                try: 
                    info = ticker.info
                except: 
                    time.sleep(1) # 실패시 1초 쉬고 재시도
                    try: info = ticker.info
                    except: info = {}
                
                # 0.5초 대기 (서버 부하 방지)
                time.sleep(0.5)

                name = info.get('shortName', t)
                if price == 0: price = info.get('currentPrice', 0) # Bulk 실패시 info에서 재시도
                
                per = info.get('trailingPE', 0)
                roe = info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0
                eps = info.get('trailingEps', 0)
                debt = info.get('debtToEquity', 0)

                # 가격이라도 있으면 추가
                if price > 0 or per > 0:
                    data.append({'티커':t, '종목명':name, '현재가':price, 'PER':per, 'ROE':roe, 'EPS':eps, '부채비율':debt})
            except: pass
            
            # 진행상황 업데이트 (0.3 ~ 1.0 구간 매핑)
            prog = 0.3 + (0.7 * (i+1) / len(target_tickers))
            bar.progress(min(prog, 1.0))
            
        bar.empty()

    # 🇰🇷 한국 (Naver + Code Extract)
    else:
        session = get_session()
        sosok = 0 if index == 'KOSPI' else 1
        url_base = f"https://finance.naver.com/sise/sise_market_sum.naver?sosok={sosok}&page="
        
        all_dfs = []
        bar = st.progress(0, text="🇰🇷 네이버 증권 데이터 스캔 중...")
        
        for page in range(1, 5): 
            try:
                res = session.get(url_base + str(page))
                soup = BeautifulSoup(res.text, 'html.parser')
                code_map = {}
                for a in soup.select('a.tltle'):
                    if 'code=' in a['href']:
                        code_map[a.text] = a['href'].split('code=')[1]
                
                dfs = pd.read_html(res.text, encoding='euc-kr', header=0, flavor='bs4')
                df = dfs[1].dropna(subset=['종목명'])
                df = df[df['종목명'] != '종목명']
                df['Code'] = df['종목명'].map(code_map)
                all_dfs.append(df)
                bar.progress(page / 4)
                time.sleep(0.3)
            except: pass
        bar.empty()
            
        if all_dfs:
            final_df = pd.concat(all_dfs)
            for _, row in final_df.iterrows():
                try:
                    name = row['종목명']
                    code = row.get('Code', '')
                    if not code or pd.isna(code): continue
                    price = clean_numeric(row['현재가'])
                    per = clean_numeric(row['PER'])
                    roe = clean_numeric(row['ROE'])
                    eps = (price/per) if per>0 else 0
                    debt = 0 
                    data.append({'티커':code, '종목명':name, '현재가':price, 'PER':per, 'ROE':roe, 'EPS':int(eps), '부채비율':debt})
                except: continue

    return pd.DataFrame(data)

# --- 4. 메인 실행 ---
if 'res' not in st.session_state: st.session_state['res'] = None
if 'chat_history' not in st.session_state: st.session_state['chat_history'] = []
if 'current_ticker' not in st.session_state: st.session_state['current_ticker'] = ""

if st.button("🚀 데이터 분석 시작", type="primary"):
    df = analyze_data(country, market_index, target_sector)
    
    if not df.empty:
        for c in ['PER','ROE','EPS','부채비율']: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        
        df['temp_per'] = df['PER'].apply(lambda x: x if x > 0 else 99999)
        max_p, min_p = df['temp_per'].max(), df['temp_per'].min()
        df['S_PER'] = (max_p - df['temp_per']) / ((max_p - min_p) if max_p != min_p else 1)
        if not df[df['PER']<=0].empty: df.loc[df['PER']<=0, 'S_PER'] = 0

        max_r, min_r = df['ROE'].max(), df['ROE'].min()
        df['S_ROE'] = (df['ROE'] - min_r) / ((max_r - min_r) if max_r != min_r else 1)

        max_e, min_e = df['EPS'].max(), df['EPS'].min()
        df['S_EPS'] = (df['EPS'] - min_e) / ((max_e - min_e) if max_e != min_e else 1)
        
        max_d, min_d = df['부채비율'].max(), df['부채비율'].min()
        df['S_Debt'] = (max_d - df['부채비율']) / ((max_d - min_d) if max_d != min_d else 1)
        
        df['점수'] = (df['S_PER']*w_per + df['S_ROE']*w_roe + df['S_EPS']*w_eps + df['S_Debt']*w_debt)
        
        final_max = df['점수'].max()
        df['점수'] = (df['점수'] / final_max * 100).round(1) if final_max > 0 else 0
        
        res = df.sort_values('점수', ascending=False).reset_index(drop=True)
        res['순위'] = res.index + 1
        res['Size'] = res['EPS'].apply(lambda x: max(x, 100) if x > -9999 else 100)
        
        st.session_state['res'] = res
        st.session_state['chat_history'] = []
        st.rerun()
    else:
        st.error("데이터 수집 실패. (잠시 후 다시 시도해주세요)")

if st.session_state['res'] is not None:
    res = st.session_state['res']
    
    avg_per = res[res['PER']>0]['PER'].mean()
    avg_roe = res['ROE'].mean()
    
    # 1. 차트 영역
    fig = px.scatter(
        res, x='PER', y='ROE', 
        size='Size', color='점수', 
        hover_name='종목명', 
        hover_data={'부채비율':True, 'EPS':True},
        title=f"📈 {market_index} 밸류에이션 맵",
        color_continuous_scale='RdYlGn',
        log_x=use_log_x, 
        log_y=use_log_y
    )
    if show_avg:
        if avg_per > 0: fig.add_vline(x=avg_per, line_dash="dash", line_color="gray", annotation_text=f"Avg PER: {avg_per:.1f}")
        if avg_roe > 0: fig.add_hline(y=avg_roe, line_dash="dash", line_color="gray", annotation_text=f"Avg ROE: {avg_roe:.1f}%")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 2. 랭킹 리스트
    st.subheader("🏆 랭킹 리스트")
    st.dataframe(res[['순위','종목명','점수','현재가','PER','ROE','EPS','부채비율']].set_index('순위')
                    .style.format({'현재가':'{:.0f}', 'PER':'{:.2f}', 'ROE':'{:.2f}', 'EPS':'{:.2f}', '부채비율':'{:.2f}'}), 
                    use_container_width=True)

    st.markdown("---") 

    # 3. 퀀트 컨설턴트
    st.subheader("💬 Gemini 퀀트 컨설턴트 (종목 심층 분석)")
    
    stock_list = res['종목명'].tolist()
    c_sel, c_dummy = st.columns([1, 2])
    with c_sel:
        target_name = st.selectbox("분석할 종목을 선택하세요", stock_list)

    if target_name != st.session_state['current_ticker']:
        st.session_state['current_ticker'] = target_name
        st.session_state['chat_history'] = []
        
        t_data = res[res['종목명']==target_name].iloc[0]
        ticker_code = t_data['티커']
        
        chart_fig = None
        tech_data = None
        
        with st.spinner(f"{target_name} 차트 및 지표 분석 중..."):
             chart_fig = draw_chart(str(ticker_code), country, market_index)
             tech_data = calculate_technicals(str(ticker_code), country, market_index)
        
        if chart_fig: st.plotly_chart(chart_fig, use_container_width=True)

        if tech_data:
            tech_msg = f"""
            📊 **기술적 지표 (Technical)**
            - **RSI**: {tech_data['RSI']:.2f}
            - **Stochastic K**: {tech_data['Stochastic_K']:.2f}
            - **CCI**: {tech_data['CCI']:.2f}
            - **Williams %R**: {tech_data['Williams_R']:.2f}
            """
            st.session_state['tech_context'] = tech_msg
        else:
            tech_msg = "\n(차트 데이터 수집 실패)"
            st.session_state['tech_context'] = ""

        welcome_msg = f"**{target_name}** ({ticker_code})\nPER: {t_data['PER']:.2f} | ROE: {t_data['ROE']:.2f}% | 부채: {t_data['부채비율']:.0f}%" + tech_msg
        st.session_state['chat_history'].append({"role": "assistant", "content": welcome_msg})

    chat_container = st.container(height=500)
    for msg in st.session_state['chat_history']:
        with chat_container.chat_message(msg["role"]):
            st.write(msg["content"])
    
    if prompt := st.chat_input("질문 입력..."):
        if not api_key: st.error("API 키 필요")
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
                    tech_info = st.session_state.get('tech_context', '')
                    ctx = f"종목:{t_data['종목명']}, 재무:[현재가:{t_data['현재']}, PER:{t_data['PER']}, ROE:{t_data['ROE']}, 부채:최근 자료 검색해서 확인 해줘]. 기술적분석:{tech_info}. 질문:{prompt}. 한국어 답변."
                    response = model.generate_content(ctx, stream=True)
                    for chunk in response:
                        if chunk.text:
                            full_res += chunk.text
                            msg_ph.markdown(full_res + "▌")
                            time.sleep(0.02)
                    msg_ph.markdown(full_res)
                    st.session_state['chat_history'].append({"role": "assistant", "content": full_res})
                except Exception as e: st.error(f"Error: {e}")

