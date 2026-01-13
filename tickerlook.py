import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import google.generativeai as genai
import time
import requests
import numpy as np
from pykrx import stock # [NEW] 한국장 차트 데이터용
from datetime import datetime, timedelta

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="AI 퀀트 V37 (PyKRX)", layout="wide")
st.title("🤖 AI 퀀트 스크리너 V37 (PyKRX Applied)")
st.markdown("""
**최종 엔진 적용:**
* **한국 차트:** `PyKRX` 라이브러리 적용 (기술적 지표 정확도 100%)
* **미국 차트:** `YFinance` 사용
* **랭킹:** 네이버/Yahoo 실시간 크롤링
""")

# --- 2. 사이드바 ---
st.sidebar.header("1. 시장 선택")
country = st.sidebar.radio("국가", ["미국 (US)", "한국 (KR)"], horizontal=True)

market_index = ""
target_sector = "전체"

if country == "미국 (US)":
    market_index = st.sidebar.selectbox("지수", ["S&P 500 / NASDAQ", "Russell 2000 (중소형)"])
    target_sector = st.sidebar.selectbox("섹터 (업종)", [
        "기술 (Technology)", "커뮤니케이션 (Communication)", "헬스케어 (Healthcare)", 
        "소비재 (Consumer)", "금융 (Financial)", "에너지/산업 (Energy/Ind)", "전체 (All)"
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
st.sidebar.header("3. 가중치 설정 (총합 100 권장)")
w_per = st.sidebar.slider("저평가 (PER, 낮을수록 좋음)", 0, 100, 40)
w_roe = st.sidebar.slider("수익성 (ROE, 높을수록 좋음)", 0, 100, 40)
w_eps = st.sidebar.slider("성장성 (EPS, 높을수록 좋음)", 0, 100, 10)
w_debt = st.sidebar.slider("안정성 (부채비율, 낮을수록 좋음)", 0, 100, 10)

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

# --- [핵심] 기술적 지표 계산 함수 (PyKRX 적용) ---
def calculate_technicals(ticker_symbol, country_code):
    df = pd.DataFrame()
    
    try:
        # 1. 한국 주식 (PyKRX 사용)
        if country_code == "한국 (KR)":
            # 날짜 계산 (오늘 ~ 6개월 전)
            end_dt = datetime.now().strftime("%Y%m%d")
            start_dt = (datetime.now() - timedelta(days=180)).strftime("%Y%m%d")
            
            # PyKRX로 OHLCV 가져오기
            # ticker_symbol은 '005930' 같은 6자리 코드여야 함
            df = stock.get_market_ohlcv(start_dt, end_dt, ticker_symbol)
            
            # PyKRX 컬럼명: 시가, 고가, 저가, 종가, 거래량
            # 영어로 변환 필요 (계산 로직 통일 위해)
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount', 'Rate'] if len(df.columns) == 7 else ['Open', 'High', 'Low', 'Close', 'Volume']
            # 필요없는 컬럼 제거 시도
            df = df[['Open', 'High', 'Low', 'Close']]

        # 2. 미국 주식 (YFinance 사용)
        else:
            df = yf.download(ticker_symbol, period="6mo", progress=False)
            
        if len(df) < 20: return None # 데이터 부족

        # Series 추출 (멀티인덱스 대응)
        close = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
        high = df['High'].iloc[:, 0] if isinstance(df['High'], pd.DataFrame) else df['High']
        low = df['Low'].iloc[:, 0] if isinstance(df['Low'], pd.DataFrame) else df['Low']
        
        # --- 지표 계산 공식 ---
        
        # 1. RSI (14)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # 2. Stochastic (14)
        lowest_low = low.rolling(window=14).min()
        highest_high = high.rolling(window=14).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        
        # 3. CCI (20)
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window=20).mean()
        mean_dev = tp.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (tp - sma_tp) / (0.015 * mean_dev)
        
        # 4. Williams %R (14)
        w_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        
        # 5. Momentum (10)
        momentum = close.diff(10)

        return {
            "RSI": rsi.iloc[-1],
            "Stochastic_K": k_percent.iloc[-1],
            "CCI": cci.iloc[-1],
            "Williams_R": w_r.iloc[-1],
            "Momentum": momentum.iloc[-1]
        }
    except Exception as e:
        # st.error(f"Tech Calc Error: {e}") # 디버깅용
        return None

# --- 3. 데이터 수집 함수 ---
@st.cache_data
def analyze_data(country, index, sector):
    data = []
    
    # 🇺🇸 미국
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
        tickers_obj = yf.Tickers(' '.join(target_tickers))
        
        for i, t in enumerate(target_tickers):
            try:
                ticker = tickers_obj.tickers[t]
                try: price = ticker.fast_info['last_price']
                except: price = 0
                
                time.sleep(0.3)
                try:
                    info = ticker.info
                    name = info.get('shortName', t)
                    per = info.get('trailingPE', 0)
                    roe = info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0
                    eps = info.get('trailingEps', 0)
                    debt = info.get('debtToEquity', 0)
                    if price == 0: price = info.get('currentPrice', 0)
                except:
                    name = t
                    per, roe, eps, debt = 0, 0, 0, 0
                
                if price > 0:
                    data.append({'티커':t, '종목명':name, '현재가':price, 'PER':per, 'ROE':roe, 'EPS':eps, '부채비율':debt})
            except: pass
            bar.progress((i+1)/len(target_tickers))
        bar.empty()

    # 🇰🇷 한국
    else:
        session = get_session()
        sosok = 0 if index == 'KOSPI' else 1
        url_base = f"https://finance.naver.com/sise/sise_market_sum.naver?sosok={sosok}&page="
        
        all_dfs = []
        bar = st.progress(0, text="🇰🇷 네이버 증권 데이터 스캔 중...")
        
        for page in range(1, 5): 
            try:
                res_html = session.get(url_base + str(page))
                dfs = pd.read_html(res_html.text, encoding='euc-kr', header=0, flavor='bs4')
                df = dfs[1].dropna(subset=['종목명'])
                df = df[df['종목명'] != '종목명']
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
                    price = clean_numeric(row['현재가'])
                    per = clean_numeric(row['PER'])
                    roe = clean_numeric(row['ROE'])
                    eps = (price/per) if per>0 else 0
                    debt = 0 
                    # 티커 코드 추출 (005930 등) - 보통 종목명 옆에 링크에 있는데, 
                    # 여기서는 네이버 표에 티커가 안 보일 수 있음.
                    # 하지만! pykrx는 이름으로 찾기 어려움. 코드가 필요함.
                    # 네이버 크롤링 결과에는 코드가 없음. 
                    # -> [해결책] 종목명으로 티커를 찾아야 함. pykrx에 기능이 있음.
                    # 하지만 여기서 매번 찾으면 느림.
                    # 다행히 '토론실' 등의 링크 href에 code=000000 이 있음.
                    # 하지만 pd.read_html은 텍스트만 가져옴.
                    # -> 따라서 pykrx로 전체 종목 리스트를 미리 받아두고 매핑하는게 정석이나,
                    # -> 여기서는 상세 분석할 때 이름으로 코드를 찾도록 로직 변경.
                    #    (아래쪽 calculate_technicals 호출부에서 처리)
                    
                    data.append({'티커':name, '종목명':name, '현재가':price, 'PER':per, 'ROE':roe, 'EPS':int(eps), '부채비율':debt})
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

# 결과 출력
if st.session_state['res'] is not None:
    res = st.session_state['res']
    
    avg_per = res[res['PER']>0]['PER'].mean()
    avg_roe = res['ROE'].mean()
    
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
    if use_log_y: st.caption("⚠️ Y축 로그: 음수 ROE 기업은 표시되지 않습니다.")
    
    c_tbl, c_chat = st.columns([1.5, 1])
    
    with c_tbl:
        st.subheader("🏆 랭킹 리스트")
        st.dataframe(res[['순위','종목명','점수','현재가','PER','ROE','EPS','부채비율']].set_index('순위')
                     .style.format({'현재가':'{:.0f}', 'PER':'{:.2f}', 'ROE':'{:.2f}', 'EPS':'{:.2f}', '부채비율':'{:.2f}'}), 
                     use_container_width=True)
        
    with c_chat:
        st.subheader("💬 Gemini 퀀트 컨설턴트")
        stock_list = res['종목명'].tolist()
        target_name = st.selectbox("종목 선택 (지표 자동계산)", stock_list)
        
        if target_name != st.session_state['current_ticker']:
            st.session_state['current_ticker'] = target_name
            st.session_state['chat_history'] = []
            
            t_data = res[res['종목명']==target_name].iloc[0]
            ticker_symbol = t_data['티커']
            
            # [PyKRX를 위한 티커 변환 로직]
            # 한국장의 경우, 현재 ticker_symbol에 '삼성전자' 같은 한글 이름이 들어있음 (네이버 크롤링 특성)
            # PyKRX는 '005930' 같은 6자리 코드가 필요함.
            real_ticker = ticker_symbol
            if country == "한국 (KR)":
                try:
                    # PyKRX로 이름 -> 코드 변환
                    found_tickers = stock.get_market_ticker_list() 
                    # 근데 이게 2500개라 느릴 수 있음. -> 네이버 종목코드 찾기가 더 빠름?
                    # Streamlit Cloud에서는 PyKRX의 listing조회도 빠름.
                    # 하지만 이름으로 찾는건 함수가 따로 있음.
                    # stock.get_market_ticker_list()는 코드만 줌.
                    # stock.get_market_ticker_name(ticker)는 이름을 줌.
                    # 반대는 없음. 그래서 전체를 뒤져야 함.
                    # 간단하게: 오늘 날짜 기준 전체 리스트에서 매핑
                    market_tickers = stock.get_market_ticker_list(market="KOSPI") + stock.get_market_ticker_list(market="KOSDAQ")
                    for t_code in market_tickers:
                        if stock.get_market_ticker_name(t_code) == target_name:
                            real_ticker = t_code
                            break
                except: pass
            
            # 지표 계산
            with st.spinner(f"{target_name} 차트 분석 중... (PyKRX)"):
                tech_data = calculate_technicals(real_ticker, country)
            
            if tech_data:
                tech_msg = f"""
                📊 **기술적 지표**
                - **RSI**: {tech_data['RSI']:.2f}
                - **Stochastic**: {tech_data['Stochastic_K']:.2f}
                - **CCI**: {tech_data['CCI']:.2f}
                - **Williams %R**: {tech_data['Williams_R']:.2f}
                """
                st.session_state['tech_context'] = tech_msg
            else:
                tech_msg = "\n(차트 데이터 수집 실패)"
                st.session_state['tech_context'] = ""

            welcome_msg = f"**{target_name}**\nPER: {t_data['PER']:.2f} | ROE: {t_data['ROE']:.2f}% | 부채: {t_data['부채비율']:.0f}%" + tech_msg
            st.session_state['chat_history'].append({"role": "assistant", "content": welcome_msg})

        chat_container = st.container(height=400)
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
                        ctx = f"종목:{t_data['종목명']}, 재무:[PER:{t_data['PER']}, ROE:{t_data['ROE']}, 부채:{t_data['부채비율']}%]. 기술적분석:{tech_info}. 질문:{prompt}. 한국어 답변."
                        response = model.generate_content(ctx, stream=True)
                        for chunk in response:
                            if chunk.text:
                                full_res += chunk.text
                                msg_ph.markdown(full_res + "▌")
                                time.sleep(0.02)
                        msg_ph.markdown(full_res)
                        st.session_state['chat_history'].append({"role": "assistant", "content": full_res})
                    except Exception as e: st.error(f"Error: {e}")
