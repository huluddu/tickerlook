import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import google.generativeai as genai
import time
import requests

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="AI 퀀트 V35 (Ranking Fix)", layout="wide")
st.title("🤖 AI 퀀트 스크리너 V35 (Ranking & Parsing Fix)")
st.markdown("데이터 파싱 엔진을 교체하고, **랭킹 로직을 '줄 세우기' 방식**으로 직관화했습니다.")

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

    # 🇰🇷 한국 (파싱 엔진 bs4로 교체)
    else:
        session = get_session()
        sosok = 0 if index == 'KOSPI' else 1
        url_base = f"https://finance.naver.com/sise/sise_market_sum.naver?sosok={sosok}&page="
        
        all_dfs = []
        bar = st.progress(0, text="🇰🇷 네이버 증권 데이터 스캔 중...")
        
        for page in range(1, 5): 
            try:
                res_html = session.get(url_base + str(page))
                # [핵심 수정] flavor='bs4' 사용 (html5lib 엔진 가동)
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
        
        # === 🏆 랭킹 로직 완전 분해 (절대값 순위) ===
        
        # 1. PER 점수 (낮을수록 좋음, 0 이하는 0점)
        # 0보다 큰 애들만 뽑음
        valid_per = df[df['PER'] > 0].copy()
        if not valid_per.empty:
            # Rank(ascending=False) -> 값이 크면 1등(High Rank). 작으면 N등(Low Rank).
            # 우리가 원하는 것: 작으면 고득점(High Rank).
            # Rank(ascending=False): PER 100 -> Rank 1. PER 5 -> Rank 100.
            # 이 Rank 그대로 쓰면 PER 100이 점수 먹음.
            # -> Rank(ascending=False)를 쓰면 PER 높은게 1등.
            # -> 100 - Score 하면 안됨?
            # 아니다. Rank(ascending=False)로 하면 큰 값이 상위 랭크(1, 2, 3...). 
            # 점수 = Rank. 그러니까 PER 클수록 점수가 큼. -> 틀림.
            
            # [수정] Rank(ascending=False): 큰 값이 1등(숫자 작음).
            # 아 헷갈리니 min-max 정규화로 갑니다.
            # 점수 = (Max - 내값) / (Max - Min) * 100. (내값이 작을수록 100에 가까움)
            max_p = valid_per['PER'].max()
            min_p = valid_per['PER'].min()
            # 분모가 0이면 모두 100점
            denom = (max_p - min_p) if max_p != min_p else 1
            
            # 공식: 내 PER가 작을수록 점수가 커야 함.
            # Score = (Max_PER - My_PER) / Denom
            df.loc[df['PER'] > 0, 'S_PER'] = (max_p - df['PER']) / denom
            df.loc[df['PER'] <= 0, 'S_PER'] = 0 # 적자는 0점
        else:
            df['S_PER'] = 0

        # 2. ROE 점수 (높을수록 좋음)
        # Score = (My_ROE - Min) / (Max - Min)
        max_r = df['ROE'].max()
        min_r = df['ROE'].min()
        denom = (max_r - min_r) if max_r != min_r else 1
        df['S_ROE'] = (df['ROE'] - min_r) / denom

        # 3. EPS 점수 (높을수록 좋음)
        max_e = df['EPS'].max()
        min_e = df['EPS'].min()
        denom = (max_e - min_e) if max_e != min_e else 1
        df['S_EPS'] = (df['EPS'] - min_e) / denom
        
        # 4. 부채비율 점수 (낮을수록 좋음)
        # Score = (Max_Debt - My_Debt) / (Max - Min)
        max_d = df['부채비율'].max()
        min_d = df['부채비율'].min()
        denom = (max_d - min_d) if max_d != min_d else 1
        df['S_Debt'] = (max_d - df['부채비율']) / denom
        
        # 5. 가중치 적용 및 최종 점수 (0~100)
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
        target_name = st.selectbox("분석할 종목 선택", stock_list)
        
        if target_name != st.session_state['current_ticker']:
            st.session_state['current_ticker'] = target_name
            st.session_state['chat_history'] = []
            t_data = res[res['종목명']==target_name].iloc[0]
            welcome_msg = f"**{target_name}**\nPER: {t_data['PER']:.2f} | ROE: {t_data['ROE']:.2f}% | 부채: {t_data['부채비율']:.0f}%"
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
                        ctx = f"종목:{t_data['종목명']}, 주가:{t_data['현재가']}, PER:{t_data['PER']}, ROE:{t_data['ROE']}, 부채비율:{t_data['부채비율']}%. 질문:{prompt}. 한국어 답변."
                        response = model.generate_content(ctx, stream=True)
                        for chunk in response:
                            if chunk.text:
                                full_res += chunk.text
                                msg_ph.markdown(full_res + "▌")
                                time.sleep(0.02)
                        msg_ph.markdown(full_res)
                        st.session_state['chat_history'].append({"role": "assistant", "content": full_res})
                    except Exception as e: st.error(f"Error: {e}")
