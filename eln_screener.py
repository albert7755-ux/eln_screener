import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import itertools
from datetime import datetime, timedelta

# --- 1. 基礎設定 ---
st.set_page_config(page_title="ELN 全方位掃描器 (V13.0)", layout="wide")
st.title("🎯 ELN 結構型商品 - 智能選股與籃子推薦")
st.markdown("""
結合 **IV 波動率、財報基本面、法人評級、技術趨勢** 四大維度。
並依據您的需求，提供 **指標解釋** 與 **最佳籃子組合建議**。
""")
st.divider()

# --- 2. 側邊欄：參數設定 ---
st.sidebar.header("1️⃣ 標的池設定")
default_pool = "NVDA, TSLA, AAPL, MSFT, GOOG, AMD, MSTR, COIN, JPM, KO, MCD, DIS, INTC, AMZN"
tickers_input = st.sidebar.text_area("輸入觀察名單 (逗號分隔)", value=default_pool, height=100)

st.sidebar.divider()
st.sidebar.header("2️⃣ 評分權重設定")
w_iv = st.sidebar.slider("波動率 (配息) 權重", 0.0, 1.0, 0.4, step=0.1)
w_fund = st.sidebar.slider("財報 (安全) 權重", 0.0, 1.0, 0.2, step=0.1)
w_analyst = st.sidebar.slider("法人 (評級) 權重", 0.0, 1.0, 0.2, step=0.1)
w_trend = st.sidebar.slider("技術 (趨勢) 權重", 0.0, 1.0, 0.2, step=0.1)

st.sidebar.divider()
st.sidebar.header("3️⃣ 籃子組合設定")
basket_size = st.sidebar.selectbox("推薦幾檔湊一籃?", [2, 3, 4], index=1)

run_btn = st.sidebar.button("🔍 執行全方位掃描 & 組籃", type="primary")

# --- 3. 核心函數 ---

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_comprehensive_data(ticker):
    """獲取數據 (含強力容錯)"""
    data = {'Code': ticker}
    tk = yf.Ticker(ticker)
    
    # --- A. 技術面 ---
    try:
        hist = tk.history(period="1y")
        if hist.empty: return None
        
        current_price = hist['Close'].iloc[-1]
        ma200 = hist['Close'].rolling(window=200).mean().iloc[-1]
        if pd.isna(ma200): ma200 = current_price 
        
        trend_score = 100 if current_price > ma200 else 0
        rsi_series = calculate_rsi(hist['Close'])
        rsi = rsi_series.iloc[-1] if not pd.isna(rsi_series.iloc[-1]) else 50
        
        data['Price'] = current_price
        data['Trend'] = '⬆️ 多頭' if current_price > ma200 else '⬇️ 空頭'
        data['Trend_Score'] = trend_score
        
    except: return None

    # --- B. 波動率 (IV) ---
    try:
        iv_val = 0
        use_hv = False
        try:
            exp_dates = tk.options
            if exp_dates:
                opt = tk.option_chain(exp_dates[0])
                puts = opt.puts
                if not puts.empty:
                    puts['abs_diff'] = abs(puts['strike'] - current_price)
                    iv_val = puts.sort_values('abs_diff').iloc[0]['impliedVolatility']
            if iv_val == 0 or iv_val is None: raise ValueError
        except:
            use_hv = True
            log_ret = np.log(hist['Close'] / hist['Close'].shift(1))
            iv_val = log_ret.std() * np.sqrt(252)
            
        data['IV'] = f"{iv_val*100:.1f}%" + (" (HV)" if use_hv else "")
        data['Raw_IV'] = iv_val
    except:
        data['IV'] = "N/A"
        data['Raw_IV'] = 0

    # --- C. 基本面與法人 ---
    try:
        info = tk.info
        rec_key = info.get('recommendationKey', 'none')
        if rec_key is None: rec_key = 'none'
        rec_key = rec_key.lower()
        
        rating_map = {'strong_buy': 100, 'buy': 80, 'overweight': 70, 'hold': 50, 'underweight': 30, 'sell': 10, 'none': 50}
        analyst_score = rating_map.get(rec_key, 50)
        
        data['Analyst'] = rec_key.replace('_', ' ').title()
        
        target_price = info.get('targetMeanPrice')
        if target_price:
            upside = (target_price - current_price) / current_price
            data['Target_Upside'] = f"{upside*100:.1f}%"
        else:
            data['Target_Upside'] = "-"
            
        fund_score = 0
        pe = info.get('forwardPE'); margin = info.get('profitMargins'); debt = info.get('debtToEquity')
        
        if pe and 0 < pe < 35: fund_score += 40
        elif pe is None: fund_score += 20
        if margin and margin > 0.15: fund_score += 30
        elif margin is None: fund_score += 15
        if debt and debt < 100: fund_score += 30
        elif debt is None: fund_score += 15
        
        data['Fund_Score'] = fund_score
        data['Analyst_Score'] = analyst_score
    except:
        data['Analyst'] = "N/A"; data['Target_Upside'] = "-"; data['Fund_Score'] = 50; data['Analyst_Score'] = 50

    # --- D. 綜合計算 ---
    iv_score_calc = min(data['Raw_IV'] * 100, 100)
    final_score = (iv_score_calc * w_iv) + (data['Fund_Score'] * w_fund) + (data['Analyst_Score'] * w_analyst) + (data['Trend_Score'] * w_trend)
    data['Total_Score'] = round(final_score, 1)
    
    return data

# --- 4. 主程式邏輯 ---

if run_btn:
    ticker_list = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    
    if not ticker_list:
        st.warning("請輸入代碼")
    else:
        results = []
        progress_bar = st.progress(0)
        
        for i, ticker in enumerate(ticker_list):
            data = get_comprehensive_data(ticker)
            if data:
                results.append(data)
            progress_bar.progress((i + 1) / len(ticker_list))
            
        if not results:
            st.error("查無資料")
        else:
            df = pd.DataFrame(results)
            df = df.sort_values('Total_Score', ascending=False).reset_index(drop=True)
            
            # --- 第一區：個股掃描結果 ---
            st.subheader("📋 個股全方位健檢報告")
            
            def highlight_trend(val):
                color = '#d4edda' if '多頭' in val else '#f8d7da'
                return f'background-color: {color}'
            
            def highlight_score(val):
                color = '#d4edda' if val >= 70 else '#fff3cd' if val >= 50 else '#f8d7da'
                return f'background-color: {color}'

            display_cols = ['Code', 'Price', 'Trend', 'Analyst', 'Target_Upside', 'IV', 'Fund_Score', 'Total_Score']
            
            st.dataframe(
                df[display_cols].style
                .applymap(highlight_trend, subset=['Trend'])
                .applymap(highlight_score, subset=['Total_Score'])
                .format({'Price': "{:.2f}", 'Fund_Score': "{:.0f}", 'Total_Score': "{:.1f}"}),
                use_container_width=True
            )
            
            # --- 新增功能：指標解釋 (Request 1) ---
            with st.expander("ℹ️ 如何解讀這些指標？ (點擊展開說明)", expanded=True):
                st.markdown("""
                * **Trend (技術趨勢)**：判斷股價是否在 **200日均線 (年線)** 之上。
                    * `⬆️ 多頭`：股價強勢，不易觸及下檔保護 (KI)，較安全。
                    * `⬇️ 空頭`：股價弱勢，接刀風險高，建議避開。
                * **IV (隱含波動率)**：選擇權市場預期的波動程度。
                    * **數值越高**，代表權利金越貴，**ELN 的配息率 (Coupon) 就越好**。
                    * 但若 IV 過高 (如 >80%)，通常代表該公司有重大風險或炒作。
                * **Analyst (法人評級)**：華爾街分析師的共識建議 (Buy/Hold/Sell)。跟著法人走，避免買到地雷。
                * **Target Upside (目標價空間)**：分析師目標價與現價的距離。若為負值，代表股價可能已超漲。
                * **Fund Score (財報分)**：滿分 100。基於本益比、淨利率、負債比計算。分數越高，公司體質越穩健。
                * **Total Score (綜合評分)**：依據您設定的權重計算出的總分，越高代表 CP 值越好。
                """)

            st.divider()
            
            # --- 新增功能：智能組籃 (Request 2) ---
            st.subheader(f"💡 AI 推薦最佳 {basket_size} 檔籃子組合")
            st.caption("從前 8 名高分個股中，排列組合出「綜合評分」最高的組合：")
            
            # 取前 8 名來做排列組合
            top_candidates = df.head(8)
            
            if len(top_candidates) < basket_size:
                st.warning("有效標的不足以組籃，請增加觀察名單。")
            else:
                combs = list(itertools.combinations(top_candidates.index, basket_size))
                basket_results = []
                
                for comb in combs:
                    stocks = top_candidates.loc[list(comb)]
                    
                    # 計算籃子平均數據
                    avg_score = stocks['Total_Score'].mean()
                    avg_iv_raw = stocks['Raw_IV'].mean()
                    tickers = stocks['Code'].tolist()
                    
                    # 籃子評分：綜合總分 (70%) + IV潛力 (30%)
                    # 這樣可以兼顧「體質好」跟「配息不要太差」
                    basket_ranking_score = (avg_score * 0.7) + (avg_iv_raw * 100 * 0.3)
                    
                    basket_results.append({
                        '組合標的': " + ".join(tickers),
                        '平均綜合評分': round(avg_score, 1),
                        '預估平均 IV': f"{avg_iv_raw*100:.1f}%",
                        'ranking_score': basket_ranking_score
                    })
                
                # 排序並顯示前 5 名
                df_basket = pd.DataFrame(basket_results).sort_values('ranking_score', ascending=False).head(5)
                
                for idx, row in df_basket.iterrows():
                    # 用不同顏色標示推薦強度
                    score = row['平均綜合評分']
                    emoji = "🥇" if idx == 0 else "🥈" if idx == 1 else "🥉" if idx == 2 else "🔹"
                    
                    st.success(f"""
                    **{emoji} 推薦組合 #{idx+1}： {row['組合標的']}**
                    * **平均綜合評分**：{score} 分 (體質與趨勢皆優)
                    * **預估平均 IV**：{row['預估平均 IV']} (配息來源)
                    """)

else:
    st.info("👈 輸入股票代碼，點擊「執行全方位掃描」。")
