import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import itertools
from datetime import datetime, timedelta

# --- 1. 基礎設定 ---
st.set_page_config(page_title="ELN 全方位掃描 (V14.0 透明版)", layout="wide")
st.title("🎯 ELN 結構型商品 - 數據透明化掃描")
st.markdown("""
本版本特別將 **IV 來源 (選擇權到期日)** 與 **財報原始數據 (PE/Margin/Debt)** 完整列出，
讓您清楚知道分數是如何評估出來的。
""")
st.divider()

# --- 2. 側邊欄設定 ---
st.sidebar.header("1️⃣ 標的池")
default_pool = "NVDA, TSLA, AAPL, MSFT, GOOG, AMD, MSTR, COIN, JPM, KO, INTC"
tickers_input = st.sidebar.text_area("股票代碼", value=default_pool, height=100)

st.sidebar.header("2️⃣ 權重設定")
w_iv = st.sidebar.slider("IV 權重", 0.0, 1.0, 0.4)
w_fund = st.sidebar.slider("財報權重", 0.0, 1.0, 0.2)
w_analyst = st.sidebar.slider("法人權重", 0.0, 1.0, 0.2)
w_trend = st.sidebar.slider("趨勢權重", 0.0, 1.0, 0.2)

basket_size = st.sidebar.selectbox("組籃檔數", [2, 3, 4], index=1)
run_btn = st.sidebar.button("🔍 執行透明化掃描", type="primary")

# --- 3. 核心函數 ---

def get_detailed_data(ticker):
    data = {'Code': ticker}
    tk = yf.Ticker(ticker)
    
    # --- A. 技術面 ---
    try:
        hist = tk.history(period="1y")
        if hist.empty: return None
        current_price = hist['Close'].iloc[-1]
        ma200 = hist['Close'].rolling(window=200).mean().iloc[-1]
        if pd.isna(ma200): ma200 = current_price
        
        data['Price'] = current_price
        data['Trend'] = '⬆️ 多頭' if current_price > ma200 else '⬇️ 空頭'
        data['Trend_Score'] = 100 if current_price > ma200 else 0
    except: return None

    # --- B. 波動率 (顯示來源) ---
    try:
        iv_val = 0
        iv_source = "N/A"
        try:
            exp_dates = tk.options
            if exp_dates:
                # 找第一個到期日 (通常是最近月)
                target_date = exp_dates[0]
                opt = tk.option_chain(target_date)
                puts = opt.puts
                if not puts.empty:
                    # 找 ATM Put
                    puts['abs_diff'] = abs(puts['strike'] - current_price)
                    row = puts.sort_values('abs_diff').iloc[0]
                    iv_val = row['impliedVolatility']
                    # 紀錄來源
                    iv_source = f"Option ({target_date})"
            
            if iv_val == 0 or iv_val is None: raise ValueError
        except:
            # 降級使用歷史波動率
            log_ret = np.log(hist['Close'] / hist['Close'].shift(1))
            iv_val = log_ret.std() * np.sqrt(252)
            iv_source = "Historical (30D)"
            
        data['IV'] = f"{iv_val*100:.1f}%"
        data['Raw_IV'] = iv_val
        data['IV_Source'] = iv_source # 新增欄位
    except:
        data['IV'] = "N/A"; data['Raw_IV'] = 0; data['IV_Source'] = "Error"

    # --- C. 基本面 (顯示原始數據) ---
    try:
        info = tk.info
        
        # 法人
        rec = info.get('recommendationKey', 'none')
        rating_map = {'strong_buy': 100, 'buy': 80, 'overweight': 70, 'hold': 50, 'underweight': 30, 'sell': 10, 'none': 50}
        data['Analyst'] = str(rec).replace('_', ' ').title() if rec else 'None'
        data['Analyst_Score'] = rating_map.get(str(rec).lower(), 50)
        
        # 財報數據
        pe = info.get('forwardPE')
        margin = info.get('profitMargins')
        debt = info.get('debtToEquity')
        
        # 紀錄原始數據供顯示
        data['Raw_PE'] = f"{pe:.1f}" if pe else "N/A"
        data['Raw_Margin'] = f"{margin*100:.1f}%" if margin else "N/A"
        data['Raw_Debt'] = f"{debt:.1f}%" if debt else "N/A"
        
        # 評分邏輯
        fund_score = 0
        if pe and 0 < pe < 35: fund_score += 40
        elif pe is None: fund_score += 20
        if margin and margin > 0.15: fund_score += 30
        elif margin is None: fund_score += 15
        if debt and debt < 100: fund_score += 30
        elif debt is None: fund_score += 15
        
        data['Fund_Score'] = fund_score
    except:
        data['Raw_PE'] = "-"; data['Raw_Margin'] = "-"; data['Raw_Debt'] = "-"
        data['Fund_Score'] = 50; data['Analyst_Score'] = 50; data['Analyst'] = "N/A"

    # --- D. 總分 ---
    iv_score_calc = min(data['Raw_IV'] * 100, 100)
    final_score = (iv_score_calc * w_iv) + (data['Fund_Score'] * w_fund) + (data['Analyst_Score'] * w_analyst) + (data['Trend_Score'] * w_trend)
    data['Total_Score'] = round(final_score, 1)
    
    return data

# --- 4. 執行與顯示 ---

if run_btn:
    ticker_list = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    results = []
    
    with st.spinner("正在解析 IV 來源與財報數據..."):
        progress_bar = st.progress(0)
        for i, ticker in enumerate(ticker_list):
            d = get_detailed_data(ticker)
            if d: results.append(d)
            progress_bar.progress((i + 1) / len(ticker_list))
    
    if not results:
        st.error("查無資料")
    else:
        df = pd.DataFrame(results)
        df = df.sort_values('Total_Score', ascending=False).reset_index(drop=True)
        
        st.subheader("📋 標的詳細透視表")
        st.caption("向右滑動表格可查看詳細財報數字")
        
        # 設定顯示欄位 (加入詳細數據)
        cols = [
            'Code', 'Total_Score', 'Price', 'Trend', 
            'IV', 'IV_Source', # 這裡顯示 IV 來源
            'Analyst', 
            'Raw_PE', 'Raw_Margin', 'Raw_Debt' # 這裡顯示財報細節
        ]
        
        # 欄位重新命名 (中文友善)
        rename_map = {
            'Code': '代碼', 'Total_Score': '總分', 'Price': '股價',
            'Trend': '趨勢', 'IV_Source': 'IV 來源日期',
            'Analyst': '法人評級',
            'Raw_PE': '本益比', 'Raw_Margin': '淨利率', 'Raw_Debt': '負債比'
        }
        
        display_df = df[cols].rename(columns=rename_map)
        
        # 顏色樣式
        def highlight_score(val):
            color = '#d4edda' if val >= 75 else '#fff3cd' if val >= 50 else '#f8d7da'
            return f'background-color: {color}'

        st.dataframe(
            display_df.style
            .applymap(highlight_score, subset=['總分'])
            .format({'股價': "{:.2f}", '總分': "{:.1f}"}),
            use_container_width=True
        )
        
        # --- 智能組籃邏輯 ---
        st.divider()
        st.subheader(f"💡 AI 推薦最佳 {basket_size} 檔籃子")
        
        top_candidates = df.head(8)
        if len(top_candidates) >= basket_size:
            combs = list(itertools.combinations(top_candidates.index, basket_size))
            basket_res = []
            
            for comb in combs:
                stocks = top_candidates.loc[list(comb)]
                avg_score = stocks['Total_Score'].mean()
                avg_iv = stocks['Raw_IV'].mean()
                tickers = stocks['Code'].tolist()
                
                basket_res.append({
                    '組合': " + ".join(tickers),
                    '平均評分': round(avg_score, 1),
                    '平均 IV': f"{avg_iv*100:.1f}%"
                })
            
            best_baskets = pd.DataFrame(basket_res).sort_values('平均評分', ascending=False).head(3)
            
            for idx, row in best_baskets.iterrows():
                st.success(f"🏅 **推薦組合 {idx+1}**: {row['組合']} (評分: {row['平均評分']} / IV: {row['平均 IV']})")
        else:
            st.warning("標的不足，無法組籃")

else:
    st.info("👈 輸入代碼，點擊執行")
