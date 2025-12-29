import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import itertools

# --- 1. 基礎設定 ---
st.set_page_config(page_title="ELN 歷史波動率掃描 (V17.0)", layout="wide")
st.title("🎯 ELN 結構型商品 - HV30 歷史波動率掃描")
st.markdown("""
本版本依據 **過去 30 天的真實股價波動 (HV30)** 進行篩選。
優點：**數據百分之百存在**、計算速度快、能穩定反映該標的近期的活潑程度。
""")
st.divider()

# --- 2. 側邊欄設定 ---
st.sidebar.header("1️⃣ 標的池")
default_pool = "NVDA, TSLA, AAPL, MSFT, GOOG, AMD, MSTR, COIN, JPM, KO, INTC, AMZN, NFLX"
tickers_input = st.sidebar.text_area("股票代碼", value=default_pool, height=100)

st.sidebar.header("2️⃣ 權重設定")
w_vol = st.sidebar.slider("波動率 (HV30) 權重", 0.0, 1.0, 0.4)
w_fund = st.sidebar.slider("財報權重", 0.0, 1.0, 0.2)
w_analyst = st.sidebar.slider("法人權重", 0.0, 1.0, 0.2)
w_trend = st.sidebar.slider("趨勢權重", 0.0, 1.0, 0.2)

basket_size = st.sidebar.selectbox("組籃檔數", [2, 3, 4], index=1)
run_btn = st.sidebar.button("🔍 執行 HV30 掃描", type="primary")

# --- 3. 核心函數 ---

def get_hv30_data(ticker):
    data = {'Code': ticker}
    tk = yf.Ticker(ticker)
    
    # --- A. 技術面 & 波動率 (HV30) ---
    try:
        # 下載過去 1 年資料 (計算年線用)，但波動率只取最近 30 天
        hist = tk.history(period="1y")
        if hist.empty: return None
        
        current_price = hist['Close'].iloc[-1]
        
        # 1. 計算年線趨勢
        ma200 = hist['Close'].rolling(window=200).mean().iloc[-1]
        if pd.isna(ma200): ma200 = current_price
        
        # 2. 計算 HV30 (關鍵修正)
        # 對數報酬率
        log_ret = np.log(hist['Close'] / hist['Close'].shift(1))
        # 取最後 30 個交易日
        last_30_ret = log_ret.tail(30)
        # 計算標準差並年化 (x 根號252)
        hv_val = last_30_ret.std() * np.sqrt(252)
        
        data['Price'] = current_price
        data['Trend'] = '⬆️ 多頭' if current_price > ma200 else '⬇️ 空頭'
        data['Trend_Score'] = 100 if current_price > ma200 else 0
        
        data['HV30'] = f"{hv_val*100:.1f}%"
        data['Raw_Vol'] = hv_val
        
    except: 
        return None # 連股價都抓不到，直接跳過

    # --- B. 基本面 ---
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
        
        # 紀錄原始數據
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

    # --- C. 總分 ---
    # 波動率分數：HV 越高越好，但超過 100% 視為滿分
    vol_score_calc = min(data['Raw_Vol'] * 100, 100)
    
    final_score = (
        (vol_score_calc * w_vol) +
        (data['Fund_Score'] * w_fund) +
        (data['Analyst_Score'] * w_analyst) +
        (data['Trend_Score'] * w_trend)
    )
    data['Total_Score'] = round(final_score, 1)
    
    return data

# --- 4. 執行與顯示 ---

if run_btn:
    ticker_list = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    results = []
    
    with st.spinner("正在計算 30日歷史波動率 (HV30)..."):
        progress_bar = st.progress(0)
        for i, ticker in enumerate(ticker_list):
            d = get_hv30_data(ticker)
            if d: results.append(d)
            progress_bar.progress((i + 1) / len(ticker_list))
    
    if not results:
        st.error("查無資料")
    else:
        df = pd.DataFrame(results)
        df = df.sort_values('Total_Score', ascending=False).reset_index(drop=True)
        
        st.subheader("📋 標的 HV30 波動率排行表")
        st.caption("HV30 = 過去 30 個交易日的真實年化波動率，數據絕對穩定。")
        
        cols = [
            'Code', 'Total_Score', 'Price', 'Trend', 
            'HV30', # 改為顯示 HV30
            'Analyst', 
            'Raw_PE', 'Raw_Margin', 'Raw_Debt'
        ]
        
        rename_map = {
            'Code': '代碼', 'Total_Score': '總分', 'Price': '股價',
            'Trend': '趨勢', 'HV30': 'HV30 (波動率)',
            'Analyst': '法人評級',
            'Raw_PE': '本益比', 'Raw_Margin': '淨利率', 'Raw_Debt': '負債比'
        }
        
        display_df = df[cols].rename(columns=rename_map)
        
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
                avg_vol = stocks['Raw_Vol'].mean()
                tickers = stocks['Code'].tolist()
                
                basket_res.append({
                    '組合': " + ".join(tickers),
                    '平均評分': round(avg_score, 1),
                    '平均 HV30': f"{avg_vol*100:.1f}%"
                })
            
            best_baskets = pd.DataFrame(basket_res).sort_values('平均評分', ascending=False).head(3)
            
            for idx, row in best_baskets.iterrows():
                st.success(f"🏅 **推薦組合 {idx+1}**: {row['組合']} (評分: {row['平均評分']} / HV30: {row['平均 HV30']})")
        else:
            st.warning("標的不足，無法組籃")

else:
    st.info("👈 輸入代碼，點擊執行")
