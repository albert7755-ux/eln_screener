import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import itertools
from datetime import datetime, timedelta

# --- 1. 基礎設定 ---
st.set_page_config(page_title="ELN 鎖定天期掃描 (V16.0)", layout="wide")
st.title("🎯 ELN 結構型商品 - 鎖定 24 天期 IV 掃描")
st.markdown("""
本版本簡化邏輯，直接鎖定市場上 **「最接近 24 天到期」** 的選擇權合約。
不再進行理論合成，直接呈現市場真實報價。
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

# 新增：讓您可以微調目標天數 (預設 24)
target_days_input = st.sidebar.number_input("目標抓取天數 (Days)", min_value=7, max_value=90, value=24)

basket_size = st.sidebar.selectbox("組籃檔數", [2, 3, 4], index=1)
run_btn = st.sidebar.button(f"🔍 搜尋最接近 {target_days_input} 天的合約", type="primary")

# --- 3. 核心函數 ---

def get_atm_iv(ticker_obj, exp_date, current_price):
    """取得指定到期日的 ATM Put IV"""
    try:
        opt = ticker_obj.option_chain(exp_date)
        puts = opt.puts
        if puts.empty: return None
        # 找 ATM
        puts['abs_diff'] = abs(puts['strike'] - current_price)
        row = puts.sort_values('abs_diff').iloc[0]
        return row['impliedVolatility']
    except:
        return None

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

    # --- B. 波動率 (鎖定目標天數) ---
    try:
        iv_val = 0
        iv_source = "N/A"
        try:
            exp_dates = tk.options
            today = datetime.now().date()
            
            # 1. 整理所有到期日與天數差
            dates_info = []
            for d_str in exp_dates:
                d_date = datetime.strptime(d_str, "%Y-%m-%d").date()
                days_diff = (d_date - today).days
                if days_diff > 0: # 只看未來
                    dates_info.append({'date': d_str, 'days': days_diff})
            
            if not dates_info: raise ValueError

            # 2. 找出最接近 target_days_input (例如 24) 的合約
            # 使用 min 函數找絕對值差最小的
            closest_contract = min(dates_info, key=lambda x: abs(x['days'] - target_days_input))
            
            # 3. 抓取該合約的 IV
            iv_val = get_atm_iv(tk, closest_contract['date'], data['Price'])
            
            # 顯示實際抓到的天數
            iv_source = f"Option ({closest_contract['days']}d)"
            
            if iv_val == 0 or iv_val is None: raise ValueError

        except:
            # 降級使用歷史波動率
            log_ret = np.log(hist['Close'] / hist['Close'].shift(1))
            iv_val = log_ret.std() * np.sqrt(252)
            iv_source = "Historical (30D)"
            
        data['IV'] = f"{iv_val*100:.1f}%"
        data['Raw_IV'] = iv_val
        data['IV_Source'] = iv_source 
    except:
        data['IV'] = "N/A"; data['Raw_IV'] = 0; data['IV_Source'] = "Error"

    # --- C. 基本面 ---
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

    # --- D. 總分 ---
    iv_score_calc = min(data['Raw_IV'] * 100, 100)
    final_score = (iv_score_calc * w_iv) + (data['Fund_Score'] * w_fund) + (data['Analyst_Score'] * w_analyst) + (data['Trend_Score'] * w_trend)
    data['Total_Score'] = round(final_score, 1)
    
    return data

# --- 4. 執行與顯示 ---

if run_btn:
    ticker_list = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    results = []
    
    with st.spinner(f"正在搜尋最接近 {target_days_input} 天的選擇權合約..."):
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
        
        st.subheader(f"📋 標的詳細透視表 (目標: {target_days_input}天期 IV)")
        
        cols = [
            'Code', 'Total_Score', 'Price', 'Trend', 
            'IV', 'IV_Source', # 顯示實際抓到的天數
            'Analyst', 
            'Raw_PE', 'Raw_Margin', 'Raw_Debt'
        ]
        
        rename_map = {
            'Code': '代碼', 'Total_Score': '總分', 'Price': '股價',
            'Trend': '趨勢', 'IV_Source': 'IV 合約天數',
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
