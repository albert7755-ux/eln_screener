import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import itertools
from datetime import datetime

# --- 1. 基礎設定 ---
st.set_page_config(page_title="ELN 評級動能掃描 (V18.0)", layout="wide")
st.title("🎯 ELN 結構型商品 - 評級變動偵測版")
st.markdown("""
除了基本面與波動率，本版本加入 **「評級變動 (Rating Action)」** 偵測。
讓您一眼看出最近法人是在 **調升 (Upgrade)** 還是 **調降 (Downgrade)** 該標的。
""")
st.divider()

# --- 2. 側邊欄設定 ---
st.sidebar.header("1️⃣ 標的池")
# 故意加入一些近期可能有變動的股票
default_pool = "NVDA, TSLA, AAPL, MSFT, GOOG, AMD, MSTR, COIN, JPM, KO, INTC, AMZN, NFLX"
tickers_input = st.sidebar.text_area("股票代碼", value=default_pool, height=100)

st.sidebar.header("2️⃣ 權重設定")
w_vol = st.sidebar.slider("波動率 (HV30) 權重", 0.0, 1.0, 0.4)
w_fund = st.sidebar.slider("財報權重", 0.0, 1.0, 0.2)
w_analyst = st.sidebar.slider("法人權重", 0.0, 1.0, 0.2)
w_trend = st.sidebar.slider("趨勢權重", 0.0, 1.0, 0.2)

basket_size = st.sidebar.selectbox("組籃檔數", [2, 3, 4], index=1)
run_btn = st.sidebar.button("🔍 執行評級變動掃描", type="primary")

# --- 3. 核心函數 ---

def get_latest_rating_change(ticker_obj):
    """
    抓取最近一次的評級變動
    回傳格式: {'text': 字串, 'type': 'up'|'down'|'main'}
    """
    try:
        # 抓取升降評紀錄
        upgrades = ticker_obj.upgrades_downgrades
        
        if upgrades is None or upgrades.empty:
            return {'text': "無近期變動", 'type': 'none'}
            
        # 依照日期排序 (最新的在最下面，或是有些版本是最上面，保險起見 sort_index)
        upgrades = upgrades.sort_index(ascending=False)
        
        # 抓最新的一筆
        latest = upgrades.iloc[0]
        date_str = latest.name.strftime('%Y-%m-%d')
        firm = latest['Firm']
        action = str(latest['Action']).lower() # up, down, main, init, reit
        from_grade = latest['FromGrade']
        to_grade = latest['ToGrade']
        
        # 處理空值 (如果是 Init 或是 Reit 可能沒有 FromGrade)
        if not from_grade: from_grade = "New"
        
        display_text = f"{date_str} [{firm}] {from_grade} -> {to_grade}"
        
        # 判斷方向給顏色
        action_type = 'main'
        if 'up' in action: action_type = 'up'
        elif 'down' in action: action_type = 'down'
        
        return {'text': display_text, 'type': action_type}
        
    except Exception as e:
        return {'text': "-", 'type': 'none'}

def get_hv30_data(ticker):
    data = {'Code': ticker}
    tk = yf.Ticker(ticker)
    
    # --- A. 技術面 & 波動率 (HV30) ---
    try:
        hist = tk.history(period="1y")
        if hist.empty: return None
        
        current_price = hist['Close'].iloc[-1]
        ma200 = hist['Close'].rolling(window=200).mean().iloc[-1]
        if pd.isna(ma200): ma200 = current_price
        
        # HV30 計算
        log_ret = np.log(hist['Close'] / hist['Close'].shift(1))
        hv_val = log_ret.tail(30).std() * np.sqrt(252)
        
        data['Price'] = current_price
        data['Trend'] = '⬆️ 多頭' if current_price > ma200 else '⬇️ 空頭'
        data['Trend_Score'] = 100 if current_price > ma200 else 0
        data['HV30'] = f"{hv_val*100:.1f}%"
        data['Raw_Vol'] = hv_val
        
    except: return None 

    # --- B. 基本面 ---
    try:
        info = tk.info
        rec = info.get('recommendationKey', 'none')
        rating_map = {'strong_buy': 100, 'buy': 80, 'overweight': 70, 'hold': 50, 'underweight': 30, 'sell': 10, 'none': 50}
        data['Analyst_Score'] = rating_map.get(str(rec).lower(), 50)
        
        # 這裡改為顯示「最近變動」
        rating_change = get_latest_rating_change(tk)
        data['Rating_Change_Text'] = rating_change['text']
        data['Rating_Change_Type'] = rating_change['type']
        
        pe = info.get('forwardPE')
        margin = info.get('profitMargins')
        debt = info.get('debtToEquity')
        
        # 紀錄原始數據
        data['Raw_PE'] = f"{pe:.1f}" if pe else "N/A"
        data['Raw_Margin'] = f"{margin*100:.1f}%" if margin else "N/A"
        data['Raw_Debt'] = f"{debt:.1f}%" if debt else "N/A"
        
        fund_score = 0
        if pe and 0 < pe < 35: fund_score += 40
        elif pe is None: fund_score += 20
        if margin and margin > 0.15: fund_score += 30
        elif margin is None: fund_score += 15
        if debt and debt < 100: fund_score += 30
        elif debt is None: fund_score += 15
        data['Fund_Score'] = fund_score
    except:
        data['Rating_Change_Text'] = "-"; data['Rating_Change_Type'] = 'none'
        data['Raw_PE'] = "-"; data['Raw_Margin'] = "-"; data['Raw_Debt'] = "-"
        data['Fund_Score'] = 50; data['Analyst_Score'] = 50

    # --- C. 總分 ---
    vol_score_calc = min(data['Raw_Vol'] * 100, 100)
    final_score = (vol_score_calc * w_vol) + (data['Fund_Score'] * w_fund) + (data['Analyst_Score'] * w_analyst) + (data['Trend_Score'] * w_trend)
    data['Total_Score'] = round(final_score, 1)
    
    return data

# --- 4. 執行與顯示 ---

if run_btn:
    ticker_list = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    results = []
    
    with st.spinner("正在掃描法人評級變動紀錄..."):
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
        
        st.subheader("📋 標的評級動能分析")
        st.caption("評級變動欄位：🟢=升評(利多) / 🔴=降評(利空) / ⚪=重申或無變動")
        
        # 設定顯示欄位
        cols = [
            'Code', 'Total_Score', 'Price', 'Trend', 
            'HV30', 
            'Rating_Change_Text', # 這是重點欄位
            'Raw_PE', 'Raw_Margin', 'Raw_Debt'
        ]
        
        rename_map = {
            'Code': '代碼', 'Total_Score': '總分', 'Price': '股價',
            'Trend': '趨勢', 'HV30': 'HV30',
            'Rating_Change_Text': '最近一次評級變動 (機構/方向)',
            'Raw_PE': '本益比', 'Raw_Margin': '淨利率', 'Raw_Debt': '負債比'
        }
        
        display_df = df[cols].rename(columns=rename_map)
        
        # 1. 總分顏色
        def highlight_score(val):
            color = '#d4edda' if val >= 75 else '#fff3cd' if val >= 50 else '#f8d7da'
            return f'background-color: {color}'
        
        # 2. 評級變動顏色 (核心功能)
        def highlight_rating_change(s):
            # s 是一個 Series (整欄資料)
            # 我們需要對照原始 df 的 'Rating_Change_Type' 來決定顏色
            # 因為 display_df 的 index 和 df 是對應的，所以可以直接用 index
            colors = []
            for idx in s.index:
                change_type = df.loc[idx, 'Rating_Change_Type']
                if change_type == 'up':
                    colors.append('background-color: #d4edda; color: #155724; font-weight: bold') # 綠底深綠字
                elif change_type == 'down':
                    colors.append('background-color: #f8d7da; color: #721c24; font-weight: bold') # 紅底深紅字
                else:
                    colors.append('') # 無變色
            return colors

        # 應用樣式
        styled_df = display_df.style\
            .applymap(highlight_score, subset=['總分'])\
            .apply(highlight_rating_change, subset=['最近一次評級變動 (機構/方向)'])\
            .format({'股價': "{:.2f}", '總分': "{:.1f}"})

        st.dataframe(styled_df, use_container_width=True)
        
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
                
                # 額外加分：如果籃子裡有最近被「升評」的股票，分數加成
                upgrade_bonus = 0
                for t in stocks['Rating_Change_Type']:
                    if t == 'up': upgrade_bonus += 5
                
                final_basket_score = avg_score + upgrade_bonus

                basket_res.append({
                    '組合': " + ".join(tickers),
                    '平均評分': round(final_basket_score, 1),
                    '平均 HV30': f"{avg_vol*100:.1f}%",
                    'bonus': upgrade_bonus
                })
            
            best_baskets = pd.DataFrame(basket_res).sort_values('平均評分', ascending=False).head(3)
            
            for idx, row in best_baskets.iterrows():
                bonus_text = f"(含升評加分 +{row['bonus']})" if row['bonus'] > 0 else ""
                st.success(f"🏅 **推薦組合 {idx+1}**: {row['組合']} (評分: {row['平均評分']} {bonus_text} / HV30: {row['平均 HV30']})")
        else:
            st.warning("標的不足，無法組籃")

else:
    st.info("👈 輸入代碼，點擊執行")
