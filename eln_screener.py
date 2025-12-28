import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import itertools
from datetime import datetime, timedelta

# --- 1. 基礎設定 ---
st.set_page_config(page_title="ELN 全方位掃描器 (V12.1)", layout="wide")
st.title("🎯 ELN 結構型商品 - 全方位多維度選股 (強力修復版)")
st.markdown("""
加入 **容錯機制** 與 **歷史波動率替代方案**，確保數據讀取穩定。
* **法人觀點**：參考華爾街分析師建議 (Buy/Hold/Sell)。
* **技術趨勢**：確認股價位於年線之上 (多頭排列)。
""")
st.divider()

# --- 2. 側邊欄：參數設定 ---
st.sidebar.header("1️⃣ 標的池設定")
default_pool = "NVDA, TSLA, AAPL, MSFT, GOOG, AMD, MSTR, COIN, JPM, KO, MCD, DIS, INTC"
tickers_input = st.sidebar.text_area("輸入觀察名單 (逗號分隔)", value=default_pool, height=100)

st.sidebar.divider()
st.sidebar.header("2️⃣ 評分權重設定")
w_iv = st.sidebar.slider("波動率 (配息) 權重", 0.0, 1.0, 0.4, step=0.1)
w_fund = st.sidebar.slider("財報 (安全) 權重", 0.0, 1.0, 0.2, step=0.1)
w_analyst = st.sidebar.slider("法人 (評級) 權重", 0.0, 1.0, 0.2, step=0.1)
w_trend = st.sidebar.slider("技術 (趨勢) 權重", 0.0, 1.0, 0.2, step=0.1)

run_btn = st.sidebar.button("🔍 執行全方位掃描", type="primary")

# --- 3. 核心函數 ---

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_comprehensive_data(ticker):
    """
    獲取數據 (分段處理，避免單一錯誤導致全部失敗)
    """
    data = {'Code': ticker}
    tk = yf.Ticker(ticker)
    
    # --- A. 技術面 (最重要，必須成功) ---
    try:
        # 下載 1 年資料
        hist = tk.history(period="1y")
        if hist.empty: return None # 連股價都沒有，直接跳過
        
        current_price = hist['Close'].iloc[-1]
        ma200 = hist['Close'].rolling(window=200).mean().iloc[-1]
        
        # 處理剛上市或資料不足 200 天的情況
        if pd.isna(ma200): ma200 = current_price 
        
        trend_score = 100 if current_price > ma200 else 0
        
        rsi_series = calculate_rsi(hist['Close'])
        rsi = rsi_series.iloc[-1] if not pd.isna(rsi_series.iloc[-1]) else 50
        
        data['Price'] = current_price
        data['MA200'] = ma200
        data['Trend'] = '⬆️ 多頭' if current_price > ma200 else '⬇️ 空頭'
        data['RSI'] = round(rsi, 1)
        data['Trend_Score'] = trend_score
        
    except Exception as e:
        return None # 技術面失敗視為無效標的

    # --- B. 波動率 (IV) ---
    # 策略：優先抓選擇權 IV，抓不到則用歷史波動率 (HV) 替代
    try:
        iv_val = 0
        use_hv = False
        try:
            # 嘗試抓選擇權
            exp_dates = tk.options
            if exp_dates:
                opt = tk.option_chain(exp_dates[0])
                puts = opt.puts
                if not puts.empty:
                    puts['abs_diff'] = abs(puts['strike'] - current_price)
                    iv_val = puts.sort_values('abs_diff').iloc[0]['impliedVolatility']
            
            if iv_val == 0 or iv_val is None: raise ValueError("No Option Data")
                
        except:
            # 失敗：改算歷史波動率 (HV)
            use_hv = True
            log_ret = np.log(hist['Close'] / hist['Close'].shift(1))
            iv_val = log_ret.std() * np.sqrt(252) # 年化
            
        data['IV'] = f"{iv_val*100:.1f}%" + (" (HV)" if use_hv else "")
        data['Raw_IV'] = iv_val
        
    except:
        data['IV'] = "N/A"
        data['Raw_IV'] = 0

    # --- C. 基本面與法人 (最常失敗，需強力容錯) ---
    try:
        info = tk.info
        
        # 1. 法人評級
        rec_key = info.get('recommendationKey', 'none')
        # 如果是 None，給預設值
        if rec_key is None: rec_key = 'none'
        rec_key = rec_key.lower()
        
        rating_map = {'strong_buy': 100, 'buy': 80, 'overweight': 70, 'hold': 50, 'underweight': 30, 'sell': 10, 'none': 50}
        analyst_score = rating_map.get(rec_key, 50)
        
        data['Analyst'] = rec_key.replace('_', ' ').title()
        
        # 2. 目標價
        target_price = info.get('targetMeanPrice')
        if target_price:
            upside = (target_price - current_price) / current_price
            data['Target_Upside'] = f"{upside*100:.1f}%"
        else:
            data['Target_Upside'] = "-"
            
        # 3. 財報分數
        fund_score = 0
        pe = info.get('forwardPE')
        margin = info.get('profitMargins')
        debt = info.get('debtToEquity')
        
        # 容錯判斷
        if pe and 0 < pe < 35: fund_score += 40
        elif pe is None: fund_score += 20 # 沒資料給一半
        
        if margin and margin > 0.15: fund_score += 30
        elif margin is None: fund_score += 15
        
        if debt and debt < 100: fund_score += 30
        elif debt is None: fund_score += 15
        
        data['Fund_Score'] = fund_score
        data['Analyst_Score'] = analyst_score
        
    except:
        # 萬一 info 全掛，給中性分數讓程式跑下去
        data['Analyst'] = "N/A"
        data['Target_Upside'] = "-"
        data['Fund_Score'] = 50 
        data['Analyst_Score'] = 50

    # --- D. 綜合計算 ---
    iv_score_calc = min(data['Raw_IV'] * 100, 100)
    
    final_score = (
        (iv_score_calc * w_iv) +
        (data['Fund_Score'] * w_fund) +
        (data['Analyst_Score'] * w_analyst) +
        (data['Trend_Score'] * w_trend)
    )
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
            st.error("所有標的皆無法讀取資料，請檢查網絡或 yfinance 版本。")
        else:
            df = pd.DataFrame(results)
            df = df.sort_values('Total_Score', ascending=False).reset_index(drop=True)
            
            # --- 顯示結果表格 ---
            st.subheader("📋 全方位健檢報告")
            
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
            
            # --- 詳細解讀 ---
            st.divider()
            st.subheader("🧐 標的深度解析")
            
            # 檢查 MSTR
            mstr_row = df[df['Code'] == 'MSTR']
            if not mstr_row.empty:
                mstr_data = mstr_row.iloc[0]
                st.warning(f"""
                **針對 MSTR (MicroStrategy) 的檢視：**
                * **波動率 (配息來源)**：{mstr_data['IV']}
                * **趨勢狀態**：{mstr_data['Trend']}
                * **法人看法**：{mstr_data['Analyst']}
                * **綜合評分**：{mstr_data['Total_Score']} 分
                """)
            
            # 推薦最高分
            top_pick = df.iloc[0]
            st.success(f"""
            **🏆 目前綜合評分最高：{top_pick['Code']} (總分 {top_pick['Total_Score']})**
            * **趨勢**：{top_pick['Trend']}
            * **法人觀點**：{top_pick['Analyst']}
            * **波動率**：{top_pick['IV']}
            """)

else:
    st.info("👈 輸入股票代碼，點擊「執行全方位掃描」。")
