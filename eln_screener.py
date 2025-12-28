import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import itertools
from datetime import datetime, timedelta

# --- 1. 基礎設定 ---
st.set_page_config(page_title="ELN 全方位掃描器 (V12.0)", layout="wide")
st.title("🎯 ELN 結構型商品 - 全方位多維度選股")
st.markdown("""
不再只看波動率！本系統加入 **法人評級** 與 **技術趨勢**，幫您避開「高波動但體質差」的陷阱。
* **法人觀點**：參考華爾街分析師建議 (Buy/Hold/Sell)。
* **技術趨勢**：確認股價位於年線之上 (多頭排列)。
""")
st.divider()

# --- 2. 側邊欄：參數設定 ---
st.sidebar.header("1️⃣ 標的池設定")
# 預設加入一些穩健與積極標的對比
default_pool = "NVDA, TSLA, AAPL, MSFT, GOOG, AMD, MSTR, COIN, JPM, KO, MCD, DIS, INTC"
tickers_input = st.sidebar.text_area("輸入觀察名單 (逗號分隔)", value=default_pool, height=100)

st.sidebar.divider()
st.sidebar.header("2️⃣ 評分權重設定")
w_iv = st.sidebar.slider("波動率 (配息) 權重", 0.0, 1.0, 0.4, step=0.1)
w_fund = st.sidebar.slider("財報 (安全) 權重", 0.0, 1.0, 0.2, step=0.1)
w_analyst = st.sidebar.slider("法人 (評級) 權重", 0.0, 1.0, 0.2, step=0.1)
w_trend = st.sidebar.slider("技術 (趨勢) 權重", 0.0, 1.0, 0.2, step=0.1)

st.sidebar.info(f"目前總權重: {w_iv + w_fund + w_analyst + w_trend:.1f} (建議總和為 1.0)")

run_btn = st.sidebar.button("🔍 執行全方位掃描", type="primary")

# --- 3. 核心函數 ---

def calculate_rsi(series, period=14):
    """計算 RSI"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_comprehensive_data(ticker):
    """
    獲取：IV、財報、法人評級、技術指標
    """
    try:
        tk = yf.Ticker(ticker)
        
        # --- A. 取得現價與歷史資料 (技術面) ---
        # 下載 1 年資料算年線
        hist = tk.history(period="1y")
        if hist.empty: return None
        
        current_price = hist['Close'].iloc[-1]
        
        # 計算年線 (MA200)
        ma200 = hist['Close'].rolling(window=200).mean().iloc[-1]
        trend_score = 100 if current_price > ma200 else 0 # 在年線之上給滿分，之下給0分
        
        # 計算 RSI
        rsi_series = calculate_rsi(hist['Close'])
        rsi = rsi_series.iloc[-1] if not pd.isna(rsi_series.iloc[-1]) else 50
        
        # --- B. 取得 IV (波動率) ---
        # 簡易算法：抓取選擇權鏈推算，或是直接用歷史波動率替代 (這裡用歷史波動率 HV30 近似 IV 趨勢，為了加速)
        # 為了精準，我們還是嘗試抓 Option (如果失敗則用 HV)
        iv_display = 0
        try:
            exp_dates = tk.options
            if exp_dates:
                # 找近月合約
                opt = tk.option_chain(exp_dates[0])
                # 找 ATM Put
                puts = opt.puts
                puts['abs_diff'] = abs(puts['strike'] - current_price)
                atm_iv = puts.sort_values('abs_diff').iloc[0]['impliedVolatility']
                iv_display = atm_iv
            else:
                # 無選擇權，改用 30日歷史波動率
                log_ret = np.log(hist['Close'] / hist['Close'].shift(1))
                iv_display = log_ret.std() * np.sqrt(252)
        except:
            iv_display = 0

        # --- C. 取得財報與法人資訊 (基本面) ---
        info = tk.info
        
        # 法人評級分數
        rec_key = info.get('recommendationKey', 'none').lower()
        # 轉換為分數
        rating_map = {'strong_buy': 100, 'buy': 80, 'overweight': 70, 'hold': 50, 'underweight': 30, 'sell': 10, 'none': 50}
        analyst_score = rating_map.get(rec_key, 50)
        
        # 目標價空間
        target_price = info.get('targetMeanPrice', current_price)
        upside = ((target_price - current_price) / current_price) if target_price else 0
        
        # 基本面分數 (PE + Margin + Debt)
        fund_score = 0
        pe = info.get('forwardPE', 100)
        margin = info.get('profitMargins', 0)
        debt = info.get('debtToEquity', 100)
        
        if pe is not None and 0 < pe < 35: fund_score += 40
        if margin is not None and margin > 0.15: fund_score += 30
        if debt is not None and debt < 100: fund_score += 30
        
        # --- D. 綜合計算 ---
        # IV 分數 (上限 80%，超過不加分反而扣分，因為太高代表妖股)
        # 這裡設定一個甜蜜點：30%~60% 是 ELN 最好的區間
        iv_score_calc = min(iv_display * 100, 100)
        
        final_score = (
            (iv_score_calc * w_iv) +
            (fund_score * w_fund) +
            (analyst_score * w_analyst) +
            (trend_score * w_trend)
        )
        
        return {
            'Code': ticker,
            'Price': current_price,
            'MA200': ma200,
            'Trend': '⬆️ 多頭' if current_price > ma200 else '⬇️ 空頭',
            'RSI': round(rsi, 1),
            'Analyst': rec_key.replace('_', ' ').title(),
            'Target_Upside': f"{upside*100:.1f}%",
            'IV': f"{iv_display*100:.1f}%",
            'Fund_Score': fund_score,
            'Total_Score': round(final_score, 1),
            'Raw_IV': iv_display
        }

    except Exception as e:
        return None

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
            
            # --- 顯示結果表格 ---
            st.subheader("📋 全方位健檢報告")
            
            # 使用 Pandas Styler 進行視覺化
            def highlight_trend(val):
                color = '#d4edda' if '多頭' in val else '#f8d7da'
                return f'background-color: {color}'
            
            def highlight_score(val):
                color = '#d4edda' if val >= 70 else '#fff3cd' if val >= 50 else '#f8d7da'
                return f'background-color: {color}'

            # 顯示主要欄位
            display_cols = ['Code', 'Price', 'Trend', 'Analyst', 'Target_Upside', 'IV', 'Fund_Score', 'Total_Score']
            
            st.dataframe(
                df[display_cols].style
                .applymap(highlight_trend, subset=['Trend'])
                .applymap(highlight_score, subset=['Total_Score'])
                .format({'Price': "{:.2f}", 'Fund_Score': "{:.0f}", 'Total_Score': "{:.1f}"}),
                use_container_width=True
            )
            
            # --- 詳細解讀 MSTR vs 其他 ---
            st.divider()
            st.subheader("🧐 標的深度解析")
            
            # 找出 MSTR (如果有的話)
            mstr_row = df[df['Code'] == 'MSTR']
            if not mstr_row.empty:
                mstr_data = mstr_row.iloc[0]
                st.warning(f"""
                **針對 MSTR (MicroStrategy) 的警示：**
                雖然它的 IV 高達 **{mstr_data['IV']}**，配息極佳，但請注意：
                1. **法人評級**：目前為 **{mstr_data['Analyst']}**。
                2. **技術趨勢**：{mstr_data['Trend']} (若為空頭請小心)。
                3. **基本面**：財報分數僅 **{mstr_data['Fund_Score']} 分** (通常因高負債導致分數低)。
                
                **結論**：這是一檔高度投機標的，適合極度積極型客戶，不適合追求長期穩定收息的 ELN 組合。
                """)
            
            # 找出高分且穩健的標的
            top_pick = df.iloc[0]
            st.success(f"""
            **🏆 目前綜合評分最高：{top_pick['Code']} (總分 {top_pick['Total_Score']})**
            * **趨勢**：{top_pick['Trend']} (站穩年線之上)
            * **法人觀點**：{top_pick['Analyst']}
            * **Upside**：分析師認為還有 {top_pick['Target_Upside']} 的上漲空間
            * **波動率**：{top_pick['IV']} (提供不錯的配息來源)
            
            這類標的較適合放入 ELN 籃子中，作為穩定收益的核心。
            """)

else:
    st.info("👈 輸入股票代碼，點擊「執行全方位掃描」。")
