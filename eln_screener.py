import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import itertools
from datetime import datetime

# --- 1. 基礎設定 (必須放第一行) ---
st.set_page_config(page_title="ELN 旗艦版 (V20.0)", layout="wide")

# --- 2. 密碼保護機制 ---
def check_password():
    if st.session_state.get('password_correct', False):
        return True
    st.header("🔒 請輸入密碼以存取系統")
    password_input = st.text_input("Password", type="password")
    if st.button("登入"):
        try:
            if password_input == st.secrets["PASSWORD"]:
                st.session_state['password_correct'] = True
                st.rerun()
            else:
                st.error("❌ 密碼錯誤")
        except:
            st.error("⚠️ 請先設定 Secrets")
    return False

if not check_password():
    st.stop()

# =========================================================
# V20.0 主程式
# =========================================================

st.title("🎯 ELN 結構型商品 - 旗艦選股 (數據修復版)")
st.markdown("修正部分數據顯示為 `-` 的問題，並優化相關係數顯示介面。")
st.divider()

# --- 3. 側邊欄 ---
st.sidebar.header("1️⃣ 標的池")
default_pool = "NVDA, TSLA, AAPL, MSFT, GOOG, AMD, MSTR, COIN, JPM, KO, MCD, XOM, PFE"
tickers_input = st.sidebar.text_area("股票代碼", value=default_pool, height=100)

st.sidebar.header("2️⃣ 權重設定")
w_vol = st.sidebar.slider("波動率 (HV30) 權重", 0.0, 1.0, 0.4)
w_fund = st.sidebar.slider("財報權重", 0.0, 1.0, 0.2)
w_analyst = st.sidebar.slider("法人權重", 0.0, 1.0, 0.2)
w_trend = st.sidebar.slider("趨勢權重", 0.0, 1.0, 0.2)

basket_size = st.sidebar.selectbox("組籃檔數", [2, 3, 4], index=1)
run_btn = st.sidebar.button("🔍 執行修復掃描", type="primary")

# --- 4. 核心函數 ---

def get_latest_rating_change(ticker_obj):
    try:
        upgrades = ticker_obj.upgrades_downgrades
        if upgrades is None or upgrades.empty: return {'text': "-", 'type': 'none'}
        upgrades = upgrades.sort_index(ascending=False)
        latest = upgrades.iloc[0]
        date_str = latest.name.strftime('%Y-%m-%d')
        firm = latest['Firm']
        action = str(latest['Action']).lower()
        from_grade = latest['FromGrade'] if latest['FromGrade'] else "New"
        to_grade = latest['ToGrade']
        action_type = 'up' if 'up' in action else 'down' if 'down' in action else 'main'
        return {'text': f"{date_str} [{firm}] {from_grade}->{to_grade}", 'type': action_type}
    except: return {'text': "-", 'type': 'none'}

def get_stock_data(ticker):
    data = {'Code': ticker}
    tk = yf.Ticker(ticker)
    
    # --- A. 技術面 & 波動率 ---
    try:
        hist = tk.history(period="1y")
        if hist.empty: return None
        data['History_Series'] = hist['Close']
        current_price = hist['Close'].iloc[-1]
        ma200 = hist['Close'].rolling(window=200).mean().iloc[-1]
        if pd.isna(ma200): ma200 = current_price
        
        log_ret = np.log(hist['Close'] / hist['Close'].shift(1))
        hv_val = log_ret.tail(30).std() * np.sqrt(252)
        
        data['Price'] = current_price
        data['Trend'] = '⬆️ 多頭' if current_price > ma200 else '⬇️ 空頭'
        data['Trend_Score'] = 100 if current_price > ma200 else 0
        data['HV30'] = f"{hv_val*100:.1f}%"
        data['Raw_Vol'] = hv_val
    except: return None 

    # --- B. 基本面 (強力抓取修復) ---
    try:
        info = tk.info
        
        # 1. 評級
        rec = info.get('recommendationKey', None)
        if rec:
            data['Analyst_Score'] = {'strong_buy':100, 'buy':80, 'overweight':70, 'hold':50, 'underweight':30, 'sell':10}.get(rec.lower(), 50)
        else:
            data['Analyst_Score'] = 50 # 預設中性
            
        rating_change = get_latest_rating_change(tk)
        data['Rating_Change_Text'] = rating_change['text']
        data['Rating_Change_Type'] = rating_change['type']
        
        # 2. 財報數據 (增加候補機制)
        # PE: 優先找 Forward, 沒有就找 Trailing
        pe = info.get('forwardPE')
        if pe is None: pe = info.get('trailingPE')
        
        margin = info.get('profitMargins')
        debt = info.get('debtToEquity')

        # 格式化顯示 (若真的沒有，顯示 N/A)
        data['Raw_PE'] = f"{pe:.1f}" if pe else "N/A"
        data['Raw_Margin'] = f"{margin*100:.1f}%" if margin else "N/A"
        data['Raw_Debt'] = f"{debt:.1f}%" if debt else "N/A"
        
        # 評分
        fund_score = 0
        # PE 分數
        if pe and 0 < pe < 35: fund_score += 40
        elif pe is None: fund_score += 20 # 沒資料給一半
        else: fund_score += 0 # 負值或太高不給分
        
        # Margin 分數
        if margin and margin > 0.15: fund_score += 30
        elif margin is None: fund_score += 15
        
        # D/E 分數
        if debt:
            if debt < 100: fund_score += 30
            elif margin and margin > 0.2: fund_score += 20 # 現金牛豁免
        elif debt is None:
            fund_score += 15
            
        data['Fund_Score'] = fund_score
    except:
        data['Rating_Change_Text'] = "-"; data['Rating_Change_Type'] = 'none'
        data['Raw_PE'] = "N/A"; data['Raw_Margin'] = "N/A"; data['Raw_Debt'] = "N/A"
        data['Fund_Score'] = 50; data['Analyst_Score'] = 50

    # --- C. 總分 ---
    vol_score_calc = min(data['Raw_Vol'] * 100, 100)
    final_score = (vol_score_calc * w_vol) + (data['Fund_Score'] * w_fund) + (data['Analyst_Score'] * w_analyst) + (data['Trend_Score'] * w_trend)
    data['Total_Score'] = round(final_score, 1)
    
    return data

def calculate_basket_correlation(tickers, price_data_map):
    try:
        df_list = []
        for t in tickers:
            if t in price_data_map:
                s = price_data_map[t]
                s.name = t
                df_list.append(s)
        if not df_list: return 1.0
        price_df = pd.concat(df_list, axis=1).dropna()
        if price_df.empty: return 1.0
        
        corr_matrix = price_df.corr()
        mask = np.tril(np.ones(corr_matrix.shape), k=-1).astype(bool)
        lower_triangle = corr_matrix.where(mask)
        avg_corr = lower_triangle.mean().mean()
        return 1.0 if pd.isna(avg_corr) else avg_corr
    except: return 1.0

# --- 5. 執行與顯示 ---

if run_btn:
    ticker_list = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    results = []
    price_cache = {} 
    
    with st.spinner("正在掃描與修復數據..."):
        progress_bar = st.progress(0)
        for i, ticker in enumerate(ticker_list):
            d = get_stock_data(ticker)
            if d: 
                results.append(d)
                price_cache[ticker] = d['History_Series']
            progress_bar.progress((i + 1) / len(ticker_list))
    
    if not results:
        st.error("查無資料")
    else:
        df = pd.DataFrame(results)
        df = df.sort_values('Total_Score', ascending=False).reset_index(drop=True)
        
        # --- 個股列表 ---
        st.subheader("📋 個股掃描結果")
        
        rename_map = {
            'Code': '代碼', 'Total_Score': '總分', 'Price': '股價',
            'Trend': '趨勢', 'HV30': 'HV30',
            'Rating_Change_Text': '最近評級變動',
            'Raw_PE': '本益比', 'Raw_Margin': '淨利率', 
            'Raw_Debt': '負債權益比 (D/E)'
        }
        
        display_cols = ['Code', 'Total_Score', 'Price', 'Trend', 'HV30', 'Rating_Change_Text', 'Raw_PE', 'Raw_Margin', 'Raw_Debt']
        
        def highlight_rating_change(s):
            colors = []
            for idx in s.index:
                change_type = df.loc[idx, 'Rating_Change_Type']
                if change_type == 'up': colors.append('background-color: #d4edda; color: #155724; font-weight: bold')
                elif change_type == 'down': colors.append('background-color: #f8d7da; color: #721c24; font-weight: bold')
                else: colors.append('')
            return colors

        st.dataframe(
            df[display_cols].rename(columns=rename_map).style
            .apply(highlight_rating_change, subset=['最近評級變動'])
            .format({'股價': "{:.2f}", '總分': "{:.1f}"}),
            use_container_width=True,
            column_config={
                "最近評級變動": st.column_config.TextColumn(width="medium") # 防止這裡也被截斷
            }
        )
        
        # --- 智能組籃 (介面優化版) ---
        st.divider()
        st.subheader(f"💡 AI 智能組籃 (考量相關係數)")
        
        top_n = 10 
        candidates = df.head(top_n)
        
        if len(candidates) >= basket_size:
            combs = list(itertools.combinations(candidates['Code'], basket_size))
            basket_res = []
            
            for idx, comb in enumerate(combs):
                tickers = list(comb)
                subset = candidates[candidates['Code'].isin(tickers)]
                
                avg_score = subset['Total_Score'].mean()
                avg_vol = subset['Raw_Vol'].mean()
                corr_val = calculate_basket_correlation(tickers, price_cache)
                bonus = sum([5 for t in subset['Rating_Change_Type'] if t == 'up'])
                
                ranking_score = avg_score + bonus + ((1 - corr_val) * 20)
                
                basket_res.append({
                    '組合': " + ".join(tickers),
                    'Ranking_Score': ranking_score,
                    '平均評分': avg_score,
                    '平均 HV30': avg_vol,
                    '平均相關係數': corr_val
                })
            
            best_baskets = pd.DataFrame(basket_res).sort_values('Ranking_Score', ascending=False).head(5)
            
            # 使用 Markdown 與 HTML 來解決 metric 截斷問題
            for i, row in best_baskets.iterrows():
                corr_v = row['平均相關係數']
                
                # 設定顏色與文字
                if corr_v > 0.7: 
                    corr_color = "#f8d7da" # 紅底
                    corr_text = f"🔴 高度連動 ({corr_v:.2f})"
                elif corr_v > 0.4: 
                    corr_color = "#fff3cd" # 黃底
                    corr_text = f"🟡 中度連動 ({corr_v:.2f})"
                else: 
                    corr_color = "#d4edda" # 綠底
                    corr_text = f"🟢 低度連動 ({corr_v:.2f}) ★條件優"

                # 卡片式設計
                st.markdown(f"""
                <div style="
                    border: 1px solid #ddd; 
                    border-radius: 10px; 
                    padding: 15px; 
                    margin-bottom: 10px; 
                    background-color: #f9f9f9;
                ">
                    <h4 style="margin: 0; color: #333;">🏅 推薦組合 {i+1}：{row['組合']}</h4>
                    <div style="display: flex; justify-content: space-between; margin-top: 10px;">
                        <div>
                            <span style="font-size: 0.9em; color: #666;">平均評分 (體質)</span><br>
                            <span style="font-size: 1.2em; font-weight: bold;">{row['平均評分']:.1f}</span>
                        </div>
                        <div>
                            <span style="font-size: 0.9em; color: #666;">平均 HV30 (配息)</span><br>
                            <span style="font-size: 1.2em; font-weight: bold;">{row['平均 HV30']*100:.1f}%</span>
                        </div>
                        <div style="background-color: {corr_color}; padding: 5px 10px; border-radius: 5px;">
                            <span style="font-size: 0.9em; color: #666;">相關係數 (避險)</span><br>
                            <span style="font-size: 1.1em; font-weight: bold;">{corr_text}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.warning("標的不足，無法執行相關係數分析。")

else:
    st.info("👈 請輸入代碼並執行")
