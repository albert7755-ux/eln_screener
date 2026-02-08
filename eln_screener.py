import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import itertools

# --- 1. 基礎設定 ---
st.set_page_config(page_title="ELN 旗艦版 (V27.0 - 評級趨勢強化版)", layout="wide")

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
# V27.0 主程式
# =========================================================

st.title("🎯 ELN 結構型商品 - 旗艦選股系統")

with st.expander("📖 專業投資輔銷 - 指標應用指南", expanded=True):
    st.markdown("""
    ### 🛠️ 核心銷售邏輯
    | 指標名稱 | 輔銷話術建議 |
    | :--- | :--- |
    | **隱含波動 (IV)** | 「這檔標的目前市場熱度高，隱含波動大，現在做 ELN 領的利息比平常多很多！」 |
    | **評級變動趨勢** | 「不只我們看好，華爾街剛把這檔從 **Hold 調升到 Buy**，代表法人正在進場布局。」 |
    | **負債比率** | 「這家公司財務體質極佳，負債比低，就算市場波動，跌破價被接回來的心理壓力也小。」 |
    """)

st.divider()

# --- 3. 側邊欄 ---
st.sidebar.header("1️⃣ 標的池")
default_pool = "NVDA, TSLA, AAPL, MSFT, GOOG, AMD, AVGO, COIN, JPM, KO, MCD, XOM, LLY"
tickers_input = st.sidebar.text_area("股票代碼", value=default_pool, height=100)

st.sidebar.header("2️⃣ 權重設定")
w_vol = st.sidebar.slider("波動率 (IV) 權重", 0.0, 1.0, 0.4)
w_fund = st.sidebar.slider("財報權重", 0.0, 1.0, 0.2)
w_analyst = st.sidebar.slider("法人權重", 0.0, 1.0, 0.2)
w_trend = st.sidebar.slider("趨勢權重", 0.0, 1.0, 0.2)

basket_size = st.sidebar.selectbox("組籃檔數", [2, 3, 4], index=1)
run_btn = st.sidebar.button("🔍 執行智能掃描", type="primary")

# --- 4. 核心函數 ---

def get_latest_rating_change(ticker_obj):
    """
    抓取完整的評級變動路徑：日期 [機構] 舊評級 -> 新評級
    """
    try:
        upgrades = ticker_obj.upgrades_downgrades
        if upgrades is None or upgrades.empty: return {'text': "-", 'type': 'none'}
        
        upgrades = upgrades.sort_index(ascending=False)
        latest = upgrades.iloc[0]
        
        date_str = latest.name.strftime('%m/%d')
        firm = latest['Firm']
        from_grade = latest['FromGrade'] if (latest['FromGrade'] and str(latest['FromGrade']) != 'nan') else "?"
        to_grade = latest['ToGrade']
        action = str(latest['Action']).lower()
        
        # 判定動作類型
        if 'up' in action:
            action_type = 'up'
            arrow = "🟢 ▲"
        elif 'down' in action:
            action_type = 'down'
            arrow = "🔴 ▼"
        else:
            action_type = 'main'
            arrow = "➡️"
            
        return {
            'text': f"{date_str} [{firm}] {from_grade} {arrow} {to_grade}",
            'type': action_type
        }
    except:
        return {'text': "-", 'type': 'none'}

def get_stock_data(ticker):
    data = {'Code': ticker}
    tk = yf.Ticker(ticker)
    
    # --- A. 技術面 & IV ---
    try:
        hist = tk.history(period="1y")
        if hist.empty: return None
        data['History_Series'] = hist['Close']
        current_price = hist['Close'].iloc[-1]
        ma200 = hist['Close'].rolling(window=200).mean().iloc[-1]
        
        iv_val = 0.0
        try:
            expirations = tk.options
            if expirations:
                opt = tk.option_chain(expirations[0])
                calls = opt.calls
                atm_option = calls.iloc[(calls['strike'] - current_price).abs().argsort()[:1]]
                iv_val = atm_option['impliedVolatility'].values[0]
        except:
            log_ret = np.log(hist['Close'] / hist['Close'].shift(1))
            iv_val = log_ret.tail(30).std() * np.sqrt(252)

        data['Price'] = current_price
        data['Trend'] = '⬆️ 多頭' if current_price > (ma200 if not pd.isna(ma200) else 0) else '⬇️ 空頭'
        data['Trend_Score'] = 100 if current_price > (ma200 if not pd.isna(ma200) else 0) else 0
        data['IV'] = f"{iv_val*100:.1f}%"
        data['Raw_Vol'] = iv_val 
    except: return None 

    # --- B. 基本面 & 法人評級 ---
    try:
        info = tk.info
        rec = info.get('recommendationKey', 'none')
        data['Analyst_Score'] = {'strong_buy':100, 'buy':80, 'overweight':70, 'hold':50, 'underweight':30, 'sell':10}.get(rec.lower(), 50)
        
        # 呼叫強化的評等抓取函數
        rating_change = get_latest_rating_change(tk)
        data['Rating_Path'] = rating_change['text']
        data['Rating_Type'] = rating_change['type']
        
        pe = info.get('forwardPE') or info.get('trailingPE')
        total_debt = info.get('totalDebt')
        total_assets = info.get('totalAssets')
        if total_debt is None or total_assets is None:
            try:
                bs = tk.balance_sheet
                total_assets = bs.loc['Total Assets'].iloc[0]
                total_debt = bs.loc['Total Debt'].iloc[0] if 'Total Debt' in bs.index else bs.loc['Long Term Debt'].iloc[0]
            except: pass

        debt_ratio = (total_debt / total_assets * 100) if total_debt and total_assets else None
        
        data['Raw_PE'] = f"{pe:.1f}" if pe else "N/A"
        data['Raw_Debt_Ratio'] = f"{debt_ratio:.1f}%" if debt_ratio is not None else "N/A"
        
        f_score = 0
        if pe and 0 < pe < 35: f_score += 40
        if debt_ratio is not None and debt_ratio < 60: f_score += 60
        data['Fund_Score'] = max(f_score, 20)
    except:
        data['Fund_Score'] = 50; data['Analyst_Score'] = 50
        data['Rating_Path'] = "-"; data['Rating_Type'] = 'none'

    # --- C. 總分 ---
    vol_score_calc = min(data['Raw_Vol'] * 100, 100)
    final_score = (vol_score_calc * w_vol) + (data['Fund_Score'] * w_fund) + (data['Analyst_Score'] * w_analyst) + (data['Trend_Score'] * w_trend)
    data['Total_Score'] = round(final_score, 1)
    return data

def calculate_basket_correlation(tickers, price_data_map):
    try:
        df_list = [price_data_map[t].rename(t) for t in tickers if t in price_data_map]
        if len(df_list) < 2: return 1.0
        price_df = pd.concat(df_list, axis=1).dropna()
        return price_df.corr().where(np.tril(np.ones(price_df.corr().shape), k=-1).astype(bool)).mean().mean()
    except: return 1.0

# --- 5. 顯示結果 ---

if run_btn:
    ticker_list = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    results = []
    price_cache = {} 
    
    with st.spinner("正在解析標的並追蹤法人動向..."):
        progress_bar = st.progress(0)
        for i, ticker in enumerate(ticker_list):
            d = get_stock_data(ticker)
            if d: 
                results.append(d)
                price_cache[ticker] = d['History_Series']
            progress_bar.progress((i + 1) / len(ticker_list))
    
    if results:
        df = pd.DataFrame(results).sort_values('Total_Score', ascending=False).reset_index(drop=True)
        
        st.subheader("📋 標的池深度掃描")
        
        display_cols = ['Code', 'Total_Score', 'Price', 'Trend', 'IV', 'Rating_Path', 'Raw_PE', 'Raw_Debt_Ratio']
        rename_map = {'Code': '代碼', 'Total_Score': '總分', 'Price': '股價', 'Trend': '趨勢', 'IV': '隱含波動', 'Rating_Path': '法人評級變動軌跡', 'Raw_PE': '本益比', 'Raw_Debt_Ratio': '負債比'}

        def highlight_rating(s):
            colors = []
            for idx in s.index:
                r_type = df.loc[idx, 'Rating_Type']
                if r_type == 'up': colors.append('background-color: #e6ffed; color: #1a7f37; font-weight: bold;')
                elif r_type == 'down': colors.append('background-color: #ffeef0; color: #cf222e; font-weight: bold;')
                else: colors.append('')
            return colors

        st.dataframe(
            df[display_cols].rename(columns=rename_map).style
            .apply(highlight_rating, subset=['法人評級變動軌跡'])
            .format({'股價': "{:.2f}", '總分': "{:.1f}"}),
            use_container_width=True
        )
        
        # --- 組籃建議 ---
        st.divider()
        st.subheader("💡 AI 智能組籃建議 (含法人動向權重)")
        
        candidates = df.head(10)
        if len(candidates) >= basket_size:
            combs = list(itertools.combinations(candidates['Code'], basket_size))
            basket_res = []
            for comb in combs:
                subset = candidates[candidates['Code'].isin(comb)]
                avg_score = subset['Total_Score'].mean()
                avg_iv = subset['Raw_Vol'].mean()
                corr_val = calculate_basket_correlation(list(comb), price_cache)
                # 升評加分邏輯
                bonus = sum([10 for t in subset['Rating_Type'] if t == 'up']) 
                ranking_score = avg_score + bonus + (avg_iv * 10) + ((1 - corr_val) * 15)
                basket_res.append({'組合': " + ".join(comb), 'Ranking_Score': ranking_score, '平均評分': avg_score, '平均 IV': avg_iv, '相關係數': corr_val})
            
            best_baskets = pd.DataFrame(basket_res).sort_values('Ranking_Score', ascending=False).head(3)
            for i, row in best_baskets.iterrows():
                st.info(f"**推薦組合 {i+1}: {row['組合']}**")
                cols = st.columns(4)
                cols[0].metric("綜合戰力", f"{row['平均評分']:.1f}")
                cols[1].metric("預估配息能力", f"{row['平均 IV']*100:.1f}%")
                cols[2].metric("分散程度", "🟢 優" if row['相關係數'] < 0.4 else "🟡 中" if row['相關係數'] < 0.7 else "🔴 低")
                cols[3].metric("相關係數", f"{row['相關係數']:.2f}")
    else:
        st.error("查無資料")
else:
    st.info("👈 請輸入代碼並執行")
