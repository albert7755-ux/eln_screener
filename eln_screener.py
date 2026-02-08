import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import itertools

# --- 1. 基礎設定 ---
st.set_page_config(page_title="ELN 旗艦版 (V27.2 - 完整輔銷版)", layout="wide")

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
# V27.2 主程式
# =========================================================

st.title("🎯 ELN 結構型商品 - 旗艦選股系統")

# --- 🔥 指標說明書 (找回來了！) ---
with st.expander("📖 專業投資輔銷 - 指標應用指南與話術", expanded=True):
    st.markdown("""
    ### 🛠️ 為什麼選這些標的？ (給同仁與客戶的話術)
    | 指標名稱 | 輔銷核心價值 | 銷售話術參考 |
    | :--- | :--- | :--- |
    | **隱含波動 (IV)** | **配息來源** | 「這檔標的目前熱度高，隱含波動大，現在鎖定 ELN 的配息(Coupon)比平常更優渥！」 |
    | **評級變動軌跡** | **法人背書** | 「不只我們看好，華爾街剛把這檔從 **Hold 調升到 Buy**，代表法人正在布局，下檔有撐。」 |
    | **負債比率** | **安全氣囊** | 「我們選的公司財務體質極佳，負債比低。就算股價波動，公司穩健、倒閉風險極低，接回股票也安心。」 |
    | **相關係數** | **組籃優化** | 「這兩檔股票產業不同、連動性低。銀行避險成本低，就能把省下的成本轉化成更高的配息給客戶。」 |
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
    try:
        upgrades = ticker_obj.upgrades_downgrades
        if upgrades is None or upgrades.empty: return {'text': "無近期變動", 'type': 'none'}
        upgrades = upgrades.sort_index(ascending=False)
        latest = upgrades.iloc[0]
        date_str = latest.name.strftime('%m/%d')
        firm = latest['Firm']
        from_grade = str(latest['FromGrade']) if (latest['FromGrade'] and str(latest['FromGrade']) != 'nan') else "?"
        to_grade = str(latest['ToGrade'])
        action = str(latest['Action']).lower()
        if 'up' in action:
            return {'text': f"{date_str} [{firm}] {from_grade} 🟢 ▲ {to_grade}", 'type': 'up'}
        elif 'down' in action:
            return {'text': f"{date_str} [{firm}] {from_grade} 🔴 ▼ {to_grade}", 'type': 'down'}
        return {'text': f"{date_str} [{firm}] {from_grade} ➡️ {to_grade}", 'type': 'main'}
    except:
        return {'text': "資料暫不穩定", 'type': 'none'}

def get_stock_data(ticker):
    # 預先定義所有欄位，防止 KeyError 導致畫面全紅
    data = {
        'Code': ticker, 'Total_Score': 0.0, 'Price': 0.0, 'Trend': '-', 'Trend_Score': 0,
        'IV': 'N/A', 'Raw_Vol': 0.0, 'Rating_Path': 'N/A', 'Rating_Type': 'none',
        'Raw_PE': 'N/A', 'Raw_Debt_Ratio': 'N/A', 'Fund_Score': 50, 'Analyst_Score': 50
    }
    
    tk = yf.Ticker(ticker)
    
    try:
        # A. 技術面與 IV
        hist = tk.history(period="1y")
        if hist.empty: return None
        data['History_Series'] = hist['Close']
        current_price = hist['Close'].iloc[-1]
        ma200 = hist['Close'].rolling(window=200).mean().iloc[-1]
        data['Price'] = current_price
        data['Trend'] = '⬆️ 多頭' if current_price > (ma200 if not pd.isna(ma200) else 0) else '⬇️ 空頭'
        data['Trend_Score'] = 100 if current_price > (ma200 if not pd.isna(ma200) else 0) else 0
        
        # 抓取 IV (如果 yf 暫時抓不到 options，計算 30D HV 作為補充)
        iv_val = 0.0
        try:
            expirations = tk.options
            if expirations:
                opt = tk.option_chain(expirations[0])
                atm_option = opt.calls.iloc[(opt.calls['strike'] - current_price).abs().argsort()[:1]]
                iv_val = atm_option['impliedVolatility'].values[0]
            else: raise ValueError
        except:
            log_ret = np.log(hist['Close'] / hist['Close'].shift(1))
            iv_val = log_ret.tail(30).std() * np.sqrt(252)

        data['IV'] = f"{iv_val*100:.1f}%"
        data['Raw_Vol'] = iv_val 

        # B. 法人評級軌跡 (這部分 yfinance 偶爾會超時，我們盡力抓)
        rating_change = get_latest_rating_change(tk)
        data['Rating_Path'] = rating_change['text']
        data['Rating_Type'] = rating_change['type']
        
        # C. 財務比率 (info 抓取)
        info = tk.info
        rec = info.get('recommendationKey', 'none')
        data['Analyst_Score'] = {'strong_buy':100, 'buy':80, 'overweight':70, 'hold':50}.get(rec.lower(), 50)
        
        pe = info.get('forwardPE') or info.get('trailingPE')
        data['Raw_PE'] = f"{pe:.1f}" if pe else "N/A"
        
        total_debt = info.get('totalDebt')
        total_assets = info.get('totalAssets')
        
        # 如果 info 抓不到財報，嘗試從 balance_sheet 補充
        if not total_debt or not total_assets:
            try:
                bs = tk.balance_sheet
                total_assets = bs.loc['Total Assets'].iloc[0]
                if 'Total Debt' in bs.index: total_debt = bs.loc['Total Debt'].iloc[0]
                elif 'Long Term Debt' in bs.index: total_debt = bs.loc['Long Term Debt'].iloc[0]
            except: pass

        debt_ratio = (total_debt / total_assets * 100) if total_debt and total_assets else None
        data['Raw_Debt_Ratio'] = f"{debt_ratio:.1f}%" if debt_ratio is not None else "N/A"
        
        f_score = 0
        if pe and 0 < pe < 35: f_score += 40
        if debt_ratio is not None and debt_ratio < 60: f_score += 60
        data['Fund_Score'] = max(f_score, 20)

    except Exception:
        pass 

    # 最終總分計算
    vol_score_calc = min(data['Raw_Vol'] * 100, 100)
    data['Total_Score'] = round((vol_score_calc * w_vol) + (data['Fund_Score'] * w_fund) + (data['Analyst_Score'] * w_analyst) + (data['Trend_Score'] * w_trend), 1)
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
    
    with st.spinner("🚀 正在執行深度掃描... 若資料未出現可能是網站更新中"):
        progress_bar = st.progress(0)
        for i, ticker in enumerate(ticker_list):
            d = get_stock_data(ticker)
            if d: 
                results.append(d)
                if 'History_Series' in d: price_cache[ticker] = d['History_Series']
            progress_bar.progress((i + 1) / len(ticker_list))
    
    if results:
        df = pd.DataFrame(results).sort_values('Total_Score', ascending=False).reset_index(drop=True)
        
        st.subheader("📋 標的池深度掃描結果")
        
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
        
        st.divider()
        st.subheader("💡 AI 智能組籃建議 (含分析師加權)")
        
        candidates = df.head(10)
        if len(candidates) >= basket_size:
            combs = list(itertools.combinations(candidates['Code'], basket_size))
            basket_res = []
            for comb in combs:
                subset = candidates[candidates['Code'].isin(comb)]
                avg_score = subset['Total_Score'].mean()
                avg_iv = subset['Raw_Vol'].mean()
                corr_val = calculate_basket_correlation(list(comb), price_cache)
                # 升評加權獎勵
                bonus = sum([10 for t in subset['Rating_Type'] if t == 'up']) 
                ranking_score = avg_score + bonus + (avg_iv * 10) + ((1 - corr_val) * 15)
                basket_res.append({'組合': " + ".join(comb), 'Ranking_Score': ranking_score, '平均評分': avg_score, '平均 IV': avg_iv, '相關係數': corr_val})
            
            best_baskets = pd.DataFrame(basket_res).sort_values('Ranking_Score', ascending=False).head(3)
            for i, row in best_baskets.iterrows():
                st.info(f"**推薦組合 {i+1}: {row['組合']}**")
                cols = st.columns(4)
                cols[0].metric("綜合戰力", f"{row['平均評分']:.1f}")
                cols[1].metric("配息潛力", f"{row['平均 IV']*100:.1f}%")
                cols[2].metric("分散係數", f"{row['相關係數']:.2f}")
                cols[3].metric("推薦度", "⭐⭐⭐" if row['相關係數'] < 0.4 else "⭐⭐")
    else:
        st.error("暫時抓不到數據，請稍後再試或縮小搜尋範圍。")
else:
    st.info("👈 請輸入代碼並點擊執行，系統會即時分析 IV 與法人動向")
