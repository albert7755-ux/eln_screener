import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import itertools
from datetime import datetime

# --- 1. 基礎設定 ---
st.set_page_config(page_title="ELN 旗艦版 (V24.0 - IV 驅動)", layout="wide")

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
# V24.0 主程式
# =========================================================

st.title("🎯 ELN 結構型商品 - 旗艦選股系統 (IV 強化版)")

with st.expander("📖 指標升級說明：為什麼要看 IV (隱含波動率)？", expanded=False):
    st.markdown("""
    ### 📊 關鍵指標進化
    | 指標 | 意義 | 對 ELN 的影響 |
    | :--- | :--- | :--- |
    | **HV30** | **過去** 30 天股價實際的波動幅度。 | 反映過去的表現。 |
    | **IV (新)** | **市場預期未來** 的波動幅度（由選擇權價格回推）。 | **配息的核心來源**。IV 越高，銀行報價的 Coupon 通常越優。 |

    **銷售話術建議**：
    - 「目前這檔標的的 IV 處於高位，代表現在進場鎖定配息最划算！」
    - 「雖然過去很穩 (HV 低)，但市場預期接下來有行情 (IV 高)，正是收高額權利金的好時機。」
    """)

st.divider()

# --- 3. 側邊欄 ---
st.sidebar.header("1️⃣ 標的池")
default_pool = "NVDA, TSLA, AAPL, MSFT, GOOG, AMD, AVGO, COIN, JPM, KO, MCD, XOM, LLY"
tickers_input = st.sidebar.text_area("股票代碼", value=default_pool, height=100)

st.sidebar.header("2️⃣ 權重設定")
w_vol = st.sidebar.slider("波動率 (IV優先) 權重", 0.0, 1.0, 0.4)
w_fund = st.sidebar.slider("財報權重", 0.0, 1.0, 0.2)
w_analyst = st.sidebar.slider("法人權重", 0.0, 1.0, 0.2)
w_trend = st.sidebar.slider("趨勢權重", 0.0, 1.0, 0.2)

basket_size = st.sidebar.selectbox("組籃檔數", [2, 3, 4], index=1)
run_btn = st.sidebar.button("🔍 執行智能掃描", type="primary")

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
    
    # --- A. 技術面 & 波動率 (HV + IV) ---
    try:
        hist = tk.history(period="1y")
        if hist.empty: return None
        data['History_Series'] = hist['Close']
        current_price = hist['Close'].iloc[-1]
        ma200 = hist['Close'].rolling(window=200).mean().iloc[-1]
        
        # 計算 HV30
        log_ret = np.log(hist['Close'] / hist['Close'].shift(1))
        hv_val = log_ret.tail(30).std() * np.sqrt(252)
        
        # 獲取 IV (隱含波動率) - 抓取最接近現價的選擇權
        iv_val = hv_val # 預設值
        try:
            expirations = tk.options
            if expirations:
                # 抓取最近一個到期日 (通常流動性最好)
                opt = tk.option_chain(expirations[0])
                calls = opt.calls
                # 找最接近 Strike Price 的行使價
                atm_option = calls.iloc[(calls['strike'] - current_price).abs().argsort()[:1]]
                iv_val = atm_option['impliedVolatility'].values[0]
        except:
            pass

        data['Price'] = current_price
        data['Trend'] = '⬆️ 多頭' if current_price > (ma200 if not pd.isna(ma200) else 0) else '⬇️ 空頭'
        data['Trend_Score'] = 100 if current_price > (ma200 if not pd.isna(ma200) else 0) else 0
        data['HV30'] = f"{hv_val*100:.1f}%"
        data['IV'] = f"{iv_val*100:.1f}%"
        data['Raw_Vol'] = iv_val # 用 IV 作為主要的波動率評分基礎
    except: return None 

    # --- B. 基本面 (財報 & 評級) ---
    try:
        info = tk.info
        rec = info.get('recommendationKey', 'none')
        data['Analyst_Score'] = {'strong_buy':100, 'buy':80, 'overweight':70, 'hold':50, 'underweight':30, 'sell':10}.get(rec.lower(), 50)
        
        rating_change = get_latest_rating_change(tk)
        data['Rating_Change_Text'] = rating_change['text']
        data['Rating_Change_Type'] = rating_change['type']
        
        pe = info.get('forwardPE') or info.get('trailingPE')
        margin = info.get('profitMargins')
        
        # 負債比獲取
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
        data['Raw_Margin'] = f"{margin*100:.1f}%" if margin else "N/A"
        data['Raw_Debt_Ratio'] = f"{debt_ratio:.1f}%" if debt_ratio is not None else "N/A"
        
        # 財報綜合評分
        f_score = 0
        if pe and 0 < pe < 35: f_score += 40
        if margin and margin > 0.15: f_score += 30
        if debt_ratio is not None and debt_ratio < 60: f_score += 30
        data['Fund_Score'] = max(f_score, 20)
    except:
        data['Fund_Score'] = 50; data['Analyst_Score'] = 50
        data['Rating_Change_Text'] = "-"; data['Rating_Change_Type'] = 'none'

    # --- C. 總分計算 ---
    vol_score_calc = min(data['Raw_Vol'] * 100, 100)
    final_score = (vol_score_calc * w_vol) + (data['Fund_Score'] * w_fund) + (data['Analyst_Score'] * w_analyst) + (data['Trend_Score'] * w_trend)
    data['Total_Score'] = round(final_score, 1)
    
    return data

def calculate_basket_correlation(tickers, price_data_map):
    try:
        df_list = [price_data_map[t].rename(t) for t in tickers if t in price_data_map]
        if len(df_list) < 2: return 1.0
        price_df = pd.concat(df_list, axis=1).dropna()
        corr_matrix = price_df.corr()
        mask = np.tril(np.ones(corr_matrix.shape), k=-1).astype(bool)
        return corr_matrix.where(mask).mean().mean()
    except: return 1.0

# --- 5. 執行與顯示 ---

if run_btn:
    ticker_list = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    results = []
    price_cache = {} 
    
    with st.spinner("正在抓取即時 IV 與財報數據..."):
        progress_bar = st.progress(0)
        for i, ticker in enumerate(ticker_list):
            d = get_stock_data(ticker)
            if d: 
                results.append(d)
                price_cache[ticker] = d['History_Series']
            progress_bar.progress((i + 1) / len(ticker_list))
    
    if not results:
        st.error("查無資料，請檢查代碼是否正確")
    else:
        df = pd.DataFrame(results).sort_values('Total_Score', ascending=False).reset_index(drop=True)
        
        st.subheader("📋 標的池掃描結果 (依總分排序)")
        
        display_cols = ['Code', 'Total_Score', 'Price', 'Trend', 'IV', 'HV30', 'Rating_Change_Text', 'Raw_PE', 'Raw_Debt_Ratio']
        rename_map = {'Code': '代碼', 'Total_Score': '總分', 'Price': '股價', 'Trend': '趨勢', 'IV': '隱含波動(配息源)', 'HV30': '歷史波動', 'Rating_Change_Text': '評級變動', 'Raw_PE': '本益比', 'Raw_Debt_Ratio': '負債比'}

        def style_rating(val):
            if 'up' in str(val).lower() or 'buy' in str(val).lower(): return 'color: green; font-weight: bold'
            if 'down' in str(val).lower() or 'sell' in str(val).lower(): return 'color: red; font-weight: bold'
            return ''

        st.dataframe(
            df[display_cols].rename(columns=rename_map).style
            .map(style_rating, subset=['評級變動'])
            .format({'股價': "{:.2f}", '總分': "{:.1f}"}),
            use_container_width=True
        )
        
        # --- 組籃建議 ---
        st.divider()
        st.subheader(f"💡 AI 智能組籃建議 ({basket_size}檔一籃子)")
        
        candidates = df.head(10) # 取前10名進行組合
        if len(candidates) >= basket_size:
            combs = list(itertools.combinations(candidates['Code'], basket_size))
            basket_res = []
            
            for comb in combs:
                subset = candidates[candidates['Code'].isin(comb)]
                avg_score = subset['Total_Score'].mean()
                avg_iv = subset['Raw_Vol'].mean()
                corr_val = calculate_basket_correlation(list(comb), price_cache)
                # 評分邏輯：高總分 + 高IV + 低相關性
                ranking_score = avg_score + (avg_iv * 10) + ((1 - corr_val) * 15)
                
                basket_res.append({
                    '組合': " + ".join(comb),
                    'Ranking_Score': ranking_score,
                    '平均評分': avg_score,
                    '平均 IV': avg_iv,
                    '相關係數': corr_val
                })
            
            best_baskets = pd.DataFrame(basket_res).sort_values('Ranking_Score', ascending=False).head(3)
            
            for i, row in best_baskets.iterrows():
                c_val = row['相關係數']
                c_color = "🟢 低" if c_val < 0.4 else "🟡 中" if c_val < 0.7 else "🔴 高"
                
                st.info(f"**推薦組合 {i+1}: {row['組合']}**")
                cols = st.columns(4)
                cols[0].metric("綜合戰力", f"{row['平均評分']:.1f}")
                cols[1].metric("預估配息能力 (IV)", f"{row['平均 IV']*100:.1f}%")
                cols[2].metric("分散風險度", c_color)
                cols[3].metric("相關係數", f"{c_val:.2f}")
        else:
            st.warning("符合條件標的不足，無法組籃。")

else:
    st.info("👈 請在左側輸入美股代碼，並按下「執行智能掃描」")
