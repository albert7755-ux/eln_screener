import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import itertools

# --- 1. 基礎設定 ---
st.set_page_config(page_title="ELN 旗艦版 (V27.3 - 完整解釋版)", layout="wide")

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
# V27.3 主程式
# =========================================================

st.title("🎯 ELN 結構型商品 - 旗艦選股系統")

# --- 🔥 第一版強大的解釋文字 (全數找回) ---
with st.expander("📖 系統使用指南與指標說明 (點擊展開/收合)", expanded=True):
    st.markdown("""
    ### 🛠️ 工具設計邏輯
    本系統專為 **ELN/FCN (股權連結商品)** 設計，協助挑選「高配息且體質穩健」的標的組合。
    透過 **波動率 (配息來源)** 與 **基本面 (安全氣囊)** 的雙重過濾，降低賺了利息賠了價差的風險。

    ---
    
    ### 📊 關鍵指標解讀 (輔銷話術)
    | 指標名稱 | 意義與銷售話術 | 評分標準 |
    | :--- | :--- | :--- |
    | **隱含波動 (IV)** | **配息的來源**。代表市場對未來的預期。數值越高，銀行賣選擇權收到的權利金越高，**客戶拿到的 Coupon 就越好**。 | 越高分越高 (主要權重) |
    | **負債比率** | **安全氣囊**。公式為 `總負債 / 總資產`。數值越低，代表公司欠錢越少，在波動環境下越不容易倒閉。 | < 60% 優；> 80% 扣分 |
    | **最近評級變動** | **跟著大人走**。顯示法人最新的觀點。例如 **🟢 升評**，代表近期有大利多；**🔴 降評** 則需避開。 | Upgraded 加分 |
    | **相關係數** | **組籃優化關鍵**。若兩檔股票連動性低，**避險成本較低，銀行能開出更好的條件**。 | 越低越好 |

    ### 💡 如何使用？
    1. **左側輸入代碼**：輸入美股代碼 (如 NVDA, AAPL)。
    2. **調整權重**：依據客戶屬性 (保守/積極) 調整滑桿。
    3. **執行掃描**：系統將自動推薦最佳的組合，並分析配息潛力。
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
        return {'text': "-", 'type': 'none'}

def get_stock_data(ticker):
    data = {
        'Code': ticker, 'Total_Score': 0.0, 'Price': 0.0, 'Trend': '-', 
        'IV': 'N/A', 'Raw_Vol': 0.0, 'Rating_Path': '-', 'Rating_Type': 'none',
        'Raw_PE': 'N/A', 'Raw_Debt_Ratio': 'N/A', 'Fund_Score': 50, 'Analyst_Score': 50
    }
    tk = yf.Ticker(ticker)
    try:
        hist = tk.history(period="1y")
        if hist.empty: return None
        data['History_Series'] = hist['Close']
        current_price = hist['Close'].iloc[-1]
        ma200 = hist['Close'].rolling(window=200).mean().iloc[-1]
        
        iv_val = 0.0
        try:
            exp = tk.options
            if exp:
                opt = tk.option_chain(exp[0])
                atm = opt.calls.iloc[(opt.calls['strike'] - current_price).abs().argsort()[:1]]
                iv_val = atm['impliedVolatility'].values[0]
        except:
            log_ret = np.log(hist['Close'] / hist['Close'].shift(1))
            iv_val = log_ret.tail(30).std() * np.sqrt(252)

        data['Price'] = current_price
        data['Trend'] = '⬆️ 多頭' if current_price > (ma200 if not pd.isna(ma200) else 0) else '⬇️ 空頭'
        data['Trend_Score'] = 100 if current_price > (ma200 if not pd.isna(ma200) else 0) else 0
        data['IV'] = f"{iv_val*100:.1f}%"
        data['Raw_Vol'] = iv_val 

        rc = get_latest_rating_change(tk)
        data['Rating_Path'] = rc['text']
        data['Rating_Type'] = rc['type']
        
        info = tk.info
        rec = info.get('recommendationKey', 'none')
        data['Analyst_Score'] = {'strong_buy':100, 'buy':80, 'overweight':70, 'hold':50}.get(rec.lower(), 50)
        
        pe = info.get('forwardPE') or info.get('trailingPE')
        data['Raw_PE'] = f"{pe:.1f}" if pe else "N/A"
        
        td, ta = info.get('totalDebt'), info.get('totalAssets')
        if not td or not ta:
            try:
                bs = tk.balance_sheet
                ta = bs.loc['Total Assets'].iloc[0]
                td = bs.loc['Total Debt'].iloc[0] if 'Total Debt' in bs.index else bs.loc['Long Term Debt'].iloc[0]
            except: pass

        dr = (td / ta * 100) if td and ta else None
        data['Raw_Debt_Ratio'] = f"{dr:.1f}%" if dr is not None else "N/A"
        
        f_score = 0
        if pe and 0 < pe < 35: f_score += 40
        if dr and dr < 60: f_score += 60
        data['Fund_Score'] = max(f_score, 20)

    except: pass
    
    vol_calc = min(data['Raw_Vol'] * 100, 100)
    data['Total_Score'] = round((vol_calc * w_vol) + (data['Fund_Score'] * w_fund) + (data['Analyst_Score'] * w_analyst) + (data['Trend_Score'] * w_trend), 1)
    return data

def calculate_corr(tickers, cache):
    try:
        df_list = [cache[t].rename(t) for t in tickers if t in cache]
        if len(df_list) < 2: return 1.0
        return pd.concat(df_list, axis=1).dropna().corr().where(np.tril(np.ones((len(df_list),len(df_list))), k=-1).astype(bool)).mean().mean()
    except: return 1.0

# --- 5. 顯示結果 ---

if run_btn:
    ticker_list = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    results = []
    price_cache = {} 
    
    with st.spinner("正在執行掃描... 找回所有解釋文字與法人動向"):
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

        def highlight_r(s):
            return ['background-color: #e6ffed; font-weight: bold' if df.loc[idx, 'Rating_Type']=='up' else 'background-color: #ffeef0' if df.loc[idx, 'Rating_Type']=='down' else '' for idx in s.index]

        st.dataframe(df[display_cols].rename(columns=rename_map).style.apply(highlight_r, subset=['法人評級變動軌跡']).format({'股價': "{:.2f}", '總分': "{:.1f}"}), use_container_width=True)
        
        st.divider()
        st.subheader("💡 AI 智能組籃建議 (依配息潛力與相關係數優化)")
        
        candidates = df.head(10)
        if len(candidates) >= basket_size:
            combs = list(itertools.combinations(candidates['Code'], basket_size))
            basket_res = []
            for comb in combs:
                subset = candidates[candidates['Code'].isin(comb)]
                avg_iv = subset['Raw_Vol'].mean()
                corr_val = calculate_corr(list(comb), price_cache)
                ranking_score = subset['Total_Score'].mean() + (avg_iv * 10) + ((1 - corr_val) * 15)
                basket_res.append({'組合': " + ".join(comb), 'Ranking_Score': ranking_score, '平均評分': subset['Total_Score'].mean(), '平均 IV': avg_iv, '相關係數': corr_val})
            
            best = pd.DataFrame(basket_res).sort_values('Ranking_Score', ascending=False).head(3)
            for i, row in best.iterrows():
                st.info(f"**推薦組合 {i+1}: {row['組合']}**")
                c1, c2, c3 = st.columns(3)
                c1.metric("綜合戰力", f"{row['平均評分']:.1f}")
                c2.metric("配息潛力 (IV)", f"{row['平均 IV']*100:.1f}%", help="此數值越高，代表該組合在銀行端能談到的配息利率(Coupon)越高。")
                c3.metric("組合相關係數", f"{row['相關係數']:.2f}", delta="優選低連動" if row['相關係數']<0.4 else None)
    else: st.error("查無資料")
else: st.info("👈 請在左側輸入代碼並執行掃描")
