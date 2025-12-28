import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import itertools
from datetime import datetime, timedelta

# --- 1. 基礎設定 ---
st.set_page_config(page_title="ELN 智能選股雷達 (V11.0)", layout="wide")
st.title("🎯 ELN 結構型商品 - 智能選股與籃子優化")
st.markdown("""
結合 **選擇權市場數據 (IV)** 與 **基本面財報 (Fundamental)**，幫您篩選出「配息優、體質佳」的 ELN 標的組合。
* **高 IV** = 權利金高 = **配息好**
* **高財報分** = 體質穩健 = **不易跌破 KI**
""")
st.divider()

# --- 2. 側邊欄：參數設定 ---
st.sidebar.header("1️⃣ 標的池設定")
# 預設放入一些熱門 ELN 標的 (科技、半導體、傳產)
default_pool = "NVDA, TSLA, AMD, GOOG, AMZN, MSFT, AAPL, INTC, COIN, MSTR, DIS, KO, JPM"
tickers_input = st.sidebar.text_area("輸入觀察名單 (逗號分隔)", value=default_pool, height=100)

st.sidebar.divider()
st.sidebar.header("2️⃣ 篩選權重")
iv_weight = st.sidebar.slider("IV (配息潛力) 權重", 0.0, 1.0, 0.7, step=0.1)
fund_weight = st.sidebar.slider("財報 (安全性) 權重", 0.0, 1.0, 0.3, step=0.1)

st.sidebar.divider()
st.sidebar.header("3️⃣ 籃子組合設定")
basket_size = st.sidebar.selectbox("推薦幾檔湊一籃?", [2, 3, 4], index=1)

run_btn = st.sidebar.button("🔍 開始掃描與組籃", type="primary")

# --- 3. 核心函數 ---

def get_atm_implied_volatility(ticker):
    """
    計算 ATM (價平) 隱含波動率
    邏輯：抓取約 30 天後到期的選擇權，找最接近現價的 Put Option IV
    """
    try:
        tk = yf.Ticker(ticker)
        
        # 1. 取得選擇權到期日
        expirations = tk.options
        if not expirations:
            return None, 0 # 無選擇權資料
            
        # 2. 找離現在約 30 天的到期日 (最能代表短期波動)
        target_date = None
        min_diff = 999
        today = datetime.now().date()
        
        for exp in expirations:
            exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
            days_diff = (exp_date - today).days
            # 找 20~60 天內的
            if 20 <= days_diff <= 60:
                target_date = exp
                break
        
        # 如果沒找到合適的，就拿第一個 (最近月)
        if target_date is None:
            target_date = expirations[0]
            
        # 3. 取得該日期的選擇權鏈
        opt = tk.option_chain(target_date)
        puts = opt.puts
        
        # 4. 取得現價
        hist = tk.history(period="1d")
        if hist.empty: return None, 0
        current_price = hist['Close'].iloc[-1]
        
        # 5. 找 ATM (履約價最接近現價)
        puts['abs_diff'] = abs(puts['strike'] - current_price)
        atm_row = puts.sort_values('abs_diff').iloc[0]
        
        iv = atm_row['impliedVolatility']
        
        # 若資料異常 (IV=0 或 > 200% 通常是資料錯)，過濾掉
        if iv < 0.01 or iv > 5.0:
            return current_price, None
            
        return current_price, iv

    except Exception as e:
        return None, None

def get_financial_score(ticker):
    """
    抓取財報數據並給予評分 (0-100)
    """
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        
        # 抓取關鍵指標 (若無資料給予中位數或預設值)
        # 1. Forward PE (本益比): 越低越安全 (但太低可能是爛股，這裡簡化為 <30 加分)
        pe = info.get('forwardPE', 50)
        # 2. Profit Margins (淨利率): 越高越好
        margin = info.get('profitMargins', 0.1)
        # 3. Debt to Equity (負債比): 越低越好
        debt_eq = info.get('debtToEquity', 100)
        
        score = 0
        
        # PE 評分 (最高 40 分)
        if pe is not None and 0 < pe < 20: score += 40
        elif 20 <= pe < 40: score += 30
        elif 40 <= pe < 60: score += 10
        else: score += 0 # PE過高或虧損
        
        # Margin 評分 (最高 30 分)
        if margin is not None and margin > 0.2: score += 30
        elif margin > 0.1: score += 20
        elif margin > 0: score += 10
        else: score += 0
        
        # 負債比評分 (最高 30 分)
        if debt_eq is not None and debt_eq < 50: score += 30
        elif debt_eq < 100: score += 20
        elif debt_eq < 200: score += 10
        else: score += 0
        
        raw_data = {
            'PE': round(pe, 1) if pe else 'N/A',
            'Margin': f"{margin*100:.1f}%" if margin else 'N/A',
            'D/E': round(debt_eq, 1) if debt_eq else 'N/A'
        }
        
        return score, raw_data
        
    except:
        return 0, {'PE':'-', 'Margin':'-', 'D/E':'-'}

# --- 4. 主程式邏輯 ---

if run_btn:
    ticker_list = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    
    if not ticker_list:
        st.warning("請輸入代碼")
    else:
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, ticker in enumerate(ticker_list):
            status_text.text(f"正在掃描：{ticker} (計算 IV 與 分析財報)...")
            
            # 1. 取得 IV
            price, iv = get_atm_implied_volatility(ticker)
            
            if price is None or iv is None:
                # 抓不到資料就跳過
                progress_bar.progress((i + 1) / len(ticker_list))
                continue
                
            # 2. 取得財報分數
            fin_score, fin_data = get_financial_score(ticker)
            
            # 3. 綜合評分
            # IV 越高越好 (假設 IV=100% 為滿分)
            iv_score = min(iv * 100, 100) 
            final_score = (iv_score * iv_weight) + (fin_score * fund_weight)
            
            results.append({
                'Code': ticker,
                'Price': price,
                'IV_Annual': iv, # 用於計算
                'IV %': f"{iv*100:.1f}%", # 用於顯示
                'Safety_Score': fin_score,
                'Composite_Score': round(final_score, 1),
                'PE': fin_data['PE'],
                'Margin': fin_data['Margin'],
                'Debt/Eq': fin_data['D/E']
            })
            
            progress_bar.progress((i + 1) / len(ticker_list))
            
        status_text.text("掃描完成！進行數據分析...")
        st.empty() # 清除進度條
        
        if not results:
            st.error("無法取得任何數據，請檢查股票代碼或網絡連線。")
        else:
            df_res = pd.DataFrame(results)
            # 依照綜合分數排序
            df_res = df_res.sort_values('Composite_Score', ascending=False).reset_index(drop=True)
            
            # --- 第一區：個股掃描結果 ---
            st.subheader("📋 個股掃描排行榜 (High IV + High Safety)")
            
            # 格式化顯示
            st.dataframe(
                df_res[['Code', 'Price', 'IV %', 'Safety_Score', 'Composite_Score', 'PE', 'Margin', 'Debt/Eq']].style.background_gradient(subset=['Composite_Score'], cmap='Greens'),
                use_container_width=True
            )
            
            st.info(f"""
            **指標說明：**
            * **IV % (隱含波動率)**：數值越高，代表市場預期波動越大，**ELN 配息率通常越高**。
            * **Safety Score (財報安全分)**：滿分 100。基於本益比、淨利率、負債比計算。分數越高代表公司體質越穩，越不易倒閉或暴跌。
            * **Composite Score (綜合優選分)**：結合 IV 與財報分的加權結果 (權重由左側設定)。
            """)
            
            st.divider()
            
            # --- 第二區：智能組籃 (Basket Optimizer) ---
            st.subheader(f"💡 AI 推薦最佳 {basket_size} 檔籃子組合")
            st.write("系統將從前 6 名高分個股中，找出平均 IV 最高且安全性兼顧的組合：")
            
            # 取前 N 名來做排列組合 (避免計算量過大)
            top_candidates = df_res.head(8) 
            
            if len(top_candidates) < basket_size:
                st.warning("篩選出的有效標的不足以組籃，請增加觀察名單。")
            else:
                combs = list(itertools.combinations(top_candidates.index, basket_size))
                basket_results = []
                
                for comb in combs:
                    # comb 是 index 的 tuple
                    stocks = top_candidates.loc[list(comb)]
                    
                    avg_iv = stocks['IV_Annual'].mean()
                    avg_safety = stocks['Safety_Score'].mean()
                    tickers = stocks['Code'].tolist()
                    
                    # 簡單評分：IV 佔 80% (因為組籃主要為了Yield), 安全佔 20%
                    basket_score = (avg_iv * 100 * 0.8) + (avg_safety * 0.2)
                    
                    basket_results.append({
                        '組合標的': ", ".join(tickers),
                        '預估平均 IV': f"{avg_iv*100:.1f}%",
                        '平均安全分': round(avg_safety, 1),
                        '推薦指數': round(basket_score, 1),
                        'raw_iv': avg_iv
                    })
                
                df_basket = pd.DataFrame(basket_results).sort_values('推薦指數', ascending=False).head(5)
                
                for idx, row in df_basket.iterrows():
                    with st.expander(f"🏆 推薦組合 #{idx+1}： {row['組合標的']}", expanded=(idx==0)):
                        c1, c2, c3 = st.columns(3)
                        c1.metric("預估平均 IV (配息指標)", row['預估平均 IV'])
                        c2.metric("平均安全分", row['平均安全分'])
                        c3.metric("AI 推薦指數", row['推薦指數'])
                        
                        if row['raw_iv'] > 0.6:
                            st.caption("🔥 **極高波動組合**：配息極高，但風險較大，建議設定較低的 KI (如 60% 以下)。")
                        elif row['raw_iv'] > 0.4:
                            st.caption("💰 **高息組合**：適合追求高配息且能承受一定波動的積極型客戶。")
                        else:
                            st.caption("🛡️ **穩健組合**：波動相對溫和，配息適中，適合保守防禦型客戶。")

else:
    st.info("👈 請在左側輸入股票觀察名單，並按下「開始掃描」")
