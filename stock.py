import tkinter as tk
from tkinter import messagebox, Toplevel
import random
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

class StockSimulatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("模擬股票交易系統")
        self.root.geometry("680x800") 

        self.balance = 100000.0
        self.portfolio = {}
        self.market_prices = {}
        
        self.models = {}
        self.recent_history = {}

        self.create_widgets()
        self.update_display()
        self.log_message("🤖 系統啟動成功！輸入股票代碼。")

    def create_widgets(self):
        info_frame = tk.Frame(self.root, pady=10)
        info_frame.pack(fill=tk.X)

        self.lbl_balance = tk.Label(info_frame, text="", font=("Arial", 16, "bold"), fg="blue")
        self.lbl_balance.pack()

        self.lbl_assets = tk.Label(info_frame, text="", font=("Arial", 12))
        self.lbl_assets.pack()

        self.lbl_portfolio = tk.Label(
            info_frame, 
            text="", 
            font=("Microsoft JhengHei", 11), 
            fg="#28a745", 
            justify=tk.LEFT,
            pady=5
        )
        self.lbl_portfolio.pack()

        trade_frame = tk.LabelFrame(self.root, text="交易面板", pady=10, padx=10)
        trade_frame.pack(pady=10)

        tk.Label(trade_frame, text="股票代碼 (如 2330 ):").grid(row=0, column=0, pady=5, sticky=tk.E)
        self.entry_symbol = tk.Entry(trade_frame, font=("Arial", 12))
        self.entry_symbol.grid(row=0, column=1, pady=5)

        tk.Label(trade_frame, text="交易股數:").grid(row=1, column=0, pady=5, sticky=tk.E)
        self.entry_shares = tk.Entry(trade_frame, font=("Arial", 12))
        self.entry_shares.grid(row=1, column=1, pady=5)

        self.btn_frame = tk.Frame(trade_frame)
        self.btn_frame.grid(row=2, column=0, columnspan=2, pady=10)

        tk.Button(self.btn_frame, text="💰 買入", bg="#d4edda", width=10, command=self.buy_stock).pack(side=tk.LEFT, padx=5)
        tk.Button(self.btn_frame, text="📉 賣出", bg="#f8d7da", width=10, command=self.sell_stock).pack(side=tk.LEFT, padx=5)
        tk.Button(self.btn_frame, text="🤖 AI 預測下一天", bg="#cce5ff", width=14, command=self.fluctuate_market).pack(side=tk.LEFT, padx=5)
        tk.Button(self.btn_frame, text="📈 走勢與預測圖", bg="#fff3cd", width=14, command=self.show_price_chart).pack(side=tk.LEFT, padx=5)

        log_frame = tk.Frame(self.root)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        tk.Label(log_frame, text="系統歷史訊息紀錄:").pack(anchor=tk.W)

        self.text_log = tk.Text(log_frame, height=15, state=tk.DISABLED, bg="#f4f4f4", font=("Courier", 10))
        self.text_log.pack(fill=tk.BOTH, expand=True)

    def train_ai_model(self, symbol):
        self.log_message(f"⏳ 正在為 {symbol} 下載歷史數據並訓練 AI 模型...")
        self.root.update()

        try:
            yahoo_symbol = symbol + ".TW" if symbol.isdigit() else symbol
            stock = yf.Ticker(yahoo_symbol)
            df = stock.history(period="2y")
            
            if df.empty: return False

            df['Close_1'] = df['Close'].shift(1)
            df['Close_2'] = df['Close'].shift(2)
            df['Close_3'] = df['Close'].shift(3)
            df.dropna(inplace=True)

            X = df[['Close_1', 'Close_2', 'Close_3']].values
            Y = df['Close'].values

            model = LinearRegression()
            model.fit(X, Y)

            self.models[symbol] = model
            last_3_days = df['Close'].tail(3).values.tolist()
            self.recent_history[symbol] = last_3_days[::-1] 

            self.market_prices[symbol] = round(df['Close'].iloc[-1], 2)
            self.log_message(f"🧠 {symbol} 模型訓練完成！")
            return True

        except Exception as e:
            self.log_message(f"⚠️ {symbol} 模型訓練失敗: {str(e)}")
            return False

    def ai_predict_price(self, symbol):
        if symbol in self.models and symbol in self.recent_history:
            model = self.models[symbol]
            features = np.array([self.recent_history[symbol]]).reshape(1, -1)
            pred_price = model.predict(features)[0]

            noise = random.uniform(-0.02, 0.02)
            final_price = round(pred_price * (1 + noise), 2)

            self.recent_history[symbol].pop()
            self.recent_history[symbol].insert(0, final_price)

            return final_price
        else:
            old_price = self.market_prices[symbol]
            return round(old_price * (1 + random.uniform(-0.05, 0.05)), 2)


    def get_future_30_days(self, symbol):
        if symbol not in self.models or symbol not in self.recent_history:
            return []
        
        model = self.models[symbol]
        temp_history = self.recent_history[symbol].copy()
        future_prices = []
        
        for _ in range(30):
            features = np.array([temp_history]).reshape(1, -1)
            pred = model.predict(features)[0]
            future_prices.append(pred)
            
            temp_history.pop()
            temp_history.insert(0, pred)
            
        return future_prices

    def get_current_price(self, symbol):
        symbol = symbol.upper()
        if symbol not in self.market_prices:
            if not self.train_ai_model(symbol):
                self.market_prices[symbol] = round(random.uniform(10, 1000), 2)
        return self.market_prices[symbol]

    def log_message(self, message):
        self.text_log.config(state=tk.NORMAL)
        self.text_log.insert(tk.END, message + "\n")
        self.text_log.see(tk.END)
        self.text_log.config(state=tk.DISABLED)

    def get_inputs(self):
        symbol = self.entry_symbol.get().strip().upper()
        shares_text = self.entry_shares.get().strip()
        if not symbol:
            messagebox.showwarning("輸入錯誤", "請輸入股票代碼")
            return None, None
        try:
            shares = int(shares_text)
            if shares <= 0: raise ValueError
        except ValueError:
            messagebox.showwarning("輸入錯誤", "股數必須是正整數")
            return None, None
        return symbol, shares

    def buy_stock(self):
        symbol, shares = self.get_inputs()
        if not symbol: return

        price = self.get_current_price(symbol)
        cost = price * shares

        if self.balance >= cost:
            self.balance -= cost
            if symbol in self.portfolio:
                self.portfolio[symbol]['shares'] += shares
                self.portfolio[symbol]['total_cost'] += cost
            else:
                self.portfolio[symbol] = {'shares': shares, 'total_cost': cost}
                
            self.log_message(f"✅ 買入成功: {shares} 股 {symbol} @ ${price:.2f}")
            self.update_display()
        else:
            messagebox.showerror("餘額不足", f"需要 ${cost:.2f}\n目前只有 ${self.balance:.2f}")

    def sell_stock(self):
        symbol, shares = self.get_inputs()
        if not symbol: return

        held = self.portfolio.get(symbol, {}).get('shares', 0)
        if held < shares:
            messagebox.showerror("持股不足", f"目前只有 {held} 股")
            return

        price = self.get_current_price(symbol)
        revenue = price * shares
        avg_cost = self.portfolio[symbol]['total_cost'] / held

        self.balance += revenue
        self.portfolio[symbol]['shares'] -= shares
        self.portfolio[symbol]['total_cost'] -= (avg_cost * shares)

        if self.portfolio[symbol]['shares'] == 0:
            del self.portfolio[symbol]

        self.log_message(f"✅ 賣出成功: {shares} 股 {symbol} @ ${price:.2f}")
        self.update_display()

    def fluctuate_market(self):
        if not self.market_prices:
            self.log_message("ℹ️ 尚未有任何股票交易紀錄")
            return

        self.log_message("-" * 40)
        self.log_message("🔔 AI 模型推算明日股價...")

        for symbol in self.market_prices:
            old_price = self.market_prices[symbol]
            new_price = self.ai_predict_price(symbol)
            self.market_prices[symbol] = new_price

            change_pct = ((new_price - old_price) / old_price) * 100
            sign = "+" if change_pct > 0 else ""
            trend = "📈" if new_price >= old_price else "📉"
            
            self.log_message(f"{trend} {symbol}: ${old_price:.2f} → ${new_price:.2f} ({sign}{change_pct:.2f}%)")

        self.log_message("-" * 40)
        self.update_display()

    def update_display(self):
        self.lbl_balance.config(text=f"💵 可用現金: ${self.balance:,.2f}")
        stock_value = 0
        portfolio_details = []

        for symbol, data in self.portfolio.items():
            shares = data['shares']
            total_cost = data['total_cost']
            avg_cost = total_cost / shares
            
            current_price = self.market_prices.get(symbol, 0)
            current_value = shares * current_price
            stock_value += current_value
            
            profit = current_value - total_cost
            return_rate = (profit / total_cost) * 100
            sign = "+" if return_rate > 0 else ""
            color_icon = "🔺" if return_rate >= 0 else "🔻"

            portfolio_details.append(
                f"{symbol}: {shares:,} 股 | 均價 ${avg_cost:.2f} | 現價 ${current_price:.2f} | 報酬 {color_icon} {sign}{return_rate:.2f}%"
            )

        total_assets = self.balance + stock_value
        self.lbl_assets.config(text=f"💰 總資產估值: ${total_assets:,.2f}")

        if portfolio_details:
            display_text = "📋 目前持股明細：\n" + "\n".join(portfolio_details)
        else:
            display_text = "📋 目前持股明細：無持股"
        
        self.lbl_portfolio.config(text=display_text)


    def show_price_chart(self):
        symbol = self.entry_symbol.get().strip().upper()
        if not symbol:
            messagebox.showwarning("輸入錯誤", "請輸入股票代碼")
            return

        try:
            yahoo_symbol = symbol + ".TW" if symbol.isdigit() else symbol
            stock = yf.Ticker(yahoo_symbol)

            df = stock.history(period="6mo")

            if df.empty:
                messagebox.showerror("查詢失敗", "查無股票資料")
                return

            chart_window = Toplevel(self.root)
            chart_window.title(f"{symbol} 走勢與 AI 未來 30 天預測")
            chart_window.geometry("900x600")

            fig, ax = plt.subplots(figsize=(8, 4))
            
            
            ax.plot(df.index, df["Close"], color="blue", label="歷史收盤價")

            
            future_prices = self.get_future_30_days(symbol)

            if future_prices:
                last_date = df.index[-1]
                last_price = df["Close"].iloc[-1]
                
                future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=30, tz=df.index.tz)
                
                plot_dates = [last_date] + list(future_dates)
                plot_prices = [last_price] + future_prices

                ax.plot(plot_dates, plot_prices, color="magenta", linestyle="--", linewidth=2, label="AI 預測 (未來30天)")

            ax.set_title(f"{symbol} 近半年走勢與未來 AI 預測")
            ax.set_xlabel("日期")
            ax.set_ylabel("價格")
            ax.legend() 
            ax.grid(True)
            plt.tight_layout()

            canvas = FigureCanvasTkAgg(fig, master=chart_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            messagebox.showerror("錯誤", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = StockSimulatorApp(root)
    root.mainloop()