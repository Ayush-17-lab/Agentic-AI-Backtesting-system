import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from openai import OpenAI

# --------------------------
# Utility: Technical Indicators
# --------------------------
def calculate_indicators(df, price_col="Close"):
    df["MA20"] = df[price_col].rolling(20).mean()
    df["BB_up"] = df["MA20"] + 2 * df[price_col].rolling(20).std()
    df["BB_dn"] = df["MA20"] - 2 * df[price_col].rolling(20).std()

    delta = df[price_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df[price_col].ewm(span=12, adjust=False).mean()
    ema26 = df[price_col].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

    return df

# --------------------------
# Agents
# --------------------------
class DataAgent:
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.date_col = next((c for c in self.data.columns if "date" in c.lower()), None)
        if self.date_col:
            self.data[self.date_col] = pd.to_datetime(self.data[self.date_col], errors="coerce")
            self.data = self.data.dropna(subset=[self.date_col]).sort_values(self.date_col)
        else:
            raise ValueError("No date column found in CSV!")

        self.ticker_col = next((c for c in self.data.columns if "ticker" in c.lower() or "symbol" in c.lower()), None)

    def get_data(self, ticker=None):
        if ticker and self.ticker_col:
            return self.data[self.data[self.ticker_col] == ticker].sort_values(self.date_col)
        return self.data.sort_values(self.date_col)

class SignalAgent:
    def generate_signals(self, df, price_col="Close"):
        df["Signal"] = df[price_col].diff().apply(lambda x: "BUY" if x > 0 else "SELL")
        return df

class TradeAgent:
    def backtest(self, df, price_col="Close"):
        pnl = []
        cum_pnl = 0
        for i in range(1, len(df)):
            if df.iloc[i]["Signal"] == "BUY":
                profit = df.iloc[i][price_col] - df.iloc[i-1][price_col]
                cum_pnl += profit
            pnl.append(cum_pnl)
        df = df.iloc[1:].copy()
        df["PnL"] = pnl
        return df

class SummaryAgent:
    def __init__(self, date_col, ticker_col=None):
        self.date_col = date_col
        self.ticker_col = ticker_col

    def summarize_day(self, df, date):
        row = df[df[self.date_col] == date]
        if row.empty:
            return f"No data for {date.strftime('%Y-%m-%d')}."
        row = row.iloc[0]
        ticker_info = row[self.ticker_col] if self.ticker_col else "the stock"
        return (f"{ticker_info} on {date.strftime('%d %b %Y')}: Close={row['Close']}, "
                f"Signal={row['Signal']}, PnL={row['PnL']:.2f}, RSI={row['RSI']:.2f}, MACD={row['MACD']:.2f}")

    def best_trade(self, df):
        best = df.loc[df["PnL"].idxmax()]
        ticker_info = best[self.ticker_col] if self.ticker_col else "the stock"
        return f"Best trade: {ticker_info} on {best[self.date_col].strftime('%d %b %Y')} with PnL {best['PnL']:.2f}"

class ChatbotAgent:
    def __init__(self, summary_agent, api_key):
        self.summary_agent = summary_agent
        self.client = OpenAI(api_key=api_key)

    def answer(self, query, df):
        query_lower = query.lower()
        if "show me" in query_lower and "data" in query_lower:
            for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d-%m-%y", "%d/%m/%y"):
                try:
                    date = datetime.strptime(query.split()[-1], fmt)
                    return self.summary_agent.summarize_day(df, date)
                except:
                    continue
        if "best" in query_lower or "profit" in query_lower:
            return self.summary_agent.best_trade(df)

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a friendly trading assistant."},
                {"role": "user", "content": query}
            ]
        )
        return response.choices[0].message.content

# --------------------------
# Streamlit App
# --------------------------
def main():
    st.set_page_config(page_title="Agentic AI Trading Dashboard", layout="wide")
    st.title("📈 Agentic AI Trading Dashboard")

    # Sidebar
    st.sidebar.header("⚙️ Settings")
    csv_file = st.sidebar.file_uploader("Upload CSV", type="csv")
    api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

    if csv_file and api_key:
        data_agent = DataAgent(csv_file)
        ticker = None
        if data_agent.ticker_col:
            ticker = st.sidebar.selectbox("Select Ticker", data_agent.data[data_agent.ticker_col].unique())
        df = data_agent.get_data(ticker)
        if df.empty:
            st.warning("No data available.")
            return

        # Signals & Backtest
        df = SignalAgent().generate_signals(df)
        df = TradeAgent().backtest(df)
        df = calculate_indicators(df)

        # Chatbot
        summary_agent = SummaryAgent(data_agent.date_col, data_agent.ticker_col)
        chatbot = ChatbotAgent(summary_agent, api_key)

        # Tabs: Dashboard / Chatbot / Data
        tabs = st.tabs(["📊 Dashboard", "💬 Chatbot", "📄 Data Logs"])
        with tabs[0]:
            # KPI Cards
            total_pnl = df["PnL"].iloc[-1]
            buy_trades = len(df[df["Signal"]=="BUY"])
            sell_trades = len(df[df["Signal"]=="SELL"])
            max_drawdown = df["PnL"].min()
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Total PnL", f"{total_pnl:.2f}")
            kpi2.metric("BUY Trades", buy_trades)
            kpi3.metric("SELL Trades", sell_trades)
            kpi4.metric("Max Drawdown", f"{max_drawdown:.2f}")

            # Interactive Plotly Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df[data_agent.date_col], y=df["Close"], mode="lines", name="Close Price", line=dict(color="blue")))
            fig.add_trace(go.Scatter(x=df[data_agent.date_col], y=df["MA20"], mode="lines", name="MA20", line=dict(color="orange")))
            fig.add_trace(go.Scatter(x=df[data_agent.date_col], y=df["BB_up"], fill=None, mode="lines", name="BB Upper", line=dict(color="gray")))
            fig.add_trace(go.Scatter(x=df[data_agent.date_col], y=df["BB_dn"], fill='tonexty', mode="lines", name="BB Lower", line=dict(color="gray")))
            buy_signals = df[df["Signal"]=="BUY"]
            sell_signals = df[df["Signal"]=="SELL"]
            fig.add_trace(go.Scatter(x=buy_signals[data_agent.date_col], y=buy_signals["Close"], mode="markers", name="BUY", marker=dict(color="green", size=10, symbol="triangle-up")))
            fig.add_trace(go.Scatter(x=sell_signals[data_agent.date_col], y=sell_signals["Close"], mode="markers", name="SELL", marker=dict(color="red", size=10, symbol="triangle-down")))
            fig.update_layout(title="Price & Bollinger Bands", xaxis_title="Date", yaxis_title="Price", height=500)
            st.plotly_chart(fig, use_container_width=True)

            # RSI Chart
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df[data_agent.date_col], y=df["RSI"], mode="lines", name="RSI", line=dict(color="purple")))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
            fig_rsi.update_layout(title="RSI (14)", xaxis_title="Date", yaxis_title="RSI", height=300)
            st.plotly_chart(fig_rsi, use_container_width=True)

            # MACD Chart
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=df[data_agent.date_col], y=df["MACD"], mode="lines", name="MACD", line=dict(color="black")))
            fig_macd.add_trace(go.Scatter(x=df[data_agent.date_col], y=df["Signal_Line"], mode="lines", name="Signal Line", line=dict(color="red")))
            fig_macd.update_layout(title="MACD", xaxis_title="Date", yaxis_title="MACD", height=300)
            st.plotly_chart(fig_macd, use_container_width=True)

        with tabs[1]:
            st.subheader("💬 Chat with Backtester")
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            user_input = st.text_input("Enter your question:")
            if user_input:
                response = chatbot.answer(user_input, df)
                st.session_state.chat_history.append({"user": user_input, "bot": response})
            for chat in st.session_state.chat_history:
                st.markdown(f"**You:** {chat['user']}")
                st.markdown(f"**Bot:** {chat['bot']}")

        with tabs[2]:
            st.subheader("📄 Data Logs")
            st.dataframe(df)

if __name__ == "__main__":
    main()

