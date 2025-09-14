import pandas as pd
import matplotlib.pyplot as plt

# ====================================================
# Function: Plot RSI
# ====================================================
def plot_rsi(data, period: int = 14, source: str = "close"):
    """
    Plots RSI indicator given OHLC dataset.
    """
    # If user passed file path, read it
    if isinstance(data, str):
        if data.endswith(".csv"):
            df = pd.read_csv(data)
        elif data.endswith(".xlsx"):
            df = pd.read_excel(data)
        else:
            raise ValueError("Unsupported file type. Use CSV or Excel.")
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("Input must be file path or Pandas DataFrame.")

    # Ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # RSI calculation
    delta = df[source].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Plot Closing Price & RSI
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    
    # Price plot
    ax1.plot(df["timestamp"], df[source], label="Closing Price", color="blue")
    ax1.set_title("Closing Price")
    ax1.set_ylabel("Price")
    ax1.legend()

    # RSI plot
    ax2.plot(df["timestamp"], df["RSI"], label="RSI", color="purple")
    ax2.axhline(70, color="red", linestyle="--", linewidth=1)
    ax2.axhline(30, color="green", linestyle="--", linewidth=1)
    ax2.set_title(f"RSI ({period})")
    ax2.set_ylabel("RSI Value")
    ax2.set_xlabel("Timestamp")
    ax2.legend()

    plt.tight_layout()
    plt.show()

    return df

# ====================================================
# Function: Plot Bollinger Bands
# ====================================================
def plot_bollinger_bands(data, period: int = 20, std_factor: int = 2, source: str = "close"):
    """
    Plots Bollinger Bands given OHLC dataset.
    """
    # If user passed file path, read it
    if isinstance(data, str):
        if data.endswith(".csv"):
            df = pd.read_csv(data)
        elif data.endswith(".xlsx"):
            df = pd.read_excel(data)
        else:
            raise ValueError("Unsupported file type. Use CSV or Excel.")
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("Input must be file path or Pandas DataFrame.")

    # Ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Bollinger Bands calculation
    df["SMA"] = df[source].rolling(window=period).mean()
    df["STD"] = df[source].rolling(window=period).std()
    df["UpperBB"] = df["SMA"] + (std_factor * df["STD"])
    df["LowerBB"] = df["SMA"] - (std_factor * df["STD"])

    # Plot Closing Price with Bollinger Bands
    plt.figure(figsize=(12, 6))
    plt.plot(df["timestamp"], df[source], label="Closing Price", color="blue")
    plt.plot(df["timestamp"], df["SMA"], label=f"SMA {period}", color="orange")
    plt.plot(df["timestamp"], df["UpperBB"], label="Upper Band", color="green", linestyle="--")
    plt.plot(df["timestamp"], df["LowerBB"], label="Lower Band", color="red", linestyle="--")

    plt.fill_between(df["timestamp"], df["UpperBB"], df["LowerBB"], color="grey", alpha=0.1)

    plt.title(f"Bollinger Bands ({period}, {std_factor}σ)")
    plt.xlabel("Timestamp")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return df

# ====================================================
# MAIN TESTING SECTION
# ====================================================
if __name__ == "__main__":
    # --- Example 1: Using Fake Data (always works) ---
    df = pd.DataFrame({
        "timestamp": pd.date_range(start="2023-01-01", periods=50, freq="D"),
        "open": [i+1 for i in range(50)],
        "high": [i+2 for i in range(50)],
        "low": [i for i in range(50)],
        "close": [i+1.5 for i in range(50)],
    })

    print("Plotting RSI with fake data...")
    plot_rsi(df)

    print("Plotting Bollinger Bands with fake data...")
    plot_bollinger_bands(df)

    # --- Example 2: Using CSV or Excel (uncomment when you have real file) ---
    # plot_rsi("synthetic_stock_data.xlsx")
    #plot_bollinger_bands("synthetic_stock_data.xlsx")
