from moomoo import *
import pandas as pd
import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import yfinance as yf
from datetime import datetime

# ==========================================
# [USER INPUT] CHANGE TICKER HERE
# ==========================================
TARGET_STOCK = "TSLA"  # <--- Just type the symbol here (e.g. "NVDA", "AMD", "SPY")

# Automatic Formatting
MOOMOO_SYMBOL = f"US.{TARGET_STOCK}"  # Becomes "US.TSLA"
YAHOO_SYMBOL = TARGET_STOCK  # Becomes "TSLA"

# System Config
HOST = '172.22.48.1'  # Your WSL Bridge IP
PORT = 11111
NUM_EXPIRATIONS = 5
STRIKES_TO_SHOW = 30
RISK_FREE_RATE = 0.045
PLOT_THEME = 'dark_background'

plt.style.use(PLOT_THEME)

# --- COLORS ---
colors = [
    (0.0, '#5e0000'), (0.3, '#ff0000'), (0.5, '#252525'),
    (0.7, '#00d600'), (1.0, '#ffff00')
]
custom_cmap = mcolors.LinearSegmentedColormap.from_list("HighContrastGEX", colors)


def black_scholes_gamma(S, K, T, r, sigma):
    """Local fallback calculation for Gamma."""
    if T <= 0 or sigma <= 0: return 0.0
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        pdf_d1 = si.norm.pdf(d1)
        return pdf_d1 / (S * sigma * np.sqrt(T))
    except:
        return 0.0


def get_spot_price_fallback(symbol):
    """Fetches price from Yahoo using the DYNAMIC symbol."""
    try:
        print(f"--- Fetching Spot Price for {symbol} via Yahoo Finance... ---")
        ticker = yf.Ticker(symbol)

        # Try fast_info first, fall back to history
        try:
            price = ticker.fast_info['last_price']
        except:
            hist = ticker.history(period="1d")
            price = hist['Close'].iloc[-1]

        return price
    except Exception as e:
        print(f"Yahoo Fallback Failed: {e}")
        return None


def get_moomoo_data():
    print(f"--- Connecting to Moomoo OpenD ({HOST}:{PORT}) ---")

    quote_ctx = OpenQuoteContext(host=HOST, port=PORT)

    # 1. GET SPOT PRICE (With Dynamic Fallback)
    spot_price = 0.0
    ret, data = quote_ctx.get_market_snapshot([MOOMOO_SYMBOL])

    # Check if Moomoo gave us a valid price (ret OK and price > 0)
    if ret == RET_OK and data['last_price'][0] > 0:
        spot_price = data['last_price'][0]
        print(f"--- Moomoo Spot Price ({MOOMOO_SYMBOL}): ${spot_price:.2f} ---")
    else:
        print(f"Moomoo Spot Unavailable (Permission Denied).")
        # FALLBACK TO YAHOO using the correct symbol
        spot_price = get_spot_price_fallback(YAHOO_SYMBOL)

        if spot_price:
            print(f"--- Yahoo Spot Price ({YAHOO_SYMBOL}): ${spot_price:.2f} ---")
        else:
            print("CRITICAL: Could not get Spot Price.")
            try:
                spot_price = float(input(f"Please enter current {TARGET_STOCK} price manually: "))
            except:
                quote_ctx.close();
                return None, None

    # 2. GET DATES
    ret, dates_data = quote_ctx.get_option_expiration_date(code=MOOMOO_SYMBOL)
    if ret != RET_OK:
        print(f"Error fetching dates: {dates_data}")
        quote_ctx.close();
        return None, None

    today_str = datetime.now().strftime("%Y-%m-%d")
    valid_dates = [d for d in dates_data['strike_time'].tolist() if d >= today_str]
    valid_dates.sort()
    target_dates = valid_dates[:NUM_EXPIRATIONS]

    print(f"Analyzing Expirations: {target_dates}")
    all_dfs = []

    # 3. LOOP EXPIRIES
    for expiry in target_dates:
        print(f"Fetching {expiry}...", end=" ")

        # Get Chain structure
        ret, chain_info = quote_ctx.get_option_chain(code=MOOMOO_SYMBOL, start=expiry, end=expiry)
        if ret != RET_OK:
            print("Failed (Chain Error).")
            continue

        # Batch request snapshots (Limit 200)
        option_codes = chain_info['code'].tolist()
        chunk_size = 200
        snapshot_list = []

        for i in range(0, len(option_codes), chunk_size):
            chunk = option_codes[i:i + chunk_size]
            ret_s, snap_data = quote_ctx.get_market_snapshot(chunk)
            if ret_s == RET_OK:
                snapshot_list.append(snap_data)

        if not snapshot_list:
            print("No snapshot data.")
            continue

        full_snap = pd.concat(snapshot_list)
        merged_df = pd.merge(chain_info, full_snap, on='code', suffixes=('', '_snap'))

        # --- HYBRID CALCULATION ---
        merged_df['gamma'] = pd.to_numeric(merged_df['option_gamma'], errors='coerce').fillna(0)

        # Check if API failed (returns all zeros)
        if merged_df['gamma'].sum() == 0:
            today = datetime.now().date()
            exp_date = datetime.strptime(expiry, "%Y-%m-%d").date()
            days_to_exp = (exp_date - today).days
            T = max(days_to_exp, 1) / 365.0

            merged_df['iv'] = pd.to_numeric(merged_df['option_implied_volatility'], errors='coerce').fillna(0)

            # Normalize IV
            if merged_df['iv'].mean() > 2.0:
                merged_df['iv'] = merged_df['iv'] / 100.0

            # Calculate manually
            merged_df['gamma'] = merged_df.apply(
                lambda row: black_scholes_gamma(
                    spot_price,
                    float(row['strike_price']),
                    T,
                    RISK_FREE_RATE,
                    float(row['iv'])
                ), axis=1
            )

        all_dfs.append(merged_df)
        print("Done.")

    quote_ctx.close()

    # --- CRITICAL FIX: ignore_index=True ---
    if not all_dfs:
        return None, spot_price

    final_df = pd.concat(all_dfs, ignore_index=True)
    return final_df, spot_price


def process_and_plot(df, spot_price):
    if df is None or df.empty: return

    # Safety Reset
    df = df.reset_index(drop=True)

    # Normalize Columns
    df['OI'] = pd.to_numeric(df['option_open_interest'], errors='coerce').fillna(0)

    # Calculate GEX ($ Value)
    df['GEX_K'] = (df['gamma'] * df['OI'] * 100 * (spot_price ** 2) * 0.01) / 1000

    # Flip signs for Puts
    mask_put = df['option_type'].astype(str).str.upper() == 'PUT'
    df.loc[mask_put, 'GEX_K'] = -df.loc[mask_put, 'GEX_K']

    # Filter & Pivot
    gex_by_strike = df.groupby(['strike_price', 'strike_time'])['GEX_K'].sum().reset_index()
    unique_strikes = gex_by_strike['strike_price'].unique()
    closest_indices = np.abs(unique_strikes - spot_price).argsort()[:STRIKES_TO_SHOW]
    focused_strikes = unique_strikes[closest_indices]

    gex_by_strike = gex_by_strike[gex_by_strike['strike_price'].isin(focused_strikes)]

    # Pivot
    pivot_df = gex_by_strike.pivot_table(index='strike_price', columns='strike_time', values='GEX_K', aggfunc='sum')
    pivot_df.sort_index(ascending=False, inplace=True)
    pivot_df.fillna(0, inplace=True)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    try:
        annot_df = pivot_df.map(lambda x: f"${x:,.0f}K" if abs(x) > 1 else "")
    except:
        annot_df = pivot_df.applymap(lambda x: f"${x:,.0f}K" if abs(x) > 1 else "")

    sns.heatmap(pivot_df, annot=annot_df, fmt="", cmap=custom_cmap, center=0,
                linewidths=0.5, linecolor='#111111', ax=ax)

    ax.set_title(f"{TARGET_STOCK} Gamma Walls (Spot: ${spot_price:.2f})", fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Dynamic Filename
    filename = f"{TARGET_STOCK}_gex_heatmap.png"
    plt.savefig(filename, dpi=300)
    print(f"Success! Chart saved to '{filename}'")


if __name__ == "__main__":
    df, spot = get_moomoo_data()
    if df is not None:
        process_and_plot(df, spot)