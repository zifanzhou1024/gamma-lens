import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import seaborn as sns
from datetime import datetime

# --- CONFIGURATION ---
TICKER_SYMBOL = "SPY"
RISK_FREE_RATE = 0.045
NUM_EXPIRATIONS_TO_ANALYZE = 5
STRIKES_TO_SHOW = 30  # Focused view

# Set dark theme
plt.style.use('dark_background')

# --- COLORS ---
colors = [
    (0.0, '#5e0000'),  # Deep Red
    (0.3, '#ff0000'),  # Red
    (0.5, '#252525'),  # Dark Grey
    (0.7, '#00d600'),  # Green
    (1.0, '#ffff00')  # Yellow
]
custom_cmap = mcolors.LinearSegmentedColormap.from_list("HighContrastGEX", colors)


def black_scholes_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return 0.0
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        gamma = si.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        return gamma
    except:
        return 0.0


def process_chain_data(calls, puts, spot_price, target_date):
    if calls.empty or puts.empty:
        return pd.DataFrame()

    calls = calls.copy()
    puts = puts.copy()
    calls['Type'] = 'Call'
    puts['Type'] = 'Put'

    df = pd.concat([calls, puts]).reset_index(drop=True)

    today = datetime.now().date()
    try:
        exp_date_obj = datetime.strptime(target_date, "%Y-%m-%d").date()
    except ValueError:
        return pd.DataFrame()

    days_to_expiry = (exp_date_obj - today).days
    # Ensure T is at least 1/365 to avoid division by zero for 0DTE
    T = max(days_to_expiry, 1) / 365.0

    # Calculate Gamma
    df['gamma'] = df.apply(
        lambda row: black_scholes_gamma(spot_price, row['strike'], T, RISK_FREE_RATE, row['impliedVolatility']), axis=1)

    # GEX Calculation
    df['GEX_K'] = (df['gamma'] * df['openInterest'] * 100 * (spot_price ** 2) * 0.01) / 1000
    df.loc[df['Type'] == 'Put', 'GEX_K'] = -df['GEX_K']

    gex_by_strike = df.groupby('strike')['GEX_K'].sum().reset_index()
    gex_by_strike['expiry'] = target_date

    return gex_by_strike


def plot_heatmap(pivot_df, spot_price):
    fig, ax = plt.subplots(figsize=(12, 10))

    try:
        annot_df = pivot_df.map(lambda x: f"${x:,.0f}K" if not pd.isna(x) and x != 0 else "")
    except AttributeError:
        annot_df = pivot_df.applymap(lambda x: f"${x:,.0f}K" if not pd.isna(x) and x != 0 else "")

    sns.heatmap(pivot_df, annot=annot_df, fmt="", cmap=custom_cmap, center=0,
                linewidths=0.5, linecolor='#111111',
                cbar_kws={'label': 'Net Gamma Exposure ($ Thousands)'}, ax=ax)

    ax.set_title(f"{TICKER_SYMBOL} Gamma Walls (Spot: ${spot_price:.2f})", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Expiration Date", fontsize=12)
    ax.set_ylabel("Strike Price", fontsize=12)
    plt.xticks(rotation=45)

    # Spot Price Marker
    strikes = pivot_df.index.astype(float)
    closest_strike_idx = (np.abs(strikes - spot_price)).argmin()
    yticklabels = [label.get_text() for label in ax.get_yticklabels()]

    if 0 <= closest_strike_idx < len(yticklabels):
        yticklabels[closest_strike_idx] = f"▶ {yticklabels[closest_strike_idx]}"
        ax.set_yticklabels(yticklabels, fontweight='bold', color='white')

    # Highlight Put Wall
    if not pivot_df.empty:
        nearest_expiry = pivot_df.columns[0]
        if nearest_expiry in pivot_df.columns:
            put_wall_strike = pivot_df[nearest_expiry].idxmin()
            if put_wall_strike in pivot_df.index:
                row_idx = pivot_df.index.get_loc(put_wall_strike)
                col_idx = pivot_df.columns.get_loc(nearest_expiry)
                rect = patches.Rectangle((col_idx, row_idx), 1, 1, linewidth=3, edgecolor='#aa00ff', facecolor='none')
                ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig("gex_heatmap.png", dpi=300)
    print("Done. Chart saved to 'gex_heatmap.png'.")


def run_engine():
    print(f"--- Fetching Data for {TICKER_SYMBOL} ---")
    ticker = yf.Ticker(TICKER_SYMBOL)

    # 1. Fetch Spot
    try:
        spot_price = ticker.fast_info['last_price']
    except:
        hist = ticker.history(period="1d")
        spot_price = hist['Close'].iloc[-1]
    print(f"--- SPOT PRICE: ${spot_price:.2f} ---")

    # 2. Get Expirations & SORT THEM
    try:
        # Force sort to ensure we get the absolute nearest dates
        expirations = sorted(ticker.options)
    except Exception:
        print("Error fetching options dates.")
        return

    print(f"Found {len(expirations)} expirations.")

    all_data = []
    valid_expiries_found = 0

    # 3. Process Strictly the First Available Dates
    for expiry in expirations:
        if valid_expiries_found >= NUM_EXPIRATIONS_TO_ANALYZE:
            break

        print(f"Fetching {expiry}...", end=" ")

        try:
            opt = ticker.option_chain(expiry)
            calls, puts = opt.calls, opt.puts
        except Exception:
            print("Failed (API Error).")
            continue

        if len(calls) == 0:
            print("Skipping (Empty).")
            continue

        # Processing
        df = process_chain_data(calls, puts, spot_price, expiry)

        if not df.empty:
            all_data.append(df)
            valid_expiries_found += 1
            print("Success.")
        else:
            print("Skipping (Calculation Error).")

    if not all_data:
        print("No valid data found.")
        return

    full_df = pd.concat(all_data)

    # 4. Filter Strikes
    unique_strikes = full_df['strike'].unique()
    distances = np.abs(unique_strikes - spot_price)
    closest_indices = distances.argsort()[:STRIKES_TO_SHOW]
    focused_strikes = unique_strikes[closest_indices]

    full_df = full_df[full_df['strike'].isin(focused_strikes)]

    # 5. Pivot
    pivot_df = full_df.pivot_table(index='strike', columns='expiry', values='GEX_K', aggfunc='sum')
    pivot_df.sort_index(ascending=False, inplace=True)
    pivot_df.fillna(0, inplace=True)

    plot_heatmap(pivot_df, spot_price)


if __name__ == "__main__":
    run_engine()