import discord
from discord.ext import commands
import os
import io
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# Math & Data Imports
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as si
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from datetime import datetime

# ---------------------------------------------------------
# SETUP & CONFIG
# ---------------------------------------------------------

# 1. Load Environment Variables (Security Step)
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

# 2. Matplotlib Setup (Non-interactive backend)
matplotlib.use('Agg')
PLOT_STYLE = 'dark_background'

# 3. Discord Bot Setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)


# ---------------------------------------------------------
# FINANCIAL LOGIC (Black-Scholes & GEX)
# ---------------------------------------------------------

def black_scholes_gamma(S, K, T, r, sigma):
    """Calculates Gamma using Black-Scholes."""
    # Avoid division by zero
    if T <= 0 or sigma <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    pdf_d1 = si.norm.pdf(d1)
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    return gamma


def generate_gex_chart(ticker_symbol):
    """
    Fetches data from yfinance, calculates GEX, and returns a specific image buffer.
    Returns: (buffer, spot_price, total_gex) or (None, 0, 0) on failure.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)

        # 1. Get Spot Price
        try:
            # Try fast info first
            spot_price = ticker.fast_info['last_price']
        except:
            # Fallback to history
            hist = ticker.history(period="1d")
            if hist.empty: return None, 0, 0
            spot_price = hist['Close'].iloc[-1]

        # 2. Get Nearest Expiry
        if not ticker.options:
            return None, 0, 0

        target_date = ticker.options[0]  # Nearest expiry

        # 3. Get Option Chain
        opt = ticker.option_chain(target_date)
        calls = opt.calls.copy()
        puts = opt.puts.copy()

        if calls.empty and puts.empty:
            return None, 0, 0

        calls['Type'] = 'Call'
        puts['Type'] = 'Put'

        # Merge and clean
        df = pd.concat([calls, puts]).reset_index(drop=True)
        df = df.fillna(0)

        # 4. Time to Expiry (T)
        today = datetime.now().date()
        try:
            exp_date_obj = datetime.strptime(target_date, "%Y-%m-%d").date()
        except:
            # Handle potential yfinance date format quirks
            return None, 0, 0

        days_to_expiry = (exp_date_obj - today).days
        # If 0DTE, assume small fraction of day remaining (e.g., 0.5/365) or minimum 1 day for stability
        T = max(days_to_expiry, 0.5) / 365.0

        RISK_FREE_RATE = 0.045  # 4.5%

        # 5. Calculate Gamma & GEX
        # Vectorized calculation would be faster, but keeping your logic for readability
        df['gamma'] = df.apply(
            lambda row: black_scholes_gamma(spot_price, row['strike'], T, RISK_FREE_RATE, row['impliedVolatility']),
            axis=1
        )

        # GEX Formula: Gamma * Open Interest * 100 * Spot^2 * 0.01 (Dollar Gamma)
        df['GEX'] = df['gamma'] * df['openInterest'] * 100 * (spot_price ** 2) * 0.01

        # Dealer Short Puts -> Negative GEX, Dealer Long Calls -> Positive GEX
        df.loc[df['Type'] == 'Put', 'GEX'] = -df['GEX']

        # 6. Aggregation
        gex_by_strike = df.groupby('strike')['GEX'].sum().sort_index()

        # ---------------------------------------------------------
        # PLOTTING
        # ---------------------------------------------------------
        plt.style.use(PLOT_STYLE)
        fig, ax = plt.subplots(figsize=(10, 6))

        # Colors: Green for Call Wall (Pos), Red for Put Wall (Neg)
        colors = ['#00FF00' if x >= 0 else '#FF0000' for x in gex_by_strike.values]

        ax.bar(gex_by_strike.index, gex_by_strike.values, color=colors, alpha=0.7, width=1.0)

        # Spot Line
        ax.axvline(x=spot_price, color='white', linestyle='--', linewidth=1.5, label=f'Spot: ${spot_price:,.2f}')

        # Titles
        total_gex = gex_by_strike.sum()
        ax.set_title(f"{ticker_symbol.upper()} Gamma Exposure ({target_date})", fontsize=14, fontweight='bold',
                     color='white')
        ax.set_xlabel("Strike Price", color='white')
        ax.set_ylabel("Net Gamma ($Bn)", color='white')

        # Formatting Axis
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        # Format Y-axis to Billions
        def billions(x, pos):
            return f'${x * 1e-9:.1f}B'

        ax.yaxis.set_major_formatter(FuncFormatter(billions))

        # Zoom limits (Spot +/- 10%)
        ax.set_xlim(spot_price * 0.90, spot_price * 1.10)

        ax.grid(color='#333333', linestyle=':', linewidth=0.5)
        ax.legend()

        plt.tight_layout()

        # Save to Buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        plt.close(fig)

        return buf, spot_price, target_date

    except Exception as e:
        print(f"Error processing {ticker_symbol}: {e}")
        return None, 0, 0


# ---------------------------------------------------------
# DISCORD COMMANDS
# ---------------------------------------------------------

@bot.event
async def on_ready():
    print(f'✅ Gamma-Lens Active: {bot.user}')


@bot.command(name='gex')
async def gex(ctx, ticker: str = "SPY"):
    """
    Usage: !gex SPY
    Calculates Gamma Exposure using YFinance data.
    """
    ticker = ticker.upper()
    status_msg = await ctx.send(f"🔄 **{ticker}**: Crunching Black-Scholes data... (using yfinance)")

    # Offload blocking math/plot to a thread
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        # Run the generate_gex_chart function in a separate thread
        buf, spot, date = await loop.run_in_executor(pool, generate_gex_chart, ticker)

    if buf is None:
        await status_msg.edit(
            content=f"❌ **Error**: Could not fetch data for `{ticker}`. Market might be closed or ticker invalid.")
        return

    # Send Result
    file = discord.File(fp=buf, filename=f"{ticker}_gex.png")

    embed = discord.Embed(title=f"📊 {ticker} Net Gamma Profile", color=0x00ff00)
    embed.add_field(name="Spot Price", value=f"${spot:,.2f}", inline=True)
    embed.add_field(name="Expiry Analyzed", value=f"{date}", inline=True)
    embed.set_image(url=f"attachment://{ticker}_gex.png")
    embed.set_footer(text="Gamma-Lens • Data via yfinance • Black-Scholes Model")

    await ctx.send(file=file, embed=embed)
    await status_msg.delete()


# ---------------------------------------------------------
# RUNNER
# ---------------------------------------------------------

if __name__ == "__main__":
    if not TOKEN:
        print("❌ Error: DISCORD_TOKEN not found in .env file.")
    else:
        bot.run(TOKEN)