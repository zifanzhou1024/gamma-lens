# Gamma-Lens 📊

**Gamma-Lens** is an asynchronous financial analytics bot for Discord. It democratizes institutional-grade options analysis by calculating Net Gamma Exposure (GEX) profiles on demand.

Instead of relying on expensive data subscriptions, Gamma-Lens calculates implied volatility and gamma Greeks from scratch using the Black-Scholes structural model and public data via `yfinance`.

## ✨ Features

* **Algorithmic Analysis:** Implements the Black-Scholes pricing model to calculate Gamma ($\Gamma$) for every strike in the option chain.
* **Visualizations:** Generates "Call Wall" vs. "Put Wall" exposure charts using `matplotlib`.
* **Zero-Disk Latency:** Plots are rendered into an in-memory bytes buffer (`io.BytesIO`) and uploaded directly to Discord, avoiding slow disk I/O operations.
* **Non-Blocking Execution:** Heavy mathematical computations and rendering are offloaded to a thread pool, ensuring the bot remains responsive to other commands.
* **Secure Configuration:** Uses environment variables to securely handle API tokens.

## 🛠️ Tech Stack

* **Core:** Python 3.x, `discord.py`
* **Data:** `yfinance`, `pandas`
* **Math:** `numpy`, `scipy` (stats/norm)
* **Visualization:** `matplotlib`

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone [https://github.com/zifanzhou1024/gamma-lens.git](https://github.com/zifanzhou1024/gamma-lens.git)
cd gamma-lens
````

### 2\. Set Up Virtual Environment

It is recommended to use a virtual environment to manage dependencies.

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3\. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4\. Configuration

Create a `.env` file in the root directory to store your secrets. **Do not commit this file.**

```env
# .env
DISCORD_TOKEN=your_discord_bot_token_here
```

### 5\. Run the Bot

```bash
python discord_reply_bot.py
```

## 🎮 Usage

Once the bot is running and invited to your server, use the following command:

```text
!gex [TICKER]
```

Calculates the Net Gamma Exposure for the nearest expiration date.

**Examples:**

```text
!gex SPY
!gex NVDA
!gex TSLA
```

### How to read the chart:

  * **Green Bars (Positive Gamma):** Represent "Call Walls." Dealers are typically long gamma here, dampening volatility (buying dips, selling rips).
  * **Red Bars (Negative Gamma):** Represent "Put Walls." Dealers are typically short gamma here, amplifying volatility (selling dips, buying rips).

## 🧠 How It Works

1.  **Data Fetching:** The bot pulls the Option Chain for the requested ticker.

2.  **Greeks Calculation:** For every strike price, it calculates Gamma using the Black-Scholes formula:

    $$
    \Gamma = \frac{N'(d_1)}{S \sigma \sqrt{T}}
    $$

3.  **Net GEX Summation:**

      * Call Gamma is added.
      * Put Gamma is subtracted (assuming dealer short positioning).
      * Multiplied by Open Interest and Spot Price to get notional exposure.

4.  **Rendering:** The data is aggregated and plotted using a custom matplotlib dark theme.

## ⚠️ Limitations

  * **Data Delay:** `yfinance` data is not real-time (usually 15 min delayed or EOD for options). This tool is for educational analysis, not high-frequency trading.
  * **Assumption:** The model assumes dealers are Short Puts and Long Calls. While generally true for market makers, this is a heuristic.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
