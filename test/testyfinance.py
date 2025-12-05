import yfinance as yf

# Check what raw data looks like
ticker = yf.Ticker("SPY")
try:
    expiration = ticker.options[0]  # Get first available date
    print(f"Checking data for expiry: {expiration}")

    chain = ticker.option_chain(expiration)
    calls = chain.calls

    print("\n--- First 5 Rows of Raw Call Data ---")
    print(calls[['strike', 'lastPrice', 'impliedVolatility', 'openInterest']].head())

    # Check specifically for zeros
    zeros_iv = calls[calls['impliedVolatility'] == 0].shape[0]
    zeros_oi = calls[calls['openInterest'] == 0].shape[0]
    print(f"\nTotal Rows: {len(calls)}")
    print(f"Rows with 0 Implied Volatility: {zeros_iv}")
    print(f"Rows with 0 Open Interest: {zeros_oi}")

except Exception as e:
    print(f"Error fetching data: {e}")