import streamlit as st

st.set_page_config(page_title="Option Pricing Calculator", layout="wide")

st.title("Black–Scholes Option Calculator")
st.caption("European option prices and Greeks")

ticker = st.selectbox("Ticker", ["SOFI", "HIMS", "NOW", "VOO"])


import yfinance as yf
import numpy as np

@st.cache_data(ttl=300)
def load_market_data(symbol):
    history = yf.Ticker(symbol).history(period="1y", auto_adjust=True)
    closes = history["Close"].dropna()

    if len(closes) < 3:
        raise ValueError("Not enough price data.")

    returns = np.log(closes / closes.shift(1)).dropna()
    historical_vol = float(returns.std() * np.sqrt(252))

    # Use an unadjusted recent closing price for the stock input
    recent = yf.Ticker(symbol).history(period="5d", auto_adjust=False)
    spot = float(recent["Close"].dropna().iloc[-1])

    return spot, historical_vol

try:
    market_spot, market_vol = load_market_data(ticker)
except Exception as error:
    st.error(f"Could not load market data: {error}")
    st.stop()

st.caption("Yahoo Finance: latest available close and 1-year historical volatility.")

S = st.number_input("Stock price ($)", min_value=0.01, value=market_spot, format="%.3f", key=f"spot_{ticker}")

import pandas as pd

@st.cache_data(ttl=300)
def load_expirations(symbol):
    return yf.Ticker(symbol).options

@st.cache_data(ttl=300)
def load_strikes(symbol, expiration):
    chain = yf.Ticker(symbol).option_chain(expiration)
    return sorted(set(chain.calls["strike"]) | set(chain.puts["strike"]))

today = pd.Timestamp.now(tz="America/New_York").date()

try:
    expirations = [
        date for date in load_expirations(ticker)
        if pd.Timestamp(date).date() > today
    ]

    if not expirations:
        st.warning("No future expiration dates available.")
        st.stop()

    expiration = st.selectbox(
        "Expiration date", expirations, key=f"expiration_{ticker}"
    )

    strikes = load_strikes(ticker, expiration)

    if not strikes:
        st.warning("No strikes available for this expiration.")
        st.stop()

except Exception as error:
    st.error(f"Could not load option contracts: {error}")
    st.stop()

nearest = min(range(len(strikes)), key=lambda i: abs(strikes[i] - S))

K = st.selectbox(
    "Strike price ($)",
    strikes,
    index=nearest,
    key=f"strike_{ticker}_{expiration}"
)

days = (pd.Timestamp(expiration).date() - today).days
st.write(f"Days to expiration: **{days}**")


vol_pct = st.number_input(
    "Annual volatility (%)", min_value=0.01, value=market_vol * 100, key=f"vol_{ticker}"
)
rate_pct = st.number_input("Risk-free rate (%)", value=4.0)
dividend_pct = st.number_input(
    "Dividend yield (%)", min_value=0.0, value=0.0
)

# Convert dashboard inputs into model units
T = days / 365
sigma = vol_pct / 100
r = rate_pct / 100
q = dividend_pct / 100

st.caption("Choose a listed expiration and strike. Time uses calendar days; same-day expirations are excluded.")

import numpy as np
from scipy.stats import norm

d1 = (
    np.log(S / K) + (r - q + 0.5 * sigma**2) * T
) / (sigma * np.sqrt(T))

d2 = d1 - sigma * np.sqrt(T)

call_price = (
    S * np.exp(-q * T) * norm.cdf(d1)
    - K * np.exp(-r * T) * norm.cdf(d2)
)

put_price = (
    K * np.exp(-r * T) * norm.cdf(-d2)
    - S * np.exp(-q * T) * norm.cdf(-d1)
)

st.subheader("Theoretical prices")

left, right = st.columns(2)
left.metric("Call price per share", f"${call_price:.4f}")
right.metric("Put price per share", f"${put_price:.4f}")

import pandas as pd

discount_q = np.exp(-q * T)
discount_r = np.exp(-r * T)

gamma = discount_q * norm.pdf(d1) / (S * sigma * np.sqrt(T))
vega = S * discount_q * norm.pdf(d1) * np.sqrt(T) / 100

decay = -S * discount_q * norm.pdf(d1) * sigma / (2 * np.sqrt(T))

call_theta = (
    decay - r * K * discount_r * norm.cdf(d2)
    + q * S * discount_q * norm.cdf(d1)
) / 365

put_theta = (
    decay + r * K * discount_r * norm.cdf(-d2)
    - q * S * discount_q * norm.cdf(-d1)
) / 365

greeks = pd.DataFrame(
    {
        "Call": [
            discount_q * norm.cdf(d1),
            gamma,
            call_theta,
            vega,
            K * T * discount_r * norm.cdf(d2) / 100
        ],
        "Put": [
            discount_q * (norm.cdf(d1) - 1),
            gamma,
            put_theta,
            vega,
            -K * T * discount_r * norm.cdf(-d2) / 100
        ]
    },
    index=[
        "Delta",
        "Gamma",
        "Theta ($/share/day)",
        "Vega ($/share/volatility point)",
        "Rho ($/share/rate point)"
    ]
)

st.subheader("Option Greeks")
st.dataframe(greeks.style.format("{:.4f}"))

# Compare theoretical prices with market quotes
@st.cache_data(ttl=300)
def load_quotes(symbol, expiry):
    chain = yf.Ticker(symbol).option_chain(expiry)
    return chain.calls, chain.puts

st.subheader("Model vs. market")

if "expiration" not in globals():
    st.warning("First connect the expiration and strike dropdowns.")
else:
    try:
        calls, puts = load_quotes(ticker, expiration)
        rows = []

        for label, contracts, model_price in [
            ("Call", calls, call_price),
            ("Put", puts, put_price)
        ]:
            match = contracts.loc[np.isclose(contracts["strike"], K)]

            if match.empty:
                continue

            contract = match.iloc[0]
            bid = float(contract["bid"])
            ask = float(contract["ask"])

            valid = (
                np.isfinite(bid) and np.isfinite(ask)
                and bid >= 0 and ask > 0 and ask >= bid
            )

            midpoint = (bid + ask) / 2 if valid else np.nan

            rows.append({
                "Option": label,
                "Model price": model_price,
                "Bid": bid,
                "Ask": ask,
                "Midpoint": midpoint,
                "Model minus midpoint": model_price - midpoint
            })

        if rows:
            comparison = pd.DataFrame(rows).set_index("Option")
            st.dataframe(
                comparison.style.format("{:.4f}", na_rep="Unavailable")
            )
        else:
            st.info("No matching contracts found for this strike.")

        st.caption(
            "All prices are dollars per share. Quotes may be delayed or stale. "
            "The midpoint is not a guaranteed execution price."
        )

    except Exception as error:
        st.error(f"Could not load market quotes: {error}")

st.subheader("Option prices vs. stock price")

# Test stock prices from 30% below to 30% above the current input
stock_range = np.linspace(S * 0.7, S * 1.3, 100)

scenario_d1 = (
    np.log(stock_range / K) + (r - q + 0.5 * sigma**2) * T
) / (sigma * np.sqrt(T))

scenario_d2 = scenario_d1 - sigma * np.sqrt(T)

scenario_calls = (
    stock_range * np.exp(-q * T) * norm.cdf(scenario_d1)
    - K * np.exp(-r * T) * norm.cdf(scenario_d2)
)

scenario_puts = (
    K * np.exp(-r * T) * norm.cdf(-scenario_d2)
    - stock_range * np.exp(-q * T) * norm.cdf(-scenario_d1)
)

chart_data = pd.DataFrame({
    "Stock price ($)": stock_range,
    "Call": scenario_calls,
    "Put": scenario_puts
}).set_index("Stock price ($)")

st.line_chart(chart_data)

st.caption(
    "Theoretical option prices per share, holding time, volatility, "
    "rates, dividends, and strike constant. This is not a profit/loss chart."
)

export = pd.concat([
    pd.DataFrame(
        {"Call": [call_price], "Put": [put_price]},
        index=["Price ($/share)"]
    ),
    greeks
])

export.index.name = "Measure"
export = export.reset_index()

export["Ticker"] = ticker
export["Expiration"] = expiration
export["Strike"] = K
export["Stock price"] = S
export["Volatility (%)"] = sigma * 100
export["Risk-free rate (%)"] = r * 100
export["Dividend yield (%)"] = q * 100

st.download_button(
    label="Download prices and Greeks (CSV)",
    data=export.to_csv(index=False).encode("utf-8"),
    file_name=f"{ticker}_{expiration}_{K:g}_results.csv",
    mime="text/csv",
    key="download_option_results"
)

st.subheader("Explain my option")
st.caption("Practice mode: template explanation; no API calls.")

if st.button("Explain these results", key="explain_practice"):
    st.markdown(f"""
**{ticker} — strike ${K:.2f}, {days} days remaining**

- **Model prices:** call ${call_price:.4f}, put ${put_price:.4f} per share.
- **Delta:** a small $1 stock increase changes the call by approximately
  ${greeks.loc["Delta", "Call"]:.4f} and the put by
  ${greeks.loc["Delta", "Put"]:.4f} per share.
- **Time decay:** holding other inputs constant, one day passing changes
  the call by approximately ${call_theta:.4f} and the put by
  ${put_theta:.4f} per share.
- **Volatility:** a 1 percentage point increase changes either option
  by approximately ${vega:.4f} per share.

These are local estimates under European Black–Scholes assumptions.
""")
