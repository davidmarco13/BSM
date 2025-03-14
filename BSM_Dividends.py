import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
from numpy import log, sqrt, exp
import matplotlib.pyplot as plt
import seaborn as sns

#######################
# Page configuration
st.set_page_config(
    page_title="Black-Scholes Option Pricing Model (Dividends)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded")

# Fetch risk-free rate from Yahoo Finance
def get_risk_free_rate():
    try:
        t_bill = yf.Ticker("^IRX")  # 13-week Treasury Bill
        rate = t_bill.history(period="1d")["Close"].iloc[-1] / 100
        return rate
    except Exception as e:
        st.warning("Could not fetch risk-free rate, using default value of 5%.")
        return 0.05  # Default 5%

class BlackScholes:
    def __init__(
        self,
        time_to_maturity: float,
        strike: float,
        current_price: float,
        volatility: float,
        interest_rate: float,
        dividend_yield: float = 0.0
    ):
        self.time_to_maturity = time_to_maturity
        self.strike = strike
        self.current_price = current_price
        self.volatility = volatility
        self.interest_rate = interest_rate
        self.dividend_yield = dividend_yield

    def calculate_prices(self):
        time_to_maturity = self.time_to_maturity
        strike = self.strike
        current_price = self.current_price
        volatility = self.volatility
        interest_rate = self.interest_rate
        dividend_yield = self.dividend_yield

        d1 = (
            log(current_price / strike) +
            (interest_rate - dividend_yield + 0.5 * volatility ** 2) * time_to_maturity
        ) / (volatility * sqrt(time_to_maturity))
        d2 = d1 - volatility * sqrt(time_to_maturity)

        call_price = current_price * exp(-dividend_yield * time_to_maturity) * norm.cdf(d1) - (
            strike * exp(-(interest_rate * time_to_maturity)) * norm.cdf(d2)
        )
        put_price = (
            strike * exp(-(interest_rate * time_to_maturity)) * norm.cdf(-d2)
        ) - current_price * exp(-dividend_yield * time_to_maturity) * norm.cdf(-d1)

        return call_price, put_price

# Sidebar for User Inputs
with st.sidebar:
    st.title("📊 Black-Scholes Model")
    st.write("`Created by:`")
    linkedin_url = "https://www.linkedin.com/in/david-marco-sierra-a3a440235/"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`David Marco Sierra`</a>', unsafe_allow_html=True)

    current_price = st.number_input("Current Asset Price", value=100.0)
    strike = st.number_input("Strike Price", value=100.0)
    time_to_maturity = st.number_input("Time to Maturity (Years)", value=1.0)
    volatility = st.number_input("Volatility (σ)", value=0.2)
    dividend_yield = st.number_input("Dividend Yield (q)", value=0.0)
    
    interest_rate = get_risk_free_rate()
    st.write(f"Risk-Free Interest Rate (Auto-Fetched): {interest_rate:.2%}")

# Generate heatmaps
spot_prices = np.linspace(80, 120, 10)
volatilities = np.linspace(0.1, 0.3, 10)
call_prices = np.zeros((len(volatilities), len(spot_prices)))
put_prices = np.zeros((len(volatilities), len(spot_prices)))

for i, vol in enumerate(volatilities):
    for j, spot in enumerate(spot_prices):
        bs_model = BlackScholes(time_to_maturity, strike, spot, vol, interest_rate, dividend_yield)
        call_prices[i, j], put_prices[i, j] = bs_model.calculate_prices()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(call_prices, xticklabels=np.round(spot_prices, 2), 
            yticklabels=np.round(volatilities, 2), ax=axes[0], cmap="RdYlGn_r", annot=True)
axes[0].set_title("Call Price Heatmap")
axes[0].set_xlabel("Spot Price")
axes[0].set_ylabel("Volatility")

sns.heatmap(put_prices, xticklabels=np.round(spot_prices, 2), 
            yticklabels=np.round(volatilities, 2), ax=axes[1], cmap="RdYlGn_r", annot=True)
axes[1].set_title("Put Price Heatmap")
axes[1].set_xlabel("Spot Price")
axes[1].set_ylabel("Volatility")

plt.tight_layout()
st.pyplot(fig, clear_figure=True)
plt.close(fig)

# Tabla de Arriba y cosas esteticas
st.subheader("📌 Model Inputs")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Current Asset Price", value=f"${current_price:.2f}")

with col2:
    st.metric(label="Strike Price", value=f"${strike:.2f}")

with col3:
    st.metric(label="Time to Maturity", value=f"{time_to_maturity:.2f} years")

col4, col5 = st.columns(2)

with col4:
    st.metric(label="Volatility (σ)", value=f"{volatility:.2%}")

with col5:
   

