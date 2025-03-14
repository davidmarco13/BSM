import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
import plotly.graph_objects as go
from numpy import log, sqrt, exp  # Make sure to import these
import matplotlib.pyplot as plt
import seaborn as sns

#######################
# Page configuration
st.set_page_config(
    page_title="Black-Scholes Option Pricing Model",
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

        self.call_price = call_price
        self.put_price = put_price
        return call_price, put_price

# Sidebar for User Inputs
with st.sidebar:
    st.title("📊 Black-Scholes Model")
    with st.expander("📌 Basic Inputs", expanded=True):
        current_price = st.number_input("Current Stock Price", value=100.0, step=0.1)
        strike = st.number_input("Strike Price", value=100.0, step=0.1)
        time_to_maturity = st.slider("Time to Maturity (Years)", min_value=0.01, max_value=5.0, value=1.0, step=0.01)
        volatility = st.slider("Volatility (σ)", min_value=0.01, max_value=1.0, value=0.2, step=0.01)
    
    with st.expander("⚙️ Advanced Inputs", expanded=False):
        interest_rate = get_risk_free_rate()
        st.write(f"Risk-Free Interest Rate (Auto-Fetched): {interest_rate:.2%}")
        dividend_yield = st.slider("Dividend Yield (q)", min_value=0.00, max_value=0.10, value=0.00, step=0.001)

# Calculate Call and Put values
bs_model = BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate, dividend_yield)
call_price, put_price = bs_model.calculate_prices()

# Display Call and Put Values in colored tables
col1, col2 = st.columns([1,1], gap="small")

with col1:
    st.markdown(f"""
        <div class="metric-container metric-call">
            <div>
                <div class="metric-label">CALL Value</div>
                <div class="metric-value">${call_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-container metric-put">
            <div>
                <div class="metric-label">PUT Value</div>
                <div class="metric-value">${put_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)