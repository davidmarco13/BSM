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

# Table of Inputs
input_data = {
    "Current Asset Price": [current_price],
    "Strike Price": [strike],
    "Time to Maturity (Years)": [time_to_maturity],
    "Volatility (σ)": [volatility],
    "Risk-Free Interest Rate": [interest_rate],
}
input_df = pd.DataFrame(input_data)
st.table(input_df)

# Black-Scholes Pricing Section
st.header("Black-Scholes Option Pricing")
st.info("Visual representation of call/put price variations under the Black-Scholes Model, incorporating dividend input.")

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


# Monte Carlo Simulation Section
st.header("Monte Carlo Option Pricing")
st.info("Calculates and displays call/put option prices using the Black-Scholes model with input for dividends.")

def monte_carlo_option_pricing(S, K, T, r, sigma, simulations=10000):
    np.random.seed(42)
    Z = np.random.standard_normal(simulations)
    ST = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    call_payoff = np.maximum(ST - K, 0)
    put_payoff = np.maximum(K - ST, 0)
    call_price = np.exp(-r * T) * np.mean(call_payoff)
    put_price = np.exp(-r * T) * np.mean(put_payoff)
    return call_price, put_price

mc_call, mc_put = monte_carlo_option_pricing(current_price, strike, time_to_maturity, interest_rate, volatility)
st.write(f"Monte Carlo Call Option Price: {mc_call:.2f}")
st.write(f"Monte Carlo Put Option Price: {mc_put:.2f}")

# Display Call and Put Values in colored tables for Montecarlo Simulation
col1, col2 = st.columns([1,1], gap="small")

with col1:
    # Using the custom class for CALL value
    st.markdown(f"""
        <div class="metric-container metric-call">
            <div>
                <div class="metric-label">CALL Value</div>
                <div class="metric-value">${mc_call:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    # Using the custom class for PUT value
    st.markdown(f"""
        <div class="metric-container metric-put">
            <div>
                <div class="metric-label">PUT Value</div>
                <div class="metric-value">${mc_put:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Sidebar Inputs for Monte Carlo Simulation
with st.sidebar:
    st.title("📈 Montecarlo Simulations")
    st.subheader("Parameters")
    simulations = st.number_input("Number of Simulations", min_value=100, max_value=10000, step=100, value=100)
    time_steps = st.number_input("Number of Time Steps", min_value=50, max_value=1000, step=10, value=365)

def monte_carlo_option_pricing(S, K, T, r, sigma, simulations=100, time_steps=100):
    np.random.seed(42)
    dt = T / time_steps  # Time step
    paths = np.zeros((simulations, time_steps + 1))
    paths[:, 0] = S

    for t in range(1, time_steps + 1):
        Z = np.random.standard_normal(simulations)  # Random variables for each simulation
        paths[:, t] = paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    return paths

# Running Monte Carlo Simulation with User Inputs
simulated_paths = monte_carlo_option_pricing(current_price, strike, time_to_maturity, interest_rate, volatility, simulations=simulations, time_steps=time_steps)

# Plotting the simulated stock price paths
plt.figure(figsize=(12, 6))
for i in range(10):  # Plotting only 10 paths for clarity
    plt.plot(simulated_paths[i], lw=1)
plt.title('Simulated Stock Price Paths')
plt.xlabel('Time (Days)')
plt.ylabel('Stock Price')
plt.grid(True)
st.pyplot(plt, clear_figure=True)
