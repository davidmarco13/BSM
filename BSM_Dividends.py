# Black-Scholes Pricing Section
st.header("Black-Scholes Option Pricing")
spot_prices = np.linspace(80, 120, 10)
volatilities = np.linspace(0.1, 0.3, 10)
call_prices = np.zeros((len(volatilities), len(spot_prices)))
put_prices = np.zeros((len(volatilities), len(spot_prices)))

# Loop through spot prices and volatilities to calculate option prices
for i, vol in enumerate(volatilities):
    for j, spot in enumerate(spot_prices):
        bs_model = BlackScholes(time_to_maturity, strike, spot, vol, interest_rate, dividend_yield)
        call_prices[i, j], put_prices[i, j] = bs_model.calculate_prices()

# Plotting the heatmaps for call and put prices
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

# Now calculate the call and put prices for the current input values
bs_model = BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate, dividend_yield)
call_price, put_price = bs_model.calculate_prices()

# Display Call and Put Prices in colored tables for Black-Scholes model
col1, col2 = st.columns([1, 1], gap="small")

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
