import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

class MonteCarloOptionPricing:
    def __init__(self, asset_price, strike_price, risk_free_rate, volatility, time_to_maturity):
        self.S = asset_price
        self.K = strike_price
        self.r = risk_free_rate
        self.sigma = volatility
        self.T = time_to_maturity

        # Initialize default time steps
        self.set_time_steps(int(self.T / 0.01))

    def set_time_steps(self, steps):
        """Set the number of time steps and update the time step size."""
        self.N = steps
        self.dt = self.T / self.N

    def simulate_paths(self, num_simulations):
        dW = np.random.normal(0, np.sqrt(self.dt), (num_simulations, self.N))
        paths = np.zeros((num_simulations, self.N + 1))
        paths[:, 0] = self.S

        for t in range(1, self.N + 1):
            paths[:, t] = paths[:, t - 1] * np.exp(
                (self.r - 0.5 * self.sigma ** 2) * self.dt + self.sigma * dW[:, t - 1])

        return paths

    def european_call_option_price(self, num_simulations):
        paths = self.simulate_paths(num_simulations)
        payoff = np.maximum(paths[:, -1] - self.K, 0)
        return np.exp(-self.r * self.T) * np.mean(payoff)

    def european_put_option_price(self, num_simulations):
        paths = self.simulate_paths(num_simulations)
        payoff = np.maximum(self.K - paths[:, -1], 0)
        return np.exp(-self.r * self.T) * np.mean(payoff)


# Placeholder data
initial_data = {
    'Asset': 'AAPL',
    'Asset_Price': 150,
    'Strike_Price': 155,
    'Risk_Free_Rate': 0.05,
    'Volatility': 0.2,
    'Time_to_Maturity': 1
}

# Sample simulation with placeholder data
mc = MonteCarloOptionPricing(initial_data['Asset_Price'], initial_data['Strike_Price'],
                             initial_data['Risk_Free_Rate'], initial_data['Volatility'],
                             initial_data['Time_to_Maturity'])

call_price = mc.european_call_option_price(10000)
put_price = mc.european_put_option_price(10000)

print(f"European Call Option Price: ${call_price:.2f}")
print(f"European Put Option Price: ${put_price:.2f}")

# Fetch historical data from Yahoo Finance for a list of assets
assets = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "JPM", "V", "PG", "JNJ", "XOM", "GS", "BRK-A", "WMT", "MA"]
start_date = "2022-01-01"
end_date = "2023-01-01"

asset_data = {}
for asset in assets:
    try:
        data = yf.download(asset, start=start_date, end=end_date)
        asset_data[asset] = data['Close'].iloc[-1]
    except Exception as e:
        print(f"Error fetching data for {asset}: {e}")

apple_price = asset_data.get('AAPL', initial_data['Asset_Price'])  # Use placeholder value as default if fetch fails

updated_data = {
    'Asset': 'AAPL',
    'Asset_Price': apple_price,
    'Strike_Price': apple_price + 5,
    'Risk_Free_Rate': 0.05,
    'Volatility': 0.2,
    'Time_to_Maturity': 1
}

mc = MonteCarloOptionPricing(updated_data['Asset_Price'], updated_data['Strike_Price'],
                             updated_data['Risk_Free_Rate'], updated_data['Volatility'],
                             updated_data['Time_to_Maturity'])

call_price = mc.european_call_option_price(10000)
put_price = mc.european_put_option_price(10000)

print(f"European Call Option Price for {updated_data['Asset']}: ${call_price:.2f}")
print(f"European Put Option Price for {updated_data['Asset']}: ${put_price:.2f}")


# Black-Scholes functions
def black_scholes_call_price(S, K, r, sigma, T):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put_price(S, K, r, sigma, T):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# This assumes the MonteCarloOptionPricing instance mc is already created
def compare_steps(steps_list, asset_data, simulations=10000):
    results = {}
    for steps in steps_list:
        mc.set_time_steps(steps)
        call_price = mc.european_call_option_price(simulations)
        put_price = mc.european_put_option_price(simulations)

        bs_call_price = black_scholes_call_price(asset_data['Asset_Price'], asset_data['Strike_Price'],
                                                 asset_data['Risk_Free_Rate'], asset_data['Volatility'],
                                                 asset_data['Time_to_Maturity'])
        bs_put_price = black_scholes_put_price(asset_data['Asset_Price'], asset_data['Strike_Price'],
                                               asset_data['Risk_Free_Rate'], asset_data['Volatility'],
                                               asset_data['Time_to_Maturity'])

        diff_call = abs(call_price - bs_call_price)
        diff_put = abs(put_price - bs_put_price)

        results[steps] = {
            "MC_Call": call_price,
            "MC_Put": put_price,
            "BS_Call": bs_call_price,
            "BS_Put": bs_put_price,
            "Diff_Call": diff_call,
            "Diff_Put": diff_put
        }

    return results

steps_list = [10, 100, 500, 1000, 5000, 10000]
results = compare_steps(steps_list, updated_data)
df_results = pd.DataFrame(results).transpose()
print(df_results)