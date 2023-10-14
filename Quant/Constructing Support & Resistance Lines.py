import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np

df = yf.download('^GSPC',start='2023-06-01')
supports = df[df.Low == df.Low.rolling(5, center=True).min()].Low
resistances = df[df.High == df.High.rolling(5, center=True).min()].High

# Calculate the average size of a candlestick
s = np.mean(df['High'] - df['Low'])

# Concatenate both levels
levels_tmp = pd.concat([supports, resistances])

# Define the function to check if a level is far from other levels
def isFarFromLevel(l, levels):
    return np.sum([abs(l-x) < s for x in levels]) == 0

# Filter levels
levels = []

for level in levels_tmp:
    if isFarFromLevel(level, levels):
        levels.append(level)

# Plot the chart
mpf.plot(df, type='candle', hlines=levels, style='charles')

#https://www.youtube.com/watch?v=y78S-aCwqEM&t=220s