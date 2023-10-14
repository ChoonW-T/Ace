import pandas as pd
from apscheduler.schedulers.blocking import BlockingScheduler
from oandapyV20 import API
import oandapyV20.endpoints.orders as orders
from oandapyV20.contrib.requests import MarketOrderRequest, TakeProfitDetails, StopLossDetails
from oanda_candles import Pair, Gran, CandleClient

def get_candles(n):
    access_token = '3c14f449612584452383c19170ab39c5-e700ac9e887630d32cdacc00283775d3'
    client = CandleClient(access_token, real=False)
    collector = client.get_collector(Pair.EUR_USD, Gran.M15)
    return collector.grab(n)

def signal_generator(df):
    # Here we can expand the patterns as necessary
    patterns = [
        (is_bullish_engulfing, 2),
        (is_bearish_engulfing, 1),
        (is_morning_star, 2),
        (is_evening_star, 1),
        (is_bullish_pin_bar, 2),
        (is_bearish_pin_bar, 1),
        (is_bullish_harami, 2),
        (is_bearish_harami, 1)
    ]

    for pattern, signal in patterns:
        if pattern(df):
            return signal
    return None

def is_bullish_engulfing(df):
    prev_candle = df.iloc[-2]
    current_candle = df.iloc[-1]

    return prev_candle['Close'] < prev_candle['Open'] and \
           current_candle['Close'] > current_candle['Open'] and \
           current_candle['Close'] > prev_candle['Open'] and \
           current_candle['Open'] < prev_candle['Close']

def is_morning_star(df):
    first_candle = df.iloc[-3]
    second_candle = df.iloc[-2]
    third_candle = df.iloc[-1]

    return first_candle['Close'] < first_candle['Open'] and \
           (second_candle['Open'] > first_candle['Close'] or abs(second_candle['Close'] - second_candle['Open']) < (0.1 * abs(first_candle['Close'] - first_candle['Open']))) and \
           third_candle['Close'] > third_candle['Open'] and \
           third_candle['Close'] > (first_candle['Open'] + first_candle['Close']) / 2

def is_bearish_engulfing(df):
    prev_candle = df.iloc[-2]
    current_candle = df.iloc[-1]

    return prev_candle['Close'] > prev_candle['Open'] and \
           current_candle['Close'] < current_candle['Open'] and \
           current_candle['Open'] > prev_candle['Close'] and \
           current_candle['Close'] < prev_candle['Open']

def is_evening_star(df):
    first_candle = df.iloc[-3]
    second_candle = df.iloc[-2]
    third_candle = df.iloc[-1]

    return first_candle['Close'] > first_candle['Open'] and \
           (second_candle['Open'] < first_candle['Close'] or abs(second_candle['Close'] - second_candle['Open']) < (0.1 * abs(first_candle['Close'] - first_candle['Open']))) and \
           third_candle['Close'] < third_candle['Open'] and \
           third_candle['Close'] < (first_candle['Open'] + first_candle['Close']) / 2

def is_bullish_pin_bar(df):
    candle = df.iloc[-1]
    body_size = abs(candle['Close'] - candle['Open'])
    lower_wick = candle['Open'] - candle['Low']
    upper_wick = candle['High'] - candle['Close']

    return body_size < upper_wick and lower_wick > 2 * body_size and upper_wick < 0.1 * body_size

def is_bearish_pin_bar(df):
    candle = df.iloc[-1]
    body_size = abs(candle['Close'] - candle['Open'])
    lower_wick = candle['Open'] - candle['Low']
    upper_wick = candle['High'] - candle['Close']

    return body_size < lower_wick and upper_wick > 2 * body_size and lower_wick < 0.1 * body_size

def is_bullish_harami(df):
    prev_candle = df.iloc[-2]
    current_candle = df.iloc[-1]

    return prev_candle['Close'] < prev_candle['Open'] and \
           current_candle['Close'] > current_candle['Open'] and \
           current_candle['Open'] > prev_candle['Open'] and \
           current_candle['Close'] < prev_candle['Close']

def is_bearish_harami(df):
    prev_candle = df.iloc[-2]
    current_candle = df.iloc[-1]

    return prev_candle['Close'] > prev_candle['Open'] and \
           current_candle['Close'] < current_candle['Open'] and \
           current_candle['Open'] < prev_candle['Open'] and \
           current_candle['Close'] > prev_candle['Close']


def trading_job():
    candles = get_candles(3)
    dfstream = pd.DataFrame([(float(str(candle.bid.o)),
                              float(str(candle.bid.c)),
                              float(str(candle.bid.h)),
                              float(str(candle.bid.l))) for candle in candles],
                            columns=['Open', 'Close', 'High', 'Low'])

    signal = signal_generator(dfstream)
    accountID = "101-004-26842643-001"
    client = API(accountID)
    SLTPRatio = 2.
    previous_candleR = abs(dfstream['High'].iloc[-2] - dfstream['Low'].iloc[-2])
    current_open = dfstream['Open'].iloc[-1]

    SLBuy = current_open - previous_candleR
    SLSell = current_open + previous_candleR
    TPBuy = current_open + previous_candleR * SLTPRatio
    TPSell = current_open - previous_candleR * SLTPRatio

    print(dfstream.iloc[:-1, :])
    print(TPBuy, SLBuy, TPSell, SLSell)

    if signal == 1:  # Sell
        mo = MarketOrderRequest(instrument="EUR_USD", units=-1000,
                                takeProfitOnFill=TakeProfitDetails(price=TPSell).data,
                                stopLossOnFill=StopLossDetails(price=SLSell).data)
    elif signal == 2:  # Buy
        mo = MarketOrderRequest(instrument="EUR_USD", units=1000,
                                takeProfitOnFill=TakeProfitDetails(price=TPBuy).data,
                                stopLossOnFill=StopLossDetails(price=SLBuy).data)
    else:
        return

    r = orders.OrderCreate('101-004-26842643-001', data=mo.data)
    rv = client.request(r)
    print(rv)

trading_job()
#scheduler = BlockingScheduler()
#scheduler.add_job(trading_job, 'cron', day_of_week='mon-fri', hour='00-23', minute='1,16,31,46', start_date='2022-01-12 12:00:00', timezone='America/Chicago')
#scheduler.start()

