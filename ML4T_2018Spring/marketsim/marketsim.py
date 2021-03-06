"""MC2-P1: Market simulator.

Copyright 2017, Georgia Tech Research Corporation
Atlanta, Georgia 30332-0415
All Rights Reserved

Georgia tech.

Today I'm gonna cover the next project
which is to build a market simulator
when the project is actually graded, because you paid attention

Create a market simulator
create a web page for that to consult it

looking at the wiki for mc2 project 1

gonna talk about some of the aspects of it.

key things are - what comes in

is an orders file. What comes out
is a history of value.

"""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

def author():
    return "mdunn34"

def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000, commission=9.95, impact=0.005):
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input
    # TODO: Your code here

    orders_df = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])
    # In the template, instead of computing the value of the portfolio, we just
    # read in the value of IBM over 6 months

    orders_list = (list(orders_df.index))

    start_date = min(orders_list)
    end_date = max(orders_list)

    portfolio_symbols = list(orders_df["Symbol"].unique())
    portvals = get_data(portfolio_symbols, pd.date_range(start_date, end_date))
    portvals = portvals[portfolio_symbols]  # remove SPY

    # this is fine.
    prices = pd.DataFrame(index=portvals.index, data=portvals.as_matrix())    
    # Step 1
    # renaame the columns
    prices.columns = portfolio_symbols

    # for multiplication later
    prices["cash"] = np.ones(prices.shape[0])

    trades = pd.DataFrame(np.zeros(prices.shape), index=prices.index, columns=prices.columns)

    # look throught the orders file.
    # add orders to the trades dataframe.
    # has to be iteration, unfortunately.
    for i in range(orders_df.shape[0]):
        symbol = (orders_df["Symbol"].iloc[i])
        order = (orders_df["Order"].iloc[i])
        shares = (orders_df["Shares"].iloc[i])
        date = (orders_df.index[i])
        if date == dt.datetime(2011,6,15):
            pass
        else:
            try:
                if order == "BUY":
                    #specific date in the orders file index for trades file
                    trades[symbol].loc[date] += shares
                    # losing cash               
                if order == "SELL":
                    trades[symbol].loc[date] -= shares
                    # you gains the cash when you sells.
            except KeyError:
                pass



    trades["cash"] += np.sum((prices.iloc[:,:-1] * trades.iloc[:,:-1]) * -1, axis=1) #- commission

    holdings = pd.DataFrame(np.zeros(prices.shape), index=prices.index, columns=prices.columns)

    holdings["cash"].iloc[0] += start_val

    holdings.iloc[:,:] += trades.iloc[:,:]

    holdings = (np.cumsum(holdings,axis=0))

    values = prices * holdings

    values["cash"] -= orders_df.shape[0] * commission

    df_portval  = (values.sum(axis=1))

    return df_portval

def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders-short.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"
    
    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2008,6,1)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2,0.01,0.02,1.5]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2,0.01,0.02,1.5]

    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

if __name__ == "__main__":
    test_code()
