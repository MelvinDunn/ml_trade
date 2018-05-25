"""MC1-P2: Optimize a portfolio.

Copyright 2017, Georgia Tech Research Corporation
Atlanta, Georgia 30332-0415
All Rights Reserved
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data
import scipy.optimize as sco
# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality

def get_portvals(allocs, prices):
    # all prices divided by first row
    normed = prices / prices.ix[0]

    # allocations times normed.
    alloced = normed * allocs

    # get the amount of cash allocated to each asset, 
    # then see the value of that asset over time.
    pos_vals = alloced * 100

    port_val =  pos_vals.sum(axis=1) # add code here to compute daily portfolio values
    return port_val

def daily_returns(port_val):

    daily_return = (port_val / port_val.shift(1)) - 1.
    
    daily_return.iloc[0] = 0.

    # daily returns subset, remove first day
    daily_return = daily_return[1:]
    return daily_return


def shiggity_sharpe(allocs, prices):
    port_val = get_portvals(allocs, prices)
    daily_return = daily_returns(port_val)
    sr = np.sqrt(252.) * (np.mean(daily_return - 0) / daily_return.std())
    return -sr


def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=['GOOG','AAPL','GLD','XOM'], gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # find the allocations for the optimal portfolio
    # note that the values here ARE NOT meant to be correct for a test case
    allocs = np.ones(len(syms)) / len(syms) # add code here to find the allocations
    
    port_val = get_portvals(allocs, prices)

    # Get daily portfolio value
    daily_return = daily_returns(port_val) # add code here to compute daily portfolio values

    conds = ({'type': 'ineq', 'fun': lambda x : 1 - np.sum(x)})
            # {'type': 'ineq', 'fun': lambda x : 0.9999999 + np.sum(x)})

    biggity_bounds = tuple([(0,1) for i in (allocs)])

    minnie_the_mooch = sco.minimize(shiggity_sharpe,
                              x0=allocs,
                              args=(prices),
                              tol=1e-6,
                              options={'maxiter':1000000, 'disp': True},
                              constraints = conds,
                              bounds = biggity_bounds)

    new_allocs = minnie_the_mooch.x

    print(new_allocs)

    port_val = get_portvals(new_allocs, prices)

    # Get daily portfolio value
    new_daily_return = daily_returns(port_val) # add code here to compute daily portfolio values
   
    sr = -1 * shiggity_sharpe(new_allocs, prices)

    # average daily return
    adr = new_daily_return.mean()

    # cumulative returns
    cr = (port_val[-1] / port_val[0]) - 1

    # standard deviation of daily returns, or volatility
    sddr = new_daily_return.std()

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        pass

    return new_allocs, cr, adr, sddr, sr

def test_code():
    # This function WILL NOT be called by the auto grader
    # Do not assume that any variables defined here are available to your function/code
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!

    start_date = dt.datetime(2009,1,1)
    end_date = dt.datetime(2010,1,1)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM', 'IBM']

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        gen_plot = False)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr

if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called
    test_code()
