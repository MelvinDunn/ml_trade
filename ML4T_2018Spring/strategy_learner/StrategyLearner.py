"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch
"""

import datetime as dt
import pandas as pd
import util as ut
import random
import numpy as np
from QLearner import QLearner

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False, impact=0.0, save_data=True):
        self.verbose = verbose
        self.impact = impact
        self.nbins = 5
        self.save_data = save_data
    def author(self):
        return "mdunn34"


    def bollinger(self, prices, window_n=20, k=2):
        std = prices.rolling(window=window_n).std()
        mean = prices.rolling(window=window_n).mean()
        upper_bound = mean + (k * std)
        lower_bound = mean - (k * std)
        band = (prices-lower_bound)/(upper_bound-lower_bound)
        normed_band = (band-(band).mean())/(band.std())
        bandwidth = (upper_bound - lower_bound) / mean * 100
        normed_band = normed_band.fillna(method="bfill")
        bandwidth = bandwidth.fillna(method="bfill")
        return normed_band, bandwidth

    def add_indicators(self, prices, window_n=5, n_bins=5, save_data=False, symb=None):
        n_bins = self.nbins
        prices_new = prices.copy()

        def plot_data(df, type, title="Stock prices", xlabel="Date", ylabel="Value", symb=None):
            """Plot stock prices with a custom title and meaningful axis labels."""
            ax = df.plot(title=title, fontsize=12)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            plt.savefig('{}_{}.png'.format(type, str(symb)))

        for symb in prices.columns:
            #bollinger bands here.
            bolli, bandwidth = self.bollinger(prices[symb], window_n=window_n)

            momentum = ((prices[symb] / prices[symb].shift(-1)) * 100).fillna(method="ffill")


            # rsi
            difference = prices[symb].diff()
            diff_up, diff_down = difference.copy(), difference.copy()
            diff_up[diff_up < 0] = 0
            diff_down[diff_down > 0] = 0
            rsi =  100. - (100. / (1. + (diff_up.rolling(window=window_n).mean() / diff_down.rolling(window=window_n).mean().abs())))
            rsi = rsi.fillna(method="bfill")

            if save_data == True:
                print("Correlation between RSI and Momentum", np.corrcoef(rsi, momentum))

                plot_data(bandwidth, type='bandwidth', title='Bandwidth Values')
                plt.clf()
                plot_data(bolli, type='bollinger', title='Normed Bollinger Values')
                plt.clf()
                plot_data(momentum, type='momentum', title='Momentum Values')
                plt.clf()
                plot_data(rsi, type='rsi', title='RSI Values')
                plt.clf()

            bolli = pd.cut(bolli, bins=n_bins, labels=False)
            bandwidth = pd.cut(bandwidth, bins=n_bins, labels=False)
            momentum = pd.cut(momentum, bins=n_bins, labels=False)
            rsi = pd.cut(rsi, bins=n_bins, labels=False)
        return np.asarray(bolli), np.asarray(bandwidth), np.asarray(momentum), np.asarray(rsi)

    def discretize(self, state_bolli, state_bandwidth, state_momentum, state_rsi):
        return (state_bolli) + (state_bandwidth ** 2) + (state_momentum ** 3) + (state_rsi ** 4)

    def init_trades_and_prices_df(self, prices):
        prices_dataframe = prices.copy()
        prices_dataframe['Cash'] = np.ones(prices.shape[0])
        df_trades = pd.DataFrame(np.zeros(prices_dataframe.shape), index=prices_dataframe.index, columns=prices_dataframe.columns)
        return prices_dataframe, df_trades

    def compute_last_reward(self, prices, holdings, action, i):
        if i == 0:
            reward = 0
        else:
            # compute the daily return.
            daily_return = ((prices.iloc[i] - prices.iloc[i - 1]) / prices.iloc[i]) * 100
            if holdings != 0:
                # Sell
                if action == 0:
                    reward = -1.0 * daily_return
                # Buy
                elif action == 1:
                    reward = daily_return
                else:
                    reward = -9
            else:
                reward = 0
        return reward

    def compute_trades_and_holdings_buy(self, df_trades, prices_dataframe, symbol, holdings, diggity_day):
        if holdings == -1000 :
            df_trades["Cash"].loc[diggity_day] = df_trades["Cash"].loc[diggity_day] + 2000. * prices_dataframe[symbol].loc[diggity_day] * -1.
            df_trades[symbol].loc[diggity_day] = df_trades[symbol].loc[diggity_day] + 2000.
            holdings = holdings + 2000.
        if holdings == 0:
            df_trades["Cash"].loc[diggity_day] = df_trades["Cash"].loc[diggity_day] + 1000. * prices_dataframe[symbol].loc[diggity_day] * -1.
            df_trades[symbol].loc[diggity_day] = df_trades[symbol].loc[diggity_day] + 1000.
            holdings = holdings + 1000.
        return df_trades, holdings

    def compute_trades_and_holdings_sell(self, df_trades, prices_dataframe, symbol, holdings, diggity_day):
        if holdings == 1000:
            df_trades["Cash"].loc[diggity_day] = df_trades["Cash"].loc[diggity_day] + 2000.0 * prices_dataframe[symbol].loc[diggity_day]
            df_trades[symbol].loc[diggity_day] = df_trades[symbol].loc[diggity_day] - 2000.
            holdings = holdings - 2000
        if holdings == 0:
            df_trades["Cash"].loc[diggity_day] = df_trades["Cash"].loc[diggity_day] + 1000. * prices_dataframe[symbol].loc[diggity_day]
            df_trades[symbol].loc[diggity_day] = df_trades[symbol].loc[diggity_day] - 1000.0
            holdings = holdings - 1000.

        return df_trades, holdings

    def compute_cumulative_return(self, prices, prices_dataframe, df_trades, sv):
        holdings_df = pd.DataFrame(np.zeros(df_trades.shape), columns=df_trades.columns, index=df_trades.index)
        values_df = holdings_df.copy()
        first_diggity_day = prices.index[0]
        holdings_df["Cash"].loc[first_diggity_day] = sv
        holdings_df.loc[first_diggity_day] = holdings_df.loc[first_diggity_day] + df_trades.loc[first_diggity_day]
        holdings_df = holdings_df.cumsum()
        values_df = (holdings_df * prices_dataframe)
        df_port_val = values_df.sum(axis=1)
        cumulative_return = (df_port_val.iloc[-1] - sv) / sv
        return cumulative_return

    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 10000):

        # add your code to do learning here

        # example usage of the old backward compatible util function
        syms=[symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        if self.verbose: print prices

        # example use with new colname
        volume_all = ut.get_data(syms, dates, colname = "Volume")  # automatically adds SPY
        volume = volume_all[syms]  # only portfolio symbols
        volume_SPY = volume_all['SPY']  # only SPY, for comparison later
        if self.verbose: print volume

        # feature engineering, get your new features for the state space.
        state_bolli, state_bandwidth, state_momentum, state_rsi = self.add_indicators(prices, save_data=self.save_data, symb=symbol)

        # compute the current state, well really all the states.
        states = self.discretize(state_bolli, state_bandwidth, state_momentum, state_rsi)
        number_of_bins = self.nbins
        self.learner = QLearner(num_states=(number_of_bins ** 4), \
                                num_actions = 3, \
                                alpha = 0.5, \
                                gamma = 0.9, \
                                rar = 0.0, \
                                radr = 0.0, \
                                dyna = 0, \
                                verbose = False)

        converged = False
        count = 0
        converged_yet = 0.

        while (not converged) and (count < 30):

            total_reward = 0
            prices_dataframe, df_trades = self.init_trades_and_prices_df(prices)
            holdings = 0

            for i in range(state_bolli.shape[0]):
                current_state = states[i]
                action = self.learner.querysetstate(current_state)
                # first day
                reward = self.compute_last_reward(prices, holdings, action, i)
                # get the current state and the reward to get an action.
                action = self.learner.query(current_state,reward)
                diggity_day = prices.index[i]
                if action == 0:
                    df_trades, holdings = self.compute_trades_and_holdings_sell(df_trades, prices_dataframe, symbol, holdings, diggity_day)
                if action == 1:
                    df_trades, holdings = self.compute_trades_and_holdings_buy(df_trades, prices_dataframe, symbol, holdings, diggity_day)
                total_reward += reward

            cumulative_return = self.compute_cumulative_return(prices, prices_dataframe, df_trades, sv)
            count += 1

            if abs(converged_yet - cumulative_return) * 100. < 0.0001:
                converged = True
            else:
                converged_yet = cumulative_return

        return pd.DataFrame(df_trades[symbol])
    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 10000):

        # here we build a fake set of trades
        # your code should return the same sort of data
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        prices = prices_all[[symbol,]]  # only portfolio symbols
        trades_SPY = prices_all['SPY']  # only SPY, for comparison later
        """
        trades.values[:,:] = 0 # set them all to nothing
        trades.values[0,:] = 1000 # add a BUY at the start
        trades.values[40,:] = -1000 # add a SELL
        trades.values[41,:] = 1000 # add a BUY
        trades.values[60,:] = -2000 # go short from long
        trades.values[61,:] = 2000 # go long from short
        trades.values[-1,:] = -1000 #exit on the last day
        if self.verbose: print type(trades) # it better be a DataFrame!
        if self.verbose: print trades
        if self.verbose: print prices_all
        """
        prices_dataframe, df_trades = self.init_trades_and_prices_df(prices)
        holdings = 0
        state_bolli, state_bandwidth, state_momentum, state_rsi = self.add_indicators(prices)
        states = self.discretize(state_bolli, state_bandwidth, state_momentum, state_rsi)

        for i in range(state_bolli.shape[0]):
            current_state = states[i]
            action = self.learner.querysetstate(current_state)
            reward = self.compute_last_reward(prices, holdings, action, i)
            diggity_day = prices.index[i]
            if action == 0:
                df_trades, holdings = self.compute_trades_and_holdings_sell(df_trades, prices_dataframe, symbol, holdings, diggity_day)
            if action == 1:
                df_trades, holdings = self.compute_trades_and_holdings_buy(df_trades, prices_dataframe, symbol, holdings, diggity_day)
        return pd.DataFrame(df_trades[symbol])

if __name__=="__main__":
    print "One does not simply think up a strategy"
    learner = (StrategyLearner(save_data=True))
    print(learner.addEvidence())
    print(learner.testPolicy())
