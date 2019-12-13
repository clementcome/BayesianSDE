import pandas as pd
import numpy as np
import os

from pathlib import Path

class StockData():
    """Class that provides handling for stock prices data"""
    def __init__(self, path = Path("../daily-historical-stock-prices-1970-2018/historical_stock_prices.csv")):
        self.data_df = self._get_data(path)
        self.data_df = self._prepare_data(self.data_df)
        self.tickers_ordered = self._get_tickers_ordered(self.data_df)
    
    def _get_data(self, path):
        """
        Retrieves the data from the file found at path
        """
        data = pd.read_csv(path)
        return data
    
    def _prepare_data(self, df):
        """
        Transform the date column of the database in a usable datetime format and
        drops columns of the data base we will not use.
        """
        df['date'] = pd.to_datetime(df['date'])
        df = df[['ticker', 'open', 'date']]
        return df

    def _get_tickers_ordered(self, df):
        """
        Returns a list of tickers ordered by their minimum date representation in the data
        """
        df_min_date = df.groupby('ticker')['date'].min()
        df_min_date_sorted = df_min_date.sort_values()
        tickers_ordered = list(df_min_date_sorted.index)
        return tickers_ordered

    def _choose_tickers(self, N):
        """ 
        Returns the N tickers of interest to extract from the database optimizing the number of observations
        """
        tickers = self.tickers_ordered[:N]
        return tickers

    def _reshape_data_given_tickers(self, tickers_list):
        """
        Returns the joined dataframe of all the simultaneous observations of the stock prices 
        represented in the tickers_list.
        Rows are described by their date and each column corresponds to a stock described by its ticker
        """
        df = self.data_df
        df = df[df["ticker"].isin(tickers_list)]
        joined_df = df.pivot(index='date', columns='ticker', values='open').dropna()
        # dict_df = {
        #     ticker: df[df["ticker"] == ticker][["open", "date"]].rename(columns = {"open": ticker})
        # for ticker in tickers_list}
        # joined_df = dict_df[tickers_list[0]]
        # for ticker in tickers_list[1:]:
        #     joined_df = pd.merge(joined_df, dict_df[ticker], how="inner", on="date")
        # joined_df = joined_df.sort_values("date")
        return joined_df

    def _convert_df_to_array(self, df):
        """
        Returns:
        array np.array: the array of the stock prices observations,
        tickers [string]: list of the tickers represented in array columns
        dates [np.datetime64]: list of the dates of observation
        """
        dates = list(df.index)
        tickers = list(df.columns)
        array = df.to_numpy()
        return array, tickers, dates
    
    def get_N_stocks(self,N):
        """
        Function used to extract the simultaneous observations of N stocks.
        Returns:
        array np.array: the array of the stock prices observations,
        tickers [string]: list of the tickers represented in array columns
        dates [np.datetime64]: list of the dates of observation
        """
        tickers = self._choose_tickers(N)
        reshaped_df = self._reshape_data_given_tickers(tickers)
        return self._convert_df_to_array(reshaped_df)