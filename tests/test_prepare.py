import pytest
from pathlib import Path
import pandas

from inference.prepare import StockData

@pytest.fixture
def N_stock():
    return 2

@pytest.fixture
def init_data():
    return StockData(Path('../daily-historical-stock-prices-1970-2018/historical_stock_prices_test.csv'))

@pytest.fixture
def chosed_tickers(init_data, N_stock):
    return init_data._choose_tickers(N_stock)

@pytest.fixture
def reshaped_df(init_data, chosed_tickers):
    return init_data._reshape_data_given_tickers(chosed_tickers)

@pytest.fixture
def array_N_stocks(init_data, reshaped_df):
    array = reshaped_df.drop(columns = ["date"]).to_numpy()
    return array

def test_get_data():
    StockData(Path('../daily-historical-stock-prices-1970-2018/historical_stock_prices_test.csv'))

def test_prepare_datetime(init_data):
    assert init_data.data_df.dtypes["date"] == '<M8[ns]'

def test_prepare_drop(init_data):
    assert (("adj_closed" in init_data.data_df.columns) == False)
    assert ("open" in init_data.data_df.columns)

def test_tickers_ordered(init_data):
    tickers_ordered = init_data.tickers_ordered
    df = init_data.data_df
    ticker1 = tickers_ordered[0]
    ticker2 = tickers_ordered[1]
    min_date1 = df[df["ticker"] == ticker1]['date'].min()
    min_date2 = df[df["ticker"] == ticker2]['date'].min()
    assert min_date1 <= min_date2

def test_choose_tickers(init_data):
    tickers = init_data._choose_tickers(5)
    assert len(tickers) == 5

def test_reshape_data(init_data, chosed_tickers):
    reshaped_data = init_data._reshape_data_given_tickers(chosed_tickers)
    assert len(reshaped_data.columns) == len(chosed_tickers) +1 #A column for each ticker + a column for date
    assert (reshaped_data["date"] == reshaped_data["date"].sort_values()).all()

def test_convert_df_to_array(init_data, reshaped_df, chosed_tickers):
    array, tickers, dates = init_data._convert_df_to_array(reshaped_df)
    assert tickers == chosed_tickers
    assert len(array[0]) == len(tickers)
    assert dates == list(reshaped_df["date"])

def test_get_N_stocks(init_data, array_N_stocks, chosed_tickers, N_stock):
    array, tickers, _ = init_data.get_N_stocks(N_stock)
    assert (array == array_N_stocks).all()
    assert tickers == chosed_tickers 