import pytest
from pathlib import Path

from inference.sample import StockSampler
from inference.prepare import StockData

@pytest.fixture
def N_stock():
    return 2

@pytest.fixture
def m_sampling():
    return 2

@pytest.fixture
def data(N_stock):
    data = StockData(Path('../daily-historical-stock-prices-1970-2018/historical_stock_prices_test.csv'))
    array, tickers, dates = data.get_N_stocks(N_stock)
    return array, tickers, dates

@pytest.fixture
def sampler(m_sampling, data):
    array, tickers, dates = data
    sampler = StockSampler(array, tickers, dates, m_sampling)
    return sampler

def test_init_sampler(m_sampling, data):
    array, tickers, dates = data
    StockSampler(array, tickers, dates, m_sampling)

def test_x_creation(sampler, m_sampling):
    x = sampler.x
    assert (x[0] != 0).all()
    for i in range(1, m_sampling):
        assert (x[i] == 0).all()
    assert (x[m_sampling] != 0).all()
    assert (x[-1] != 0).all()
    assert x.shape == ((sampler.n -1)*m_sampling +1, sampler.N)