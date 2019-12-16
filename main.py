from inference.prepare import StockData

stockData = StockData()
array, tickers, dates = stockData.get_N_stocks(100)
print("Shape of the output array is: {}".format(array.shape))
print("Number of stocks observed is: {}".format(len(tickers)))
print("Number of observations is: {}".format(len(dates)))
# print(array[:5])

from inference.sample import StockSampler

sampler = StockSampler(array, tickers, dates, m=6)
sampler.sample(5)