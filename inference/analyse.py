import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class StockAnalysis():
    """This class aims at providing visualization and analysis tool for
    the results of sampling of SDE."""

    def __init__(self, prices_array, x, tickers, params, dates=None, augmented=True, m=None):
        """
        Creates an instance of stock analysis

        Parameters
        ----------
        prices_array : numpy.array
            Real values for stock prices (2D array)
        x : numpy.array
            Results of the sampling (3D array)
        tickers : list
            List of the tickers represented in the sample
        dates : list, optional
            List of the observations' dates
        augmented : bool, optional
            Wether x is based on an augmentation scheme (default to True)
            If True, some pre-processing will be done in order to give the same shape as prices_array
        m : integer necessary if augmented
            Level of data augmentation

        Raises
        ------
        ValueError
            If augmented == True and m == None
        """
        self.prices_array = prices_array
        self.x = x
        if augmented:
            if not(m):
                raise ValueError(
                    """If data is augmented, you should provide m, the level of discretization.
                    If data is not augmented, you should set augmented to False.""")
            self._deaugment_data(m)
        self.tickers = tickers
        self.dates = dates
        self.params = params

    def _deaugment_data(self, m):
        """
        Modifies self.x in order to be able to compare it with self.prices_array
        Modification is based on the last approximation made during the data_augmentation

        Parameters
        ----------
        m : integer
            Level of data augmentation
        """
        n_iter, _, _ = self.x.shape
        n, N = self.prices_array.shape
        new_x = np.full((n_iter, n, N), self.prices_array)
        for l in range(n_iter):
            for j in range(1, n):
                new_x[l][j] = self.x[l][m*j - 1]
        self.x = new_x

    def plot_quantiles(self, tickers_list=None):
        """
        Plots the quantiles of the observations from the samples for the tickers in tickers_list

        Parameters
        ----------
        tickers_list : list, optional
            tickers under observation
        """
        if not(tickers_list):
            tickers_list = self.tickers
        fig, ax = plt.subplots(nrows=len(tickers_list))
        for i in range(len(tickers_list)):
            ax[i].plot(np.percentile(self.x[:, :, i],
                                     [2.5, 97.5], axis=0).T, 'k', label="5% quantiles")
            ax[i].plot(self.prices_array[:, i], 'r', label='$x(t)$')
            ax[i].legend()
        fig.set_figheight(5*len(tickers_list))

    def plot_quantiles_percentage(self, tickers_list=None):
        """
        Plots the quantiles of the observations from the samples for the tickers in tickers_list
        minus the observations divided by the observations

        Parameters
        ----------
        tickers_list : list, optional
            tickers under observation
        """
        if not(tickers_list):
            tickers_list = self.tickers
        fig, ax = plt.subplots(nrows=len(tickers_list))
        for i in range(len(tickers_list)):
            ax[i].plot(np.percentile((self.x[:, :, i] - self.prices_array[:, i]) / self.prices_array[:, i],
                                     [2.5, 97.5], axis=0).T, 'k', label="5% quantiles percentage")
            # ax[i].plot(self.prices_array[:, i], 'r', label='$x(t)$')
            ax[i].legend()
        fig.set_figheight(5*len(tickers_list))

    def plot_parameter_distributions(self, tickers_list=None):
        """
        Plots the distributions of the parameters related to the stocks given in tickers_list

        Parameters
        ----------
        tickers_list : list
            tickers under observation
        """
        if not(tickers_list):
            tickers_list = self.tickers
        M = len(self.params)
        fig, axes = plt.subplots(ncols=M)
        keys = list(self.params.keys())
        for j in range(M):
            values = self.params[keys[j]]
            if len(values.shape) == 1:
                sns.distplot(values, ax=axes[j])
            else:
                N = values.shape[1]
                for i in range(N):
                    if self.tickers[i] in tickers_list:
                        sns.distplot(
                            values[:, i], ax=axes[j], label=str(i), hist=False)
            axes[j].set_title(keys[j])
            axes[j].legend()
