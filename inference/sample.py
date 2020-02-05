import numpy as np
import json
import copy
from tqdm import tqdm


def pick_row_randomly(array):
    _, N = array.shape
    row = np.zeros(N)
    for i in range(N):
        array_i = array[:, i]
        array_i_wo_outlyers = \
            array_i[abs(array_i - np.mean(array_i))
                    < 1.2 * np.std(array_i)]
        row[i] = np.random.choice(array_i_wo_outlyers)
    return row


class StockSampler():
    """Class implementing the sampling algorithm."""

    def __init__(self, prices_array, tickers, dates, m):
        """
        Initializes the sampler with all the parameters needed and a level of discretization given by m
        """
        self.prices_array = prices_array
        self.tickers = tickers
        self.dates = dates
        self.n, self.N = prices_array.shape
        self.m = m
        self.x = self._define_x()
        self.eta = 1*self.x
        self.R = 0*self.x

    def _define_x(self):
        """
        Create the matrix x as specified in the model
        """
        prices_array = self.prices_array
        insertion_rows = [i for i in range(
            1, self.n) for _ in range(self.m-1)]
        x = np.insert(prices_array, insertion_rows, 0, axis=0)
        return x

    def init_shape(self, n_iter, observation_limit):
        """
        Initialize the shape of the different parameters for n_iter iterations
        """
        self.b2 = np.zeros((n_iter, self.N))
        self.theta = np.zeros((n_iter, self.N))
        nbis, N = self.eta.shape
        self.etas = np.zeros((n_iter, nbis, N))
        self.Rs = np.zeros((n_iter, nbis, N))
        self.x = self.x[:(observation_limit-1)*self.m + 1]

    def init_value(self, m0, s02, a0, b0):
        """
        Initialize the value of the different parameters
        """
        self.b2[0] = 1/np.random.gamma(a0, 1/b0, self.N)
        self.theta[0] = np.random.normal(
            m0, np.sqrt(s02), size=self.N)

    def sample(self, n_iter=100, m0=0, s02=0.01, a0=4, b0=1, observation_limit=None):
        """
        Sample the different parameters for n_iter iterations
        """
        if not observation_limit:
            observation_limit = self.n
        if observation_limit > self.n:
            observation_limit = self.n
        self.init_shape(n_iter, observation_limit)
        self.init_value(m0, s02, a0, b0)
        self.beta = []
        tau_0 = np.full(self.N, 1/s02)
        for l in tqdm(range(n_iter)):
            for j in range(observation_limit-1):
                for k in range(1, self.m):
                    self.eta[self.m*j + k] = (1 + self.theta[l] /
                                              self.m)*self.eta[self.m*j + k - 1]
                    mR = self.R[self.m*j + k-1] - (
                        self.R[self.m*j + k-1])/(self.m - k + 1)
                    sR2 = (self.m - k)/(self.m - k + 1) / \
                        self.m * self.R[self.m*j + k-1]**2
                    sR2 = sR2 * self.b2[l]
                    self.R[self.m*j +
                           k] = np.random.normal(mR, np.sqrt(sR2))
                    self.x[self.m*j + k] = self.eta[self.m *
                                                    j + k] + self.R[self.m*j + k]
            if l < n_iter - 1:
                mu_ijk = self.m * \
                    (self.x/np.roll(self.x, 1, axis=1) - 1)
                mu_ijk = np.delete(mu_ijk, 0, axis=0)
                mu_i = np.sum(mu_ijk, axis=0)/(self.n*self.m)
                tau_i = self.n / self.b2[l]/(self.n*self.m)
                beta_ijk = (self.x /
                            np.roll(self.x, 1, axis=1) - 1 - self.theta[l]/self.m)**2
                beta_ijk = np.delete(beta_ijk, 0, axis=0)
                beta_i_star = np.sum(beta_ijk, axis=0)
                self.beta.append(beta_i_star)
                self.theta[l+1] = np.random.normal(
                    (mu_i + m0)/(tau_i+tau_0), 1/(tau_i + tau_0))
                proposed_b2 = 1 / \
                    np.random.gamma(
                        a0 + 1/2, self.n*self.m/(b0 + beta_i_star))
                self.b2[l+1] = 1*self.b2[l]
                for i in range(self.N):
                    if proposed_b2[i] < 1:
                        self.b2[l+1][i] = proposed_b2[i]
            self.etas[l] = 1*self.eta
            self.Rs[l] = 1*self.R

    def predict(self, n_predict=100, x0=None, param_predictor=pick_row_randomly):
        theta = param_predictor(self.theta)
        b2 = param_predictor(self.b2)
        x = np.zeros((n_predict, self.N))
        if type(x0) != np.ndarray:
            x0 = self.prices_array[-1]
        x[0] = x0
        for j in range(1, n_predict):
            w = np.random.normal(loc=0.0, scale=1.0, size=(self.N))
            x[j] = x[j-1]*(1 + theta + np.sqrt(b2)*w)
        return x

    def predict_samples(self, n_sample=1000, n_predict=100, x0=None, param_predictor=pick_row_randomly):
        predicted_samples = np.zeros((n_sample, n_predict, self.N))
        if type(x0) != np.ndarray:
            x0 = self.prices_array[-1]
        for l in tqdm(range(n_sample)):
            sample = self.predict(n_predict, x0, param_predictor)
            predicted_samples[l] = sample
        self.predicted_samples = predicted_samples

    def save_to_json(self, file_path="samples.json"):
        """
        Prints the parameters sampled into a json file at file path filepath
        """
        to_save = {
            'x': (self.etas + self.Rs).tolist(),
            'params': {
                'theta': self.theta.tolist(),
                'b2': self.b2.tolist()
            },
            'tickers': self.tickers,
            'dates': self.dates,
            'prices_array': self.prices_array.tolist()
        }
        with open(file_path, 'w') as out_file:
            json.dump(to_save, out_file)
        print("Samples saved in {}".format(file_path))
