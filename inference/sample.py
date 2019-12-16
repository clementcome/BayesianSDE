import numpy as np
import copy

class StockSampler():
    """Class implementing the sampling algortithm."""
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
        insertion_rows = [i for i in range(1,self.n) for _ in range(self.m-1)]
        x = np.insert(prices_array, insertion_rows, 0, axis=0)
        return x
    
    def init_shape(self, n_iter):
        """
        Initialize the shape of the different parameters for n_iter iterations
        """
        self.b2 = np.zeros((n_iter, self.N))
        self.theta = np.zeros(n_iter)
        nbis, N = self.eta.shape
        self.etas = np.zeros((n_iter, nbis, N))
        self.Rs = np.zeros((n_iter, nbis, N))
    
    def init_value(self, m0, s02, a0, b0):
        """
        Initialize the value of the different parameters
        """
        self.b2[0] = np.random.gamma(a0, 1/b0, self.N)
        self.theta[0] = np.random.normal(m0, np.sqrt(s02))
    
    def sample(self, n_iter = 100, m0 = 0, s02 = 0.01, a0 = 4, b0 = 1):
        """
        Sample the different parameters for n_iter iterations
        """
        self.init_shape(n_iter)
        self.init_value(m0, s02, a0, b0)
        self.mu = []
        for l in range(n_iter):
            for j in range(self.n-1):
                for k in range(1,self.m):
                    self.eta[self.m*j + k] = (1+ self.theta[l]/self.m)*self.eta[self.m*j + k -1]
                    mR = self.R[self.m*j + k-1] + (self.x[self.m*(j+1)] - self.R[self.m*j + k-1])/(self.m - k +1)
                    sR2 = (self.m - k)/(self.m - k + 1)/self.m *self.R[self.m*j +k-1]**2
                    sR2 = sR2* self.b2[l]
                    self.R[self.m*j +k] = np.random.normal(mR, np.sqrt(sR2))
                    self.x[self.m*j +k] = self.eta[self.m*j +k] + self.R[self.m*j +k]
            if l < n_iter -1:
                mu_ijk = self.m*(self.x/np.roll(self.x, 1, axis=1) -1)
                mu_ijk = np.delete(mu_ijk, 0, axis=0)
                mu_i = np.sum(mu_ijk, axis=0)/(self.n*self.m)
                tau_i = self.n/ self.b2[l]
                tau_star = (np.sum(tau_i) + 1/s02)/(self.N+1)
                mu_star = (np.sum(mu_i*tau_i) + m0/s02)/(self.N+1)/tau_star
                beta_ijk = (self.x/np.roll(self.x, 1, axis=1) -1 - self.theta[l]/self.m)**2
                beta_ijk = np.delete(beta_ijk, 0, axis=0)
                beta_i_star = np.sum(beta_ijk, axis=0)
                self.mu.append(mu_ijk)
                self.theta[l+1] = np.random.normal(mu_star, np.sqrt(1/tau_star))
                self.b2[l+1] = 1/np.random.gamma(a0 + self.n*self.m/2, 1/(b0 + beta_i_star))
            self.etas[l] = 1*self.eta 
            self.Rs[l] = 1*self.R 

    def save_to_json(self, file_path):
        """
        Prints the parameters sampled into a json file at file path filepath
        """
        pass