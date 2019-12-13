# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from inference.prepare import StockData
stockData = StockData()

#Données
data = pd.read_csv('C:/Users/Antoine/Desktop/Milan/stats/projet/historical_stock_prices.csv')
data['date'] = pd.to_datetime(data['date'])
data['delta_date'] = (data['date'] - np.datetime64('1970-01-02'))/np.timedelta64(1,'D')




def main (data,m,Niter) :
  
    
    stockData = StockData('C:/Users/Antoine/Desktop/Milan/stats/projet/historical_stock_prices.csv')
    x, tickers, dates = stockData.get_N_stocks(10)
    print("Shape of the output array is: {}".format(x.shape))
    print("Number of stocks observed is: {}".format(len(tickers)))
    print("Number of observations is: {}".format(len(dates)))
    print(x[:5])
    
    N= len(tickers)
    n=len(dates)
    nu= copy.deepcopy(x)
    R=0*x
    Dtau = 1/m
    b1= 1/(np.random.gamma(4,1))
    b=[[0]*Niter]*N
    for k in range (N):   #initialisation de b
        for l in range(Niter):
            b[k][l]=b1**(k+1)
    
    mu0=0
    theta1 = np.random.normal(mu0,0.01)
    theta = [theta1]*Niter
    
    
    for l in range (Niter):
        for j in range (n-2):
            for k in range (m-1):
                for i in range (N-1):
                    nu[m*j+k][i] = (1+b[l][i]*Dtau) * nu [m*j+k-1][i]
                    
                    muR= R[m*j+k-1][i] +Dtau*(-R[m*j+k-1][i])/(1+(k-1)/m)
                    sigmaR= (1+k/m)/(1+(k-1)/m)*Dtau*theta[l]**2*x[m*j+k-1][i]
                    
                    R[m*j+k][i]=np.random.normal(muR,sigmaR);
                
                x[m*j+k]= nu[m*j+k]+R[m+j+k];

        mu=0
        beta=0
        sigma=b[l]**2/Dtau/N/n/m
        for i in range (l):
            beta += (x[i][j]/x[i][j]-(1+theta*Dtau))**2
            for j in range (N):
               mu+= (x[i][j]/x[i][j]-1)/Dtau
               
        theta[l+1]= np.random.normal(0.01/(sigma+0.01)*mu + sigma/(sigma+0.01)*mu , (1/0.01+1/sigma)**(-1))
        b[l+1]= 1/(np.random.gamma(4+n*m/2,1+beta))
        
    X=[k for k in range (1,Niter+1)]
    plt.plot(X,theta,X,b)
    plt.show()
    return 0




