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
    array, tickers, dates = stockData.get_N_stocks(5)
    print("Shape of the output array is: {}".format(array.shape))
    print("Number of stocks observed is: {}".format(len(tickers)))
    print("Number of observations is: {}".format(len(dates)))
    
    
    N= len(tickers)
    n=len(dates)
    insertion_rows = [i for i in range(1,n) for _ in range(m-1)]
    x = np.insert(array, insertion_rows, 0, axis=0)
    eta= 1*x
    R= 0*x
    Dtau = 1/m
    
    #initialisation de b
    b1= 1/(np.random.gamma(4,1))
    b= np.zeros((Niter,N))
    for k in range (Niter):   
        for l in range(N):
            b[k][l]=b1**(l+1)
    print("test)")
    #initialisation theta
    mu0=0
    theta = np.zeros(Niter)
    theta[0]=np.random.normal(mu0,0.01)
    
    for l in range (Niter):
        for j in range (n-2):
            for k in range (m-1):
                
                eta[m*j+k] = (1+theta[l]*Dtau) * eta[m*j+k-1]
                                        
                muR= R[m*j+k-1] -R[m*j+k-1]/(m-k+1)
                sigmaR= (m-k)/(m-k+1)*Dtau * R[m*j+k-1]**2
                sigmaR = sigmaR*b[l]
                    
                R[m*j+k]=np.random.normal(muR,sigmaR);

                x[m+j+k]= eta[m+j+k]+R[m+j+k];
                

        if l < Niter -1:
            mu_i=0
            tau=0
            beta_i= np.zeros(N)
            mu_t=0
        
            #calcul de Beta*, tau* et muT
            for i in range (N):
                tau += 1/np.power(b[0][i],2)
                
                for j in range (n-1):
                    for k in range (1,m):
                        beta_i[i] += np.power(x[j+k][i]/x[j+k-1][i]-(1+theta[l]*Dtau),2)
                        
                        mu_i+= x[j+k][i]/x[j+k-1][i]
                        mu_i = mu_i/Dtau - (1/Dtau)*(n-1)*(m-1)
                        mu_t += mu_i*Dtau*n*m/np.power(b[0][i],2)
                        mu_i=0
                        tau = Dtau*n*m*tau
                        
                        theta[l+1]= np.random.normal(mu_t/tau, 1/np.sqrt(tau))
        
            # calcul de Beta*
            for i in range (N):
                b[l+1][i]= 1/(np.random.gamma(4+n*m/2,1/(1+beta_i[i])))
        

    return theta,b,x


