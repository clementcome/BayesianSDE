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
    eta= copy.deepcopy(x)
    R= np.zeros(((n-1)*m+1,N))
    Dtau = 1/m
    
    #initialisation de b
    b1= 1/(np.random.gamma(4,1))
    b= np.zeros((N,Niter))
    for k in range (Niter):   
        for l in range(N):
            b[k][l]=b1**(l+1)
    print("test)")
    #initialisation theta
    mu0=0
    theta = np.zeros((1,Niter))
    theta[0]=np.random.normal(mu0,0.01)
    
    
    for l in range (Niter):
        for j in range (n-2):
            for k in range (m-1):
                for i in range (N-1):
                    eta[m*j+k][i] = (1+b[l][i]*Dtau) * eta[m*j+k-1][i]
                                        
                    
                    muR= R[m+j+k-1][i] +Dtau*(-R[m+j+k-1][i])/(1+(k-1)/m)
                    sigmaR= (m+k)/(m+k-1)*Dtau*theta[l]**2*x[m+j+k-1][i]**2
                    
                    R[m+j+k][i]=np.random.normal(muR,sigmaR);
                
                x[m+j+k]= eta[m+j+k]+R[m+j+k];


        mui=0
        tau=0
        beta=0
        mut=0
         
        #calcul de tau* et muT
        for i in range (N):
            tau += 1/np.power(b[i],2)

            for j in range (n-1):
                for k in range (1,m):
                    
                    mui+= x[j+k][i]/x[j+k-1][i]
            mui = mui/Dtau - (1/Dtau)*(n-1)*(m-1)
            mut += mui*Dtau*n*m/np.power(b[i],2)
            mui=0
        tau = Dtau*n*m*tau  
        
        # calcul de Beta*
        for j in range (n-1):
            for k in range (1,m):
                beta += np.power(x[j+k][l]/x[j+k-1][l]-(1+theta[l]*Dtau)) #quel indice pour l ?
        
        
        theta[l+1]= np.random.normal(mut/tau, 1/tau)
        b[l+1]= 1/(np.random.gamma(4+n*m/2,1+beta))
        
    X=[k for k in range (1,Niter+1)]
    plt.plot(X,theta,X,b)
    plt.show()
    return 0


