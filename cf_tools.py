
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

'''
class cf_tools(self, rmin=140, rmax=160, binsize=5):

    def __init__(self):
        self.r=make_xibins(rmin,rmax,binsize)
'''


def make_xibins(rmin, rmax, binsize):
    n=int((rmax-rmin)/binsize)
    return np.linspace(rmin, rmax, n)


def multipoles(mu, xi):

    '''
    Find redshift space correlation function multipoles for mocks
    '''
#     dmu = np.gradient(mu)[0]
    dmu = mu[1]-mu[0]
    xi0 = np.sum(xi*dmu, axis=0)
    xi2 = xi.T.dot(2.5*(3*mu**2-1))*dmu
    xi4 = xi.T.dot(1.125*(35*mu**4-30*mu**2+3)*dmu)
    
    return xi0, xi2, xi4


def multipoles_trapz(mu, xi):

    '''
    Find redshift space correlation function multipoles for mocks
    '''

#     dmu=np.gradient(mu)[0]
    dmu = mu[1]-mu[0]
    xi0l=xi.T
    xi2l=xi.T*2.5*(3*mu**2-1)
    xi4l=xi.T*1.125*(35*mu**4 - 30 * mu**2 + 3)
    
    xi0 = np.trapz(xi0l, x=mu, axis=1)
    xi2 = np.trapz(xi2l, x=mu, axis=1)
    xi4 = np.trapz(xi4l, x=mu, axis=1)

    return xi0, xi2, xi4


def multipoles_avg(path,mockmax,nmu,mu_max=1.,method=1):
    '''
    Finds the average multipole of 'mockmax' mocks in 'path'.
    method = 0 for squared integral
    method = 1 for trapezoidal integral
    '''
        
    os.chdir(path)
    mu_max=1
    nmu=100
    mulist = np.linspace(mu_max/nmu,mu_max,nmu)

    #initiate vectors
    temp=pd.read_csv('xi_rmu_{:03d}.csv'.format(1),header=None,index_col=False).to_numpy()
    n=len(temp[0])
    xi_0_avg=np.zeros(n) #monopole
    xi_2_avg=np.zeros(n) #quadrupole
    xi_4_avg=np.zeros(n) #hexadecapole

    for mock in range(mockmax):
        xi_allmu=pd.read_csv('xi_rmu_{:03d}.csv'.format(mock),header=None,index_col=False).to_numpy()
        
        if method==0:
            xi0, xi2, xi4 = multipoles(mulist,xi_allmu)
        elif method==1:
            xi0, xi2, xi4 = multipoles_trapz(mulist,xi_allmu)

        xi_0_avg+=xi0
        xi_2_avg+=xi2
        xi_4_avg+=xi4

    xi_0_avg/=mockmax
    xi_2_avg/=mockmax
    xi_4_avg/=mockmax

    return xi_0_avg, xi_2_avg, xi_4_avg


def quick_plot(x, y, label = ["r (Mpc/h)", r"$\xi(r)$"], leg = r"$\xi_0$" ):
    
    plt.style.use('classic')
    
    fig, ax = plt.subplots(figsize=(12,8))
    xmin, xmax = x.min(), x.max()
    
    plt.xlabel(label[0])
    plt.ylabel(label[1])
    
    hor = [0]*len(x) 
    ax.plot(x,hor, c = 'red',alpha = 0.7)
        
    ax.plot(x, y, 'v', c='orange', alpha=0.7, markersize = 7.9)
    ax.plot(x, y, '--', c='gray', label = leg)
    ax.legend(loc='best')
    
    plt.show()
    plt.close()
