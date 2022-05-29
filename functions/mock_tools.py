import ReadPinocchio as rp

import numpy as np
import pandas as pd
import os

import random
from astropy.cosmology import FlatLambdaCDM as FLCDM
import scipy.interpolate as interp 

'''
Este código cálcula a densidade de halos como função de redshift
'''

def nz_mocks(path, nbins, mock, Mmin, Mmax, zmin=0.0, zmax=1.0, redshift_space = True, density = False):
    '''
    str  ,   path    :   path to mock
    int  ,   mock    :   mock number
    int  ,   nbins   :   number of bins or bins used to bin the redshift 
    float,   Mmin    :   Minimum mass threshold
    float,   Mmax    :   Maximum mass threshold
    float,   zmin    :   Minimum redshift
    float,   zmax    :   Maximum redshift
    bool, redshift_space: probes redshift space or real space
    bool,    density :   If False, the result will contain the number of samples in each bin. 
                        If True, the result is the value of the probability density function at the bin, 
                        normalized such that the integral over the range is 1. 
                        Note that the sum of the histogram values will not be equal to 1 unless bins of unity width are chosen; 
                        it is not a probability mass function.

    return: 
    np.array() --> bincenter
    np.array() --> redshift counts inside bin
    '''

    os.chdir(path)
    redshift= np.array([])
    nplc=64

    for i in range(nplc):
        lc = rp.plc("pinocchio.Miriam_{:03d}.plc.out.{}".format(mock,i))
        if redshift_space:
            sel = (lc.obsz >= zmin) & (lc.obsz <= zmax) & (lc.Mass > Mmin) & (lc.Mass <= Mmax)
            redshift=np.append(redshift, lc.obsz[sel])
        else:
            sel = (lc.redshift >= zmin) & (lc.redshift <= zmax) & (lc.Mass > Mmin) & (lc.Mass <= Mmax)
            redshift=np.append(redshift, lc.redshift[sel])
    
    counts, binedge = np.histogram(redshift,bins = nbins, density = density)
    bincenter = (binedge[1:]+binedge[:-1])/2
    return bincenter, counts



def nz_avg( nbins, nmocks, Mmin, Mmax, zmin = 0.0, zmax = 1.0, redshift_space = True, density = False):
    '''
    int  ,   nbins   :   number of bins or bins used to bin the redshift
    int  ,   nmocks  :   number of mocks to average
    float,   Mmin    :   Minimum mass threshold
    float,   Mmax    :   Maximum mass threshold
    float,   zmin    :   Minimum redshift
    float,   zmax    :   Maximum redshift
    bool, redshift_space: probes redshift space or real space
    bool,    density :   If False, the result will contain the number of samples in each bin. 
                         If True, the result is the value of the probability density function at the bin, 
                         normalized such that the integral over the range is 1. 
                         Note that the sum of the histogram values will not be equal to 1 unless bins of unity width are chosen; 
                         it is not a probability mass function.
    
    return: 
    np.array() : bincenter
    np.array() : redshift counts inside bin

    Obs: this function is customized to run ONLY at milliways computer at UFRJ.
    If your mocks are at any other location, change the plcdir str below.
    '''
    total = np.zeros(nbins)
    for mock in range(nmocks):
        plcdir="/home/hdd3/Miriam/{:03d}/plc".format(mock) #running from ssh
        os.chdir(plcdir)

        bincenter, counts = nz_mocks(plcdir, nbins, mock, Mmin, Mmax, zmin = zmin, zmax = zmax, redshift_space = redshift_space, density = density)
        total   += counts
        print('Mock ', mock+1, '/', nmocks)
    return bincenter, total/nmocks



def make_randoms(bins, counts, Nr,  coord = 'xyz'):
    '''
    Creates a random catalog of size Nr_new (close to Nr) that follows a redshift distribution given by (bins, counts).
    np.array(), bins    :    redshift bins used to sample zsample
    np.array(), counts  :    counts in each redshift bins of redshift distribution(doesn't need to be normalized)
    float     , zmin    :    minimum redshift probed (shoudn't need to include this, but haven't changed it yet)
    float     , zmax    :    maximum redshift probed (same as zmin)
    int       , Nr      :    size of random catalog created
    int       , Nsample :    number of points used to bin (bins, counts) histogram before interpolating it
    str       , kind    :    interpolation method used. See scipy.interp.interp1d for choices
    str       , coord   :    coordinate system. 'xyz' for cartesian 'ang' for angular on (1,1,1) cone basis. 
    
    Returns: (xr, yr, zr) or (z, ra, dec)
    The cartesian coordinate points for each particle, rotated so that they form a cone parallel to (1,1,1)

    WARNING: a cosmology is assumed in this function, but only for the cartesian coordinates!!!
    '''
    # set cosmology
    h=0.6774
    cosmo=FLCDM(100*h,0.3089)

    # binedge = np.append(bins-dbin/2, bins[-1]+dbin/2)
    dbin = np.gradient(bins)[0]
    norm = np.sum(counts)
    normalized = counts/norm
    total_por_bin = normalized*Nr

    random_z = np.array([])
    i = 0
    for el in total_por_bin:
        #I'm not quite sure about the line bellow...
        u = (np.random.uniform(0,1,int(round(el)))) #sorteamos a raiz cúbica p/ que seja uniforme no volume r^3 (inverse sample rule)
        binrand = u*dbin + bins[i]
        random_z = np.append(random_z, binrand)
        i+=1

    r=np.array(cosmo.comoving_distance(random_z).to_value()*h)
    
    #Nr_new is necessary as upon normalization of bincounts, a number of randoms different from Nr may be generated
    Nr_new = len(r) 
    cosap=np.cos(np.pi/4)#cosine of LC aperture
    phi = np.random.uniform(0,2*np.pi,Nr_new)
    costheta = np.random.uniform(1,cosap,Nr_new)
    
    #to cartesian coordinates
    xt = r*np.sqrt(1-costheta**2)*np.cos(phi)
    yt = r*np.sqrt(1-costheta**2)*np.sin(phi)
    zt = r*costheta
    #xt = r*np.sin(theta)*np.cos(phi)
    #yt = r*np.sin(theta)*np.sin(phi)
    #zt = r*np.cos(theta)
    
    #Now we rotate the cone axis so it is parallel to (1,1,1) using Rogrigues formula:
    xr=(np.sqrt(3)+3)*xt/6 + (np.sqrt(3)-3)*yt/6 + np.sqrt(3)*zt/3
    yr=(np.sqrt(3)-3)*xt/6 + (np.sqrt(3)+3)*yt/6 + np.sqrt(3)*zt/3
    zr=(zt-xt-yt)/np.sqrt(3)

    if coord == 'xyz':
        return xr, yr, zr

    elif coord == 'ang':
        '''
        tgphi   = yr/xr
        tgtheta = np.sqrt(xr**2 + yr**2)/zr
        theta_rot = np.arctan(tgtheta)
        phi_rot = np.arctan(tgphi)
        r = np.sqrt(xr**2 + yr**2 + zr**2)
        
        ra_r  = (np.pi/2 - theta_rot)*180/np.pi
        dec_r = phi_rot*180/np.pi
        '''

        costheta = np.random.uniform(cosap,0,Nr_new)
        ra_r  = phi*180/np.pi
        dec_r = np.arccos(costheta)*180/np.pi 
        
        # return r, ra_r, dec_r
        return random_z, ra_r, dec_r