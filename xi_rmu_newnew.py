# This code is used to find the \xi(r, \mu) correlation function. It creates the random catalog using the function make_randoms
# in mock_tools and depends on the galaxy density n(z) generated by the code run_nz_density. This code shall be used as the main code]
# for real space correlation function in the JPAS colaboration.

from __future__ import print_function
import ReadPinocchio as rp
import numpy as np
import pandas as pd
import os
import random
#from Corrfunc.theory.DD import DD
from Corrfunc.theory.DDsmu import DDsmu
from Corrfunc.utils import convert_3d_counts_to_cf
from astropy.cosmology import FlatLambdaCDM as FLCDM
import mock_tools as mt

import time
import matplotlib.pyplot as plt

#code specs
nthreads=32 #number of threads used
mockmax=50 #number of mocks used, in total there are 1000 mocks (starting at 0 and ending at 999)
nplc=64 #light cone files to read - 64 max for all light cone
mp=3 #Random catalog multiplier. Nrandom = mp*Nmocks. This means the random catalog will be mp times the size of the mock catalog.

#set correlation function range and bins
rmin=40  #Mpc/h
rmax=160 #Mpc/h
binsize=5 #Mpc/h
nbins=int(np.rint((rmax-rmin)/(binsize)))
rbins=np.linspace(rmin,rmax,nbins+1)
binfile=rbins

#redshift bin
zint=0.55 #bin centered at zint
dz=0.05 #interval {zint-dz,zint+dz}

#Mass bin
Mmin=10**13
Mmax=10**14

#set \mu bins 
nmu_bins = 100
mu_max = 1.0

#set cosmology
h=0.6774
cosmo=FLCDM(100*h,0.3089)
dmin2=cosmo.comoving_distance(zint-dz).value**2
dmax2=cosmo.comoving_distance(zint+dz).value**2

#Initiating time counter to run along program
totaltime=0
looptime=0

################## directories ##################
#plcdir="/home/hdd3/Miriam/{:03d}/plc".format(mock) #running from ssh
#plcdir='/home/prm/Documentos/JPAS/2PCF/mocks/plc' #running from Ubuntu notebook

#resultsdir='/home/prm/Documentos/JPAS/2PCF/resultstemp' #running from Ubuntu notebook
resultsdir="/home/pedromello/JPAS/correlacao/results/xirmu_nz_norecon_nonperiodic" #running from ssh

''' 
#running from wsl
homedir=os.getcwd()
os.chdir("xirmu_WSL")
resultsdir=os.getcwd()#set results dir
'''

rd = pd.read_csv('random_dist.csv') #Here we import the random catalog density

for mock in range(mockmax): #multiprocessing ou pymp para paralelizar
    plcdir="/home/hdd3/Miriam/{:03d}/plc".format(mock) #running from ssh
    #plcdir="C:/Users/User/OneDrive/Doutorado/JPAS/Mock/001/plc"#running from WSL
    #plcdir="/mnt/c/Users/User/OneDrive/Doutorado/JPAS/Mock/001/plc"#running from WSL

    print("Mock {:03d}/{}".format(mock+1,mockmax))
    print("Loop time: {}s".format(round(looptime,5)))
    print("Total time elapsed: {} h".format(round(totaltime/3600,5)))

    totaltime+=looptime
    start_time=time.time()

    print('No reconstruction')

    pos = np.array([], dtype=(np.float, 3)) #initiate position array
    redshift= np.array([])
    rall=np.array([])

    os.chdir(plcdir)
    for i in range(nplc):
        lc = rp.plc("pinocchio.Miriam_{:03d}.plc.out.{}".format(mock,i))
        sel = (lc.obsz >= zint-dz) & (lc.obsz <= zint+dz) & (lc.Mass > Mmin) & (lc.Mass <= Mmax) #here we set the mass and redshift search region
        
        #get polar coordinates
        rpos=np.array(cosmo.comoving_distance(lc.obsz[sel]).to_value()*h) #radial postion
        phipos=lc.phi[sel]*np.pi/180
        thetapos=lc.theta[sel]*np.pi/180
        thetapos=np.pi/2-thetapos
        #polar coordinates into (x,y,z)
        stack=np.array([rpos*np.sin(thetapos)*np.cos(phipos),rpos*np.sin(thetapos)*np.sin(phipos),rpos*np.cos(thetapos)]).T #.T é para transpor a matriz
        #stack coordinates
        pos = np.vstack((pos, stack))
        
        rall=np.append(rall,rpos)
        redshift= np.append(redshift, lc.obsz[sel]) #redshift observado (sem reconstrução)
        # redshift= np.append(redshift, lc.redshift[sel]) #########################################ATENÇÃO A ESSA LINHA!!!!!

    #rotation
    versor1 = np.array([1.0, 1.0, 1.0])/np.sqrt(3)
    # versor2 is taken as the cross product of versor1 with [0, 0, 1]
    versor2 = np.cross(versor1, [0, 0, 1])
    versor2 /= versor2.dot(versor2)**0.5
    # versor2 is the cross product of versor1 and versor2
    versor3 = np.cross(versor1, versor2)
    # Setting the change of basis matrix
    # M = inv(np.array([versor2, versor3, versor1]))
    M = np.array([versor2, versor3, versor1]) #inversa da inversa (inversa da transformação)
    # Finally the position on the plc basis
    pos_rot = pos.dot(M)

    x, y, z = pos_rot[:,0], pos_rot[:,1], pos_rot[:,2]
    N=len(x)


     ################## Creating randoms ##################
    Nr = N*mp
    xr, yr, zr = mt.make_randoms(rd['bincenter'],rd['counts'], Nr = Nr)


    ##################RR counts##################  
    autocorr=1
    time0=time.time()
    
    resultsRR=DDsmu(autocorr, nthreads, binfile, mu_max, nmu_bins,xr, yr, zr, periodic = False)
    
    RR_counts=np.array([], dtype=(np.float, 2))
    
    
    for r in resultsRR:
        RR_counts=np.vstack((RR_counts,np.array([r['mu_max'],r['npairs']])))
    
    RR_time = time.time()- time0
    print("\t Tempo = ",RR_time,"s")
    
    
    ##################DD counts##################
    autocorr=1
    time0=time.time()
    
    resultsDD=DDsmu(autocorr, nthreads, binfile, mu_max, nmu_bins,x,y,z, periodic = False)
    DD_counts=np.array([], dtype=(np.float, 2))
    for r in resultsDD:
        DD_counts=np.vstack((DD_counts,np.array([r['mu_max'],r['npairs']])))
    
    DD_time = time.time()- time0
    print("\t Tempo = ",DD_time,"s")
    
    
    
    
    ##################DR counts##################
    time0=time.time()
    
    autocorr=0 
    resultsDR=DDsmu(autocorr, nthreads, binfile, mu_max, nmu_bins,x,y,z,X2=xr,Y2=yr,Z2=zr, periodic = False)
    DR_counts=np.array([], dtype=(np.float, 2))
    for r in resultsDR:
        DR_counts=np.vstack((DR_counts,np.array([r['mu_max'],r['npairs']])))
    
    DR_time = time.time()- time0
    print("\t Tempo = ",DR_time,"s")
    
    
    cf=np.array([],dtype=(np.float, nbins))
    #print("i","|","cftemp","|","cf")
    for i in range(nmu_bins):
        cftemp = convert_3d_counts_to_cf(N, N, Nr, Nr,DD_counts[i::nmu_bins,1], DR_counts[i::nmu_bins,1],DR_counts[i::nmu_bins,1], RR_counts[i::nmu_bins,1])
        #print(i,'\t',cftemp[:5])
        #print(i,"|",np.shape(cftemp),'|',np.shape(cf))
        cf=np.vstack((cf,np.array(cftemp)))
        
    mubin=DD_counts[:nmu_bins,0]
    
    
    ############## Salvando função de correlação ################
    os.chdir(resultsdir)
    
    df=pd.DataFrame(cf)
    df.to_csv("xi_rmu_{:03d}.csv".format(mock),header=None,index=False)
    
    codeinfo=[[mock,zint,dz,Mmin,Mmax,N,Nr,nthreads,mu_max,nmu_bins,rmin,rmax,binsize]]

    if mock==0:
        with open("stats_run.csv", 'a') as stat:
            #here you open the ascii file
            np.savetxt(stat,np.array(codeinfo),newline='\n',header="mock, z, dz, Mmin, Mmax, Nmocks, Nrandoms, nthreads, mu_max, nmu_bins, rmin, rmax, binsize")
    
    else:
        with open("stats_run.csv", 'a') as stat:
                np.savetxt(stat,np.array(codeinfo),newline='\n')
    
    looptime = time.time()-start_time
    del(cf, cftemp,RR_counts,resultsRR,DD_counts,resultsDD,DR_counts,resultsDR) #clear variables