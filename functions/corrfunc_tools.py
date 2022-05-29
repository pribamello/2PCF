799762import Corrfunc
import numpy as np
import ReadPinocchio as rp
import os
from astropy.cosmology import FlatLambdaCDM as FLCDM #needed for select_particles only! Only in the xyz coordinates also... (I think, I think...)

def select_particles(z_range, M_range, directory, coordinates = 'radec'):
    
    #set cosmology
    h=0.6774
    cosmo=FLCDM(100*h,0.3089)
    
    nplc = 64
    zmin, zmax = z_range
    Mmin, Mmax = M_range
    
    pos  = np.array([], dtype=(np.float64, 3)) #initiate position array
    pos2 = np.array([], dtype=(np.float64, 3)) #initiate position array
    
    os.chdir(directory)
    for i in range(nplc):
        lc = rp.plc("pinocchio.Miriam_{:03d}.plc.out.{}".format(1,i))
        sel = (lc.obsz >= zmin) & (lc.obsz <= zmax) & (lc.Mass > Mmin) & (lc.Mass <= Mmax) #here we set the mass and redshift search region

        #get polar coordinates
        rpos = np.array(cosmo.comoving_distance(lc.obsz[sel]).to_value()*h) #radial postion

        phipos   = lc.phi[sel]*np.pi/180
        thetapos = lc.theta[sel]*np.pi/180
        thetapos = np.pi/2-thetapos
        ra       = lc.phi[sel]
        dec      = lc.theta[sel]
        zobs     = lc.obsz[sel]

        #polar coordinates into (x,y,z) and radec coordinates
        stack  = np.array([rpos*np.sin(thetapos)*np.cos(phipos),rpos*np.sin(thetapos)*np.sin(phipos),rpos*np.cos(thetapos)]).T #.T é para transpor a matriz
        stack2 = np.array([zobs, ra, dec]).T
        #stack coordinates
        pos    = np.vstack((pos, stack))
        pos2   = np.vstack((pos2, stack2))
    
    if coordinates == 'xyz':
        
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
        
        return x, y, z
    
    elif coordinates == 'radec':
        cz, ra, dec = pos2[:,0],pos2[:, 1],pos2[:, 2]
        return cz, ra, dec



    
def make_logbins(rmin, rmax, nbins):
    return np.logspace(np.log10(rmin), np.log10(rmax), nbins+1)




def cf_smu(radecr1, radecr2, binfile, nthreads, mu_max = 1, nmu_bins = 100):
    ra, dec, cz = radecr1[:,0], radecr1[:,1], radecr1[:,2]
    rar, decr, czr = radecr2[:,0], radecr2[:,1], radecr2[:,2]
    N, Nr = ra.size, rar.size
    nbins = binfile.size - 1
    
    is_com    = False
    cosmology = 1

    # DD
    autocorr=1
    time0=time.time()
    resultsDD = DDsmu_mocks(autocorr = autocorr, cosmology = cosmology, nthreads = nthreads, mu_max = mu_max,
                           nmu_bins = nmu_bins, binfile = binfile, 
                           RA1 = ra, DEC1 = dec, CZ1 = cz, is_comoving_dist = is_com)
    DD_counts = np.array([], dtype=(np.float, 2))
    for r in resultsDD:
        DD_counts = np.vstack((DD_counts,np.array([r['mumax'],r['npairs']])))
    DD_time = time.time()- time0
    print("\t Tempo DD = ",DD_time,"s")

    # RR
    autocorr=1
    time0=time.time()
    resultsRR = DDsmu_mocks(autocorr = autocorr, cosmology = cosmology, nthreads = nthreads, mu_max = mu_max,
                           nmu_bins = nmu_bins, binfile = binfile, 
                           RA1 = rar, DEC1 = decr, CZ1 = czr, is_comoving_dist = is_com)
    RR_counts=np.array([], dtype=(np.float, 2))
    for r in resultsRR:
        RR_counts=np.vstack((RR_counts,np.array([r['mumax'],r['npairs']])))
    RR_time = time.time()- time0
    print("\t Tempo RR = ",RR_time,"s")

    # DR
    time0=time.time()
    autocorr=0 
    resultsDR=DDsmu_mocks(autocorr, cosmology, nthreads, mu_max, nmu_bins, binfile, ra, dec, cz, RA2=rar, DEC2=decr,CZ2=czr,is_comoving_dist = is_com)
    DR_counts=np.array([], dtype=(np.float, 2))
    for r in resultsDR:
        DR_counts=np.vstack((DR_counts,np.array([r['mumax'],r['npairs']])))
    DR_time = time.time()- time0
    print("\t Tempo DR = ",DR_time,"s")
    
    # cf
    cf=np.array([],dtype=(np.float, nbins))
    #print("i","|","cftemp","|","cf")
    for i in range(nmu_bins):
        cftemp = convert_3d_counts_to_cf(N, N, Nr, Nr,DD_counts[i::nmu_bins,1], DR_counts[i::nmu_bins,1],DR_counts[i::nmu_bins,1], RR_counts[i::nmu_bins,1])
        #print(i,'\t',cftemp[:5])
        #print(i,"|",np.shape(cftemp),'|',np.shape(cf))
        cf = np.vstack((cf,np.array(cftemp)))

    mubin = DD_counts[:nmu_bins,0]
    
    return mubin, cf