# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 23:12:57 2023

@author: ericy
"""

### plot NaCs ac stark map

import numpy
from matplotlib import pyplot
import diatomic.hamiltonian as hamiltonian
from diatomic.constants import Rb87Cs133
from numpy.linalg import eigh
from scipy import constants
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
import scipy.constants
import diatomic.calculate as calculate
h = constants.h
pi = numpy.pi

h = scipy.constants.h
muN = scipy.constants.physical_constants['nuclear magneton'][0]
bohr = scipy.constants.physical_constants['Bohr radius'][0]
eps0 = scipy.constants.epsilon_0
c = scipy.constants.c
DebyeSI = 3.33564e-30

Na23Cs133 = {"I1":1.5,
            "I2":3.5,
            "d0":4.6*DebyeSI,
            "binding":114268135.25e6*h*0, ### need to figure out the true value
            "Brot":3471.295e6*h/2,
            # "Brot":1735.65937e6*h,   #fit
            #"Brot":1750.74133e6*h,
            "Drot":0*h,
            #"Q1":-50e3*h,
            "Q1":-85e3*h,       #fit   -97
            "Q2":155e3*h,       #fit   150
            "C1":14.2*h,            
            "C2":854.5*h,
            "C3":105.6*h,
            "C4":3941.8*h,      #fit    3941.8
            "MuN":0.0066*muN,   #fit   0.0073  
            "Mu1":1.478*(1-639.2e-6)*muN, 
            "Mu2":0.738*(1-6278.7e-6)*muN,
            # "Mu1":1.478*muN,
            # "Mu2":0.738*muN,
            "a0":41 * 10 ** (-4) * h * 2 * eps0 * c, #1064nm
            "a2":40 * 10 ** (-4) * h * 2 * eps0 * c, #1064nm
            "Beta":0}



Consts =Na23Cs133
Nmax = 2
Consts['a0'] = 0 #Looking at transition energy so set isotropic component to zero
Consts['Beta'] = pi * 70/180


def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''

    points = numpy.array([x, y]).T.reshape(-1, 1, 2)
    segments = numpy.concatenate([points[:-1], points[1:]], axis=1)
    
    return segments


def colorline(x, y, z=None, cmap=pyplot.get_cmap('copper'), norm=pyplot.Normalize(0.0, 1.0), linewidth=3, alpha=1.0,legend=False):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''
    
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = numpy.linspace(0.0, 1.0, len(x))
           
    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = numpy.array([z])
        
    z = numpy.asarray(z)
    
    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth)
    
    ax = pyplot.gca()
    ax.add_collection(lc)
    
    return lc


def transition_calculation(energies,states,gs,Nmax,I1,I2,TDMs=None,
            pm = +1, Offset=0,fig=pyplot.gcf(),
            log=False,minf=None,maxf=None,prefactor=1e-6,col=None):
    if TDMs == None and (Nmax == None or I1 == None or  I2 == None):
        raise RuntimeError("TDMs  or Quantum numbers must be supplied")

    elif (Nmax == None or I1 == None or  I2 == None):
        TDMs = numpy.array(TDMs)
        dm = TDMs[0,:]
        dz = TDMs[1,:]
        dp = TDMs[2,:]
    elif TDMs == None:
        dm = numpy.round(calculate.transition_dipole_moment(Nmax,I1,I2,+1,states,gs),6)
        dz = numpy.round(calculate.transition_dipole_moment(Nmax,I1,I2,0,states,gs),6)
        dp = numpy.round(calculate.transition_dipole_moment(Nmax,I1,I2,-1,states,gs),6)

    if abs(pm)>1:
        pm = int(pm/abs(pm))




    N,MN = calculate.label_states_N_MN(states,Nmax,I1,I2)
    #find the ground state that the user has put in

    N0 = N[gs]

    energies = energies-energies[gs]*0
    lim =10

    l1 = numpy.where(N==N0)[0]
    l2 = numpy.where(N==N0+pm)[0]


    if minf == None:

        emin = numpy.amin(energies[l2])
        minf = prefactor*(emin)/h - Offset

    if maxf == None:

        emax = numpy.amax(energies[l2])
        maxf = prefactor*(emax)/h - Offset

    sig_m_tf = []
    sig_p_tf = []
    pi_tf = []      
    #plotting for excited state
    for l in l2:
        f = prefactor*(energies[l])/h - Offset
        if dz[l]!=0 :
            pi_tf.append(f+Offset)
        elif dp[l] !=0 :
            sig_p_tf.append(f+Offset)
        elif dm[l] !=0:
            sig_m_tf.append(f+Offset)
    f = prefactor*(energies)/h-Offset
    dipole_p = []
    dipole_m = []
    dipole_z = []
    state_sigma_p = []
    state_pi = []
    min_d = numpy.sqrt(0.1)
    for j,d in enumerate(dz):
        #this block of code sort the dipole moments (or transition strengths
        if j<max(l2) and f[j]<maxf:
            if abs(d)>min_d and j<max(l2):
                d_z = numpy.array([f[j]+Offset,numpy.abs(d)])
                dipole_z.append(d_z)
                state_pi.append([d_z[0],d_z[1],states[:,j]])
            d=dp[j]
            if abs(d)>min_d and j<max(l2):
 
                d_p = numpy.array([f[j]+Offset,numpy.abs(d)])
                dipole_p.append(d_p)
                state_sigma_p.append([d_p[0],d_p[1],states[:,j]])
            d=dm[j]
            if abs(d)>min_d and j<max(l2):
                d_m = numpy.array([f[j]+Offset,numpy.abs(d)])
                dipole_m.append(d_m)
            
    dipole_m = numpy.array(dipole_m)
    dipole_p = numpy.array(dipole_p)
    dipole_z = numpy.array(dipole_z)            
    return sig_m_tf,sig_p_tf,pi_tf,dipole_m,dipole_p,dipole_z,state_sigma_p,state_pi

print("Building Hamiltonian...")
H0,Hz,Hdc,Hac = hamiltonian.build_hamiltonians(Nmax,Consts,zeeman=True,Edc=True,ac=True)
intensity_max = 1e7
intensity_min = 0e7
# I = intensity_max
I = numpy.linspace(intensity_min, intensity_max, 100)
E = 0
B =864e-4 #Set magnetic field range here

H = H0[..., None]+\
    Hz[..., None]*B+\
    Hdc[..., None]*E+\
    Hac[..., None]*I 
H = H.transpose(2,0,1)
# H = H0+Hz*B+Hdc*E+Hac*I 
print("Diagonalizing Hamiltonian...")
energies, states = eigh(H)

#Plot the figure
# pyplot.figure(figsize=(5,4), dpi=600)
for i in range(numpy.shape(energies)[1]):
    #colour each line as base grey
    # pyplot.plot(I/1e7, (energies[:,i]-energies[:,2])/(1e6*h), linestyle='solid', color='lightgray', zorder=0)
    
    pyplot.plot(I/1e7, (energies[:,i])/(1e6*h), linestyle='solid', color='lightgray', zorder=0)
    #add colour on top to indicate the component that has N=1, MN=1, IRb=3/2, ICs=7/2
    # cl=colorline(I/1e7,(energies[:,i]-numpy.amin(energies))/(1e6*h),z=abs(numpy.real(states[:,32,i])),norm=LogNorm(vmin=1e-2,vmax=1),linewidth=2.0)
# H0,Hz,Hdc,Hac = hamiltonian.build_hamiltonians(Nmax,Consts,zeeman=True,Edc=True,ac=True)

# intensity_max = 35e7

I = numpy.linspace(intensity_min, intensity_max, 30)
for I_i in I:
    H0,Hz,Hdc,Hac = hamiltonian.build_hamiltonians(Nmax,Consts,zeeman=True,Edc=True,ac=True)
    H = H0+Hz*B+Hdc*E+Hac*I_i
    energies, states = eigh(H)
    sigma_m_tf,sigma_p_tf,pi_tf,D_m,D_p,D_z,state_sigma_p,state_pi = transition_calculation(energies,states,2,Nmax,Na23Cs133['I1'],Na23Cs133['I2'],Offset=3471.295,prefactor=1e-6, maxf=0.8)
    # print(D_z)
    for p_i in D_p[:,0] :
        pyplot.plot(I_i/1e7,p_i, 'ro',markersize=1,label = "sigma + transition")
    for m_i in  D_m[:,0] :
        pyplot.plot(I_i/1e7,m_i, 'go',markersize=1,label = "sigma - transition")
    for pi_i in D_z[:,0]:
        pyplot.plot(I_i/1e7,pi_i, 'bo',markersize=1,label = "pi transition")

pyplot.ylim(3469.105,3469.11)
pyplot.xlim(intensity_min/1e7, intensity_max/1e7)
pyplot.xlabel("Laser intensity (kW cm$^{-2}$)")
pyplot.ylabel("Transition energy from $N=0, M_F=3$,  $E$ / $h$ (MHz)")
# pyplot.legend()
pyplot.show()
