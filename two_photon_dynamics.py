#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 15:49:49 2022

@author: weijunyuan
"""

"""
Created on Sun Aug 21 21:21:28 2022

@author: ericy
"""

from numpy.linalg import eigh,inv,norm
import time
import numpy as np
from scipy.linalg import eig, inv
import matplotlib.pyplot as plt



from numpy.linalg import eigh
#import diatomic.plotting as plotting
import diatomic.hamiltonian as hamiltonian
from diatomic.constants import Rb87Cs133
import matplotlib.pyplot as pyplot
import scipy.constants
from scipy.optimize import brute,minimize,fmin,differential_evolution
from numpy import pi
import numpy
import diatomic.calculate as calculate
import time
import csv
from sympy.physics.wigner import wigner_3j






h = scipy.constants.h
muN = scipy.constants.physical_constants['nuclear magneton'][0]
bohr = scipy.constants.physical_constants['Bohr radius'][0]
eps0 = scipy.constants.epsilon_0
c = scipy.constants.c
DebyeSI = 3.33564e-30
MHz = 1e-6
Na23Cs133 = {"I1":1.5,
            "I2":3.5,
            "d0":4.6*DebyeSI,
            "binding":114268135.25e6*h*0, ### need to figure out the true value
            #"Brot":3471.295e6*h/2,
            "Brot":1735.65937e6*h,   #fit
            #"Brot":1750.74133e6*h,
            "Drot":0*h,
            #"Q1":-50e3*h,
            "Q1":-85e3*h,       #fit   -97
            "Q2":155e3*h,       #fit   150
            "C1":14.2*h,            
            "C2":854.5*h,
            "C3":105.6*h,
            "C4":3941.8*h,      #fit    3941.8
            "MuN":0.0065*muN,   #fit   0.0073  
            "Mu1":1.478*(1-639.2e-6)*muN, 
            "Mu2":0.738*(1-6278.7e-6)*muN,
            # "Mu1":1.478*muN,
            # "Mu2":0.738*muN,
            "a0": 0, #1064nm
            "a2": 0, #1064nm
            "Beta":0}



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

    energies = energies-energies[gs]
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
    for j,d in enumerate(dz):
        #this block of code sort the dipole moments (or transition strengths
        if j<max(l2) and f[j]<maxf:
            if abs(d)>0 and j<max(l2):
                d_z = numpy.array([f[j]+Offset,numpy.abs(d)])
                dipole_z.append(d_z)
                state_pi.append([d_z[0],d_z[1],states[:,j]])
            d=dp[j]
            if abs(d)>0 and j<max(l2):
 
                d_p = numpy.array([f[j]+Offset,numpy.abs(d)])
                dipole_p.append(d_p)
                state_sigma_p.append([d_p[0],d_p[1],states[:,j]])
            d=dm[j]
            if abs(d)>0 and j<max(l2):
                d_m = numpy.array([f[j]+Offset,numpy.abs(d)])
                dipole_m.append(d_m)
            
    dipole_m = numpy.array(dipole_m)
    dipole_p = numpy.array(dipole_p)
    dipole_z = numpy.array(dipole_z)            
    return sig_m_tf,sig_p_tf,pi_tf,dipole_m,dipole_p,dipole_z,state_sigma_p,state_pi




def state_calculator(H0,Hz,B,mF):
    
    
    if mF == 4:
        state_index = 1
    if mF == 5:
        state_index = 0
    #B = 40e-4 #T
    #B = 39.4e-4
    H = H0+Hz*B
    Nmax = 2
    e_freq = []
    energies, states = eigh(H)
    g_freq = energies[1]*MHz/h
    sigma_m_tf, sigma_p_tf,pi_tf,D_m,D_p,D_z,state_sigma_p,state_pi = transition_calculation(energies,states,state_index,Nmax,Na23Cs133['I1'],Na23Cs133['I2'],Offset=3471.295,prefactor=1e-6, maxf=0.8)
    
    if mF == 4: 
        print(D_p)
        ### e_freq: the transition frequence
        e_freq = [D_p[0,0],D_p[1,0],D_p[2,0]]
        D_list = [D_p[0,1],D_p[1,1],D_p[2,1]]
    if mF == 5:
        print(D_z)
        e_freq = [D_z[0,0],D_z[1,0],D_z[2,0]]
        D_list = [D_z[0,1],D_z[1,1],D_z[2,1]]
    return g_freq, e_freq, D_list


def five_level_H(D1,D2,D3,G11,G21,G31,G12,G22,G32):
    
    ###D1 : one photon detuning respect to e1
    ###D2 : one photon detuning respect to e2
    ###D3 : one photon detuning respect to e3
    ###G11 : coupling strength between g1 and e1
    ###G21 : coupling strength between g1 and e2
    ###G31 : coupling strength between g1 and e3
    ###G12 : coupling strength between g2 and e1
    ###G22 : coupling strength between g2 and e2
    ###G32 : coupling strength between g2 and e3
    H = np.array([[0,G11,G21,G31,0],[G11,D1,0,0,G12],[G21,0,D2,0,G22],[G31,0,0,D3,G32],[0,G12,G22,G32,0]])
    
    return H


def four_level_H(D1,D2,D3,G11,G21,G31):
    
    H = np.array([[0,G11,G21,G31],[G11,D1,0,0],[G21,0,D2,0],[G31,0,0,D3]])
    
    
    return H

def Unitary_e(S,D,t):
    exp_D = np.exp(-1j*D*t)
    #print(exp_D)
    Sinv = inv(S)
    #print(Sinv)
    U = np.matmul(S,np.matmul(exp_D,Sinv))
    return U
def f_3 (phi,H,t=0):
    new_phi = np.matmul(H,phi)
    fx = new_phi*(-1j)
    return fx

def four_level_dynamics_solver():

    a = 0     # starting time
    b = 0.001    # end time
    N = 10000  # time step
    h = (b-a)/N  
    t_list = np.linspace(a,b,N)
    
    Nmax=2
    H0,Hz,Hdc,Hac = \
        hamiltonian.build_hamiltonians(Nmax,Na23Cs133,zeeman=True,Edc=True,ac=True)
    
    g_f, e_f, D_list =state_calculator(H0,Hz,40e-4,4)
    
    
    
    level = 4
    B = 0

    D1 = -5000
    E = 100000
    E1 = e_f[0]/MHz
    E2 = e_f[1]/MHz
    E3 = e_f[2]/MHz
    
    D2 = D1-(E2-E1)*2*np.pi         ### assume red detuned
    D3 = D1-(E3-E1)*2*np.pi
    G11 = 10000
    #G11 = D_list[0]*Na23Cs133['I2']*E
    print(D_list[1])
    print(D_list[2])
    G21 = G11*D_list[1]/D_list[0]
    G31 = G11*D_list[2]/D_list[0]
      
    
    H = four_level_H(D1,D2,D3,G11,G21,G31)
    pop_g_list = []
    pop_e1_list = []
    pop_e2_list = []
    pop_e3_list = []
    total_pop = []
    
    tem_r = np.array([1,0,0,0],dtype = complex) # initial condition
    for nt in t_list:
        #multivariable Runge-Kutta method
        gg = tem_r[0]
        e1 = tem_r[1]
        e2 = tem_r[2]
        e3 = tem_r[3]
        pop_g_list.append(norm(gg)**2)
        pop_e1_list.append(norm(e1)**2)
        pop_e2_list.append(norm(e2)**2)
        pop_e3_list.append(norm(e3)**2)
        total_pop.append(norm(gg)**2+norm(e1)**2+norm(e2)**2+norm(e3)**2)
        k1 = h*f_3(tem_r,H,nt)
        k2 = h*f_3(tem_r+0.5*k1,H,nt+0.5*h)    
        k3 = h*f_3(tem_r+0.5*k2,H,nt+0.5*h)
        k4 = h*f_3(tem_r+k3,H,nt+h)
        tem_r +=(k1+2*k2+2*k3+k4)/6
    plt.plot(t_list,pop_g_list,"r-",label = "g")
    plt.plot(t_list,pop_e1_list,"b-",label = "e1")
    plt.plot(t_list,pop_e2_list,"y-",label = "e2")
    plt.plot(t_list,pop_e3_list,"k-",label = "e3")
    #plt.plot(t_list,total_pop)
    #plt.plot(t_list,rhogg_list,"b-",label = "rho_gg")
    # plt.plot(t_list,rhoge_list,"g-",label = "rho_ge")
    # plt.plot(t_list,rhoge_list,"y-",label = "rho_eg")
    plt.legend()
    plt.ylabel(r'population')
    plt.xlabel(r"t")
    plt.legend()
   
    
#four_level_dynamics_solver()

def five_level_dynamics_solver():
    a = 0     # starting time
    b = 0.0006    # end time
    N = 100000  # time step
    h = (b-a)/N  
    t_list = np.linspace(a,b,N)
    
    Nmax=2
    H0,Hz,Hdc,Hac = \
        hamiltonian.build_hamiltonians(Nmax,Na23Cs133,zeeman=True,Edc=True,ac=True)
    B = 80e-4
    g_f, e_f, D_list =state_calculator(H0,Hz,B,4)
    
    g_2f,e_2_f,D_list_2 = state_calculator(H0,Hz,B,5)
    
    level = 5
    
    #D1 = 0
    D1 = -30000*2*np.pi
    E = 100000
    E1 = e_f[0]/MHz
    E2 = e_f[1]/MHz
    E3 = e_f[2]/MHz
    
    D2 = D1-(E2-E1)*2*np.pi         ### assume red detuned
    D3 = D1-(E3-E1)*2*np.pi
    G11 = 5000*2*np.pi
    #G11 = D_list[0]*Na23Cs133['I2']*E
    
    print(e_f)
    print(e_2_f)
    G21 = G11*D_list[1]/D_list[0]
    G31 = G11*D_list[2]/D_list[0]
      
    G12 = 5000*2*np.pi
    G22 = G12*D_list_2[1]/D_list_2[0]
    G32 = G12*D_list_2[2]/D_list_2[0]
    
    
    
    H = five_level_H(D1,D2,D3,G11,G21,G31,G12,G22,G32)
    pop_g1_list = []
    pop_e1_list = []
    pop_e2_list = []
    pop_e3_list = []
    pop_g2_list = []
    total_pop = []
    
    tem_r = np.array([1,0,0,0,0],dtype = complex) # initial condition
    for nt in t_list:
        #multivariable Runge-Kutta method
        g1 = tem_r[0]
        e1 = tem_r[1]
        e2 = tem_r[2]
        e3 = tem_r[3]
        g2 = tem_r[4]
        pop_g1_list.append(norm(g1)**2)
        pop_e1_list.append(norm(e1)**2)
        pop_e2_list.append(norm(e2)**2)
        pop_e3_list.append(norm(e3)**2)
        pop_g2_list.append(norm(g2)**2)
        
        total_pop.append(norm(g1)**2+norm(e1)**2+norm(e2)**2+norm(e3)**2+norm(g2)**2)
        k1 = h*f_3(tem_r,H,nt)
        k2 = h*f_3(tem_r+0.5*k1,H,nt+0.5*h)    
        k3 = h*f_3(tem_r+0.5*k2,H,nt+0.5*h)
        k4 = h*f_3(tem_r+k3,H,nt+h)
        tem_r +=(k1+2*k2+2*k3+k4)/6
    plt.plot(t_list,pop_g1_list,"r-",label = "g")
    plt.plot(t_list,pop_e1_list,"b-",label = "e1")
    plt.plot(t_list,pop_e2_list,"y-",label = "e2")
    plt.plot(t_list,pop_e3_list,"k-",label = "e3")
    plt.plot(t_list,pop_g2_list,"grey",label = "g2")
    #plt.plot(t_list,total_pop)
    #plt.plot(t_list,rhogg_list,"b-",label = "rho_gg")
    # plt.plot(t_list,rhoge_list,"g-",label = "rho_ge")
    # plt.plot(t_list,rhoge_list,"y-",label = "rho_eg")
    plt.legend()
    plt.ylabel(r'population')
    plt.xlabel(r"t")
    plt.legend()
    
five_level_dynamics_solver()    
    
    







