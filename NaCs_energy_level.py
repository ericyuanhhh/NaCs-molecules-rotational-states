# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 16:36:00 2022

@author: Columbia
"""
'''Generates a simple Zeeman plot for RbCs as an example'''

import numpy
import matplotlib.pyplot as pyplot
import diatomic.hamiltonian as hamiltonian
from diatomic.constants import Rb87Cs133
from diatomic.calculate import label_states_I_MI
from scipy.constants import h
from numpy.linalg import eigh
import scipy.constants
from numpy import pi
import numpy as np

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
            #"Brot":3471.295e6*h/2,
            "Brot":1735.65937e6*h,   #fit
            #"Brot":1750.74133e6*h,
            "Drot":0*h,
            #"Q1":-50e3*h,
            "Q1":-97e3*h,       #fit   -97
            "Q2":150e3*h,       #fit   150
            "C1":14.2*h,            
            "C2":854.5*h,
            "C3":105.6*h,
            "C4":3941.8*h,      #fit    3941.8
            "MuN":0.0066*muN,   #fit   0.0073  
            "Mu1":1.478*(1-639.2e-6)*muN, 
            "Mu2":0.738*(1-6278.7e-6)*muN,
            # "Mu1":1.478*muN,
            # "Mu2":0.738*muN,
            "a0":41 * 10 ** (-4) * h, #1064nm
            "a2":40 * 10 ** (-4) * h, #1064nm
            "Beta":0}
Nmax=2
H0,Hz,Hdc,Hac = \
    hamiltonian.build_hamiltonians(Nmax,Rb87Cs133,zeeman=True,Edc=True,ac=True)


I = np.linspace(1, 100) * 10 ** (7) #W/m^2
E =  0#V/m
#B = numpy.linspace(40 ,864, int(60))*1e-4 #T
# B = np.linspace(200*1e-4,864*1e-4,40)
B = 864*1e-4
H = H0[..., None]+\
    Hz[..., None]*B+\
    Hdc[..., None]*E+\
    Hac[..., None]*I 
H = H.transpose(2,0,1)

energies, states = eigh(H)
pyplot.figure(figsize=(2,2), dpi=200)
pyplot.plot(I, energies*1e-6/h, color='k')
pyplot.ylim(-100,100)
# pyplot.xlim(200, 850)
pyplot.ylabel("Energy/h (MHz)")
pyplot.xlabel("Intensity (W/cm ** 2)")
pyplot.show()



N_0_basis = []
N_1_basis = []
N_2_basis = []
mI_na = [3/2,1/2,-1/2,-3/2]
mI_cs = [7/2,5/2,3/2,1/2,-1/2,-3/2,-5/2,-7/2]

N = 0
m_N = 0
for i in mI_na:
    for j in mI_cs:
        m_F = i+j+m_N
        label = [0,m_N,m_F,i,j]
        N_0_basis.append(label)

N = 1
m_N_list = [1,0,-1]
mF_5_count = 0
mF_4_count = 0
mF_3_count = 0
for m_N in m_N_list:
    for i in mI_na:
        for j in mI_cs:
            m_F = i+j+m_N
            label = [N,m_N,m_F,i,j]
            if m_F == 5:
                mF_5_count += 1
            if m_F == 4:
                mF_4_count += 1
            if m_F == 3:
                mF_3_count += 1
            N_1_basis.append(label)

print(mF_5_count)
print(mF_4_count)
print(mF_3_count)
N = 2
m_N_list = [2,1,0,-1,-2]
for m_N in m_N_list:
    for i in mI_na:
        for j in mI_cs:
            m_F = i+j+m_N
            label = [N,m_N,m_F,i,j]
            N_2_basis.append(label)
bare_basis = N_0_basis+N_1_basis+N_2_basis


# print(bare_basis.index([1,0,5,3/2,7/2]))
# # print the state as [N,m_N,m_F, m_I_na,m_I_cs])
# print(bare_basis[])
# print(bare_basis[40])

# print(bare_basis[1])











        
        
        