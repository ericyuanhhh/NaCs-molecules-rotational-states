#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 11:52:51 2022

@author: weijunyuan
"""
from numpy.linalg import eigh,inv,norm
import time
import numpy as np
from scipy.linalg import eig, inv
import matplotlib.pyplot as plt


def Unitary_e(S,D,t):
    exp_D = np.exp(-1j*D*t)
    #print(exp_D)
    Sinv = inv(S)
    #print(Sinv)
    U = np.matmul(S,np.matmul(exp_D,Sinv))
    
    return U
def two_level_H(D1,delta):
    
    H = np.array([[0,D1],[D1,delta]])
    
    return H

def main():
    
    level = 2
    H = two_level_H(10,20)
    t_start = 0
    t_stop = 1
    N = 10000
    t_list = np.linspace(t_start,t_stop,N)
    eValues,eVecs = eig(H)  
    print(eVecs)
    
    D = np.zeros((level,level))
    for i in range(0,len(eValues)):
        D[i,i] = eValues[i]
    
    phi = np.zeros(level)
    phi[0] = 1
    phi_list = []
    pop_g = [1]
    pop_e = [0]

    for t_i in t_list[1:]:
        Ue = Unitary_e(eVecs,D,t_i)
        print(Ue)
        phi_i = np.matmul(Ue,phi)
        phi_list.append(phi_i)
        pop_g.append(norm(phi_i[0])**2)
        pop_e.append(norm(phi_i[1])**2)

    plt.plot(t_list,pop_g)
    return 0


def f_3 (phi,H,t=0):
    new_phi = np.matmul(H,phi)
    fx = new_phi*(-1j)
    return fx


def q1_matrix():
    #main function of question 1
    a = 0     # starting time
    b = 1.5    # end time
    N = 1000  # time step
    h = (b-a)/N  
    t_list = np.linspace(a,b,N)
    
    
    H = two_level_H(10,20)
    phi_g_list = []
    phi_e_list = []
    tem_r = np.array([1,0],dtype = complex) # initial condition
    for nt in t_list:
        #multivariable Runge-Kutta method
        gg = tem_r[0]
        ee = tem_r[1]
        phi_g_list.append(norm(gg)**2)
        phi_e_list.append(norm(ee)**2)

        
        k1 = h*f_3(tem_r,H,nt)
        k2 = h*f_3(tem_r+0.5*k1,H,nt+0.5*h)    
        k3 = h*f_3(tem_r+0.5*k2,H,nt+0.5*h)
        k4 = h*f_3(tem_r+k3,H,nt+h)
        tem_r +=(k1+2*k2+2*k3+k4)/6
    plt.plot(t_list,phi_g_list,"r-",label = "numerical")
    #plt.plot(t_list,rhogg_list,"b-",label = "rho_gg")
    # plt.plot(t_list,rhoge_list,"g-",label = "rho_ge")
    # plt.plot(t_list,rhoge_list,"y-",label = "rho_eg")
    plt.legend()
    plt.ylabel(r'$\rho_{ee}$')
    plt.xlabel(r"$\Gamma_{0}$t")
    plt.legend()
    #plt.savefig("/Users/weijunyuan/OneDrive - HKUST Connect/paper_book_pdf/courses/quantum_optics/numerical_plot_oneatom.png", dpi = 300)
    
q1_matrix()
#main()