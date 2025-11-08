# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 07:42:12 2022

@author: Columbia
"""

'''Generates plot showing available transitions with transition dipole 
moments as shown in Fig.5 of arXiv:2205.05686

Note, this uses the transition_plot function from the plotting.py module!
'''

from numpy.linalg import eigh
import diatomic.plotting as plotting
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

Na23Cs133 = {"I1":1.5,
            "I2":3.5,
            "d0":4.6*DebyeSI,
            "binding":114268135.25e6*h*0, ### need to figure out the true value
            #"Brot":3471.295e6*h/2,
            # "Brot":1735.65937e6*h,   #fit
            "Brot":1735.65999e6*h,
            "Drot":0*h,

            "Q1":-96e3*h,       #fit   -97
            "Q2":220e3*h,       #fit   150
         
            "C1":14.2*h,            
            "C2":902*h,
     
            "C3":105.6*h,
            "C4":3941.8*h,      #fit    3941.8
            "MuN":0.0073*muN,   #fit   0.0073  

            "Mu1":1.478*(1-639.2e-6)*muN, 
            "Mu2":0.738*(1-6278.7e-6)*muN,
            # "Mu1":1.478*muN,
            # "Mu2":0.738*muN,
            "a0": 0, #1064nm
            "a2": 0, #1064nm
            "Beta":0}


Nmax=2
H0,Hz,Hdc,Hac = \
    hamiltonian.build_hamiltonians(Nmax,Na23Cs133,zeeman=True,Edc=True,ac=True)

I = 0 #W/m^2
E = 0 #V/m
B = 864e-4 #T
# B  = 39.4e-4
H = H0+Hz*B+Hdc*E+Hac*I 

energies, states = eigh(H)
# pyplot.imshow(states.real)
# pyplot.show()
# fig = pyplot.figure()
# start = time.time()
# sigma_m_tf, sigma_p_tf,pi_tf,dm,dp,dz = plotting.transition_plot(energies,states,2,Nmax,Na23Cs133['I1'],Na23Cs133['I2'],Offset=3471.295,prefactor=1e-6, maxf=0.8)

# print(sigma_m_tf)
# print(sigma_p_tf)
# print(pi_tf)
# pyplot.vlines(sigma_m_tf, ymin = 0, ymax = 1,colors = 'red')
# pyplot.vlines(sigma_p_tf,ymin = 0,ymax = 1, colors = 'green')
# pyplot.vlines(pi_tf,ymin = 0,ymax = 1, colors = 'blue')
# pyplot.vlines(mf_2,ymin = 0,ymax = 1, colors = 'red')
# print(dp)
# print(dm)
# print(dz)

mf_4 = [3471.298,3471.33, 3471.342]
#mf_2 = [3471.392,3471.37, 3471.32, 3471.304]
mf_2 = [3471.304,3471.32,3471.37,3471.392]
mf_3 = [3471.32,3471.358,3471.384]



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

sigma_m_tf, sigma_p_tf,pi_tf,D_m,D_p,D_z,state_sigma_p,state_pi = transition_calculation(energies,states,2,Nmax,Na23Cs133['I1'],Na23Cs133['I2'],Offset=3471.295,prefactor=1e-6, maxf=0.8)

print("======  sigma - ===============")
print(sigma_m_tf)
print("======  sigma + ===============")
print(sigma_p_tf)
print("======  pi ===============")
print(pi_tf)
# pi = 3471.305
# sigma_m = 3471.328
# sigma_p = 3471.323

mF3_p = 3471.327 
mF3_m = 3471.332 
mF3_pi = 3471.302 
mF4_p = 3471.322 
mF4_m = 3471.312 
mF4_pi = 3471.319 

pyplot.vlines(mF3_p, ymin = 0, ymax = 1, color = 'green',ls = '--',label = 'data +')
pyplot.vlines(mF3_m , ymin = 0, ymax = 1, color = 'red',ls = '--',label = 'data -')
pyplot.vlines(mF3_pi , ymin = 0, ymax = 1, color = 'blue',ls = '--',label = 'data pi' )

pyplot.legend()
pyplot.xlim(3471.280,3471.350)
print(D_m)
print(D_p)
print(D_z)

tf = []
for D_1 in (D_m,D_p,D_z):
    TDM = D_1[:,1]
    id_max = numpy.argmax(TDM)
    tf.append(D_1[id_max,0])
pyplot.vlines(tf[0],ymin = 0,ymax = 1,color = 'red',label = 'sigma- calcula=tion')
pyplot.vlines(tf[1],ymin = 0,ymax = 1,color = 'green',label = "sigma+ calculation")
pyplot.vlines(tf[2], ymin = 0, ymax = 1, color = 'blue',label = 'pi ')
pyplot.legend()

# print((1/2)*(pi+sigma_m+sigma_p)/3)
#excited_state = state_sigma_p[1][2].real


# print("for  sigma + transition  from mF = 4 to mF = 5")
# print(state_sigma_p[0][1])
# print(state_sigma_p[0][2][33])
# print(state_sigma_p[0][2][40])
# print(state_sigma_p[0][2][64])

# print(state_sigma_p[1][1])
# print(state_sigma_p[1][2][33])
# print(state_sigma_p[1][2][40])
# print(state_sigma_p[1][2][64])

# print(state_sigma_p[2][1])
# print(state_sigma_p[2][2][33])
# print(state_sigma_p[2][2][40])
# print(state_sigma_p[2][2][64])


# sigma_m_tf, sigma_p_tf,pi_tf,D_m,D_p,D_z,state_sigma_p,state_pi = transition_calculation(energies,states,0,Nmax,Na23Cs133['I1'],Na23Cs133['I2'],Offset=3471.295,prefactor=1e-6, maxf=0.8)

# print("for pi transition from mF = 5 to mF = 5")
# print(state_pi[0][1])
# print(state_pi[0][2][33])
# print(state_pi[0][2][40])
# print(state_pi[0][2][64])

# print(state_pi[1][1])
# print(state_pi[1][2][33])
# print(state_pi[1][2][40])
# print(state_pi[1][2][64])

# print(state_pi[2][1])
# print(state_pi[2][2][33])
# print(state_pi[2][2][40])
# print(state_pi[2][2][64])

# tf = []
# for D in (D_m,D_p,D_z):
#     TDM = D[:,1]
#     id_max = numpy.argmax(TDM)
#     tf.append(D[id_max,0])
# print(tf)
   
# pyplot.vlines(tf[0],ymin = 0,ymax = 1,color = 'red',ls = '--',label = 'sigma- fitted')
# pyplot.vlines(tf[1],ymin = 0,ymax = 1,color = 'green',ls = '--',label = 'sigma+ fitted')
# pyplot.vlines(tf[2],ymin = 0,ymax = 1,color = 'blue',ls = '--',label = 'pi fitted')
# print("average:")
# print(1/2*(tf[0]+tf[1]+tf[2])/3)

# pyplot.title("At 864 G")
# pyplot.legend()
# pyplot.show()

# my_data = numpy.genfromtxt('Pi_average.csv', delimiter=',')
# DDS,nacs = my_data[1:,0],my_data[1:,1]
# pyplot.scatter(DDS,nacs,c ='blue')
# pyplot.plot(DDS,nacs,c='blue')
# my_data = numpy.genfromtxt('Sigma-_average.csv', delimiter=',')
# DDS,nacs = my_data[1:,0],my_data[1:,1]
# pyplot.scatter(DDS,nacs,c ='red')
# pyplot.plot(DDS,nacs, c= 'red')
# my_data = numpy.genfromtxt('Sigma+_average.csv', delimiter=',')
# DDS,nacs = my_data[1:,0],my_data[1:,1]
# pyplot.scatter(DDS,nacs, c= 'green')
# pyplot.plot(DDS,nacs,c = 'green')
# # pyplot.vlines(tf[0],ymin = -0.1,ymax = 1,color = 'red',ls = '--',label = 'sigma- fitted')
# # pyplot.vlines(tf[1],ymin = -0.1,ymax = 1,color = 'green',ls = '--',label = 'sigma+ fitted')
# # pyplot.vlines(tf[2],ymin = -0.1,ymax = 1,color = 'blue',ls = '--',label = 'pi fitted')
# pyplot.vlines(sigma_m_tf,ymin = -0.1,ymax = 1,color = 'red',label = 'sigma- data')
# pyplot.vlines(sigma_p_tf,ymin = -0.1,ymax = 1,color = 'green',label = "sigma+ data")
# pyplot.vlines(pi_tf, ymin = 0, ymax = 1, color = 'blue',label = 'pi data')
# pyplot.title("At 864 G")
# pyplot.legend()
# pyplot.show()


# dm = numpy.round(calculate.transition_dipole_moment(2,3/2,7/2,+1,states,states[:,1]),6)
# print(dm)



# Nmax=4
# H0,Hz,Hdc,Hac = \
#     hamiltonian.build_hamiltonians(Nmax,Na23Cs133,zeeman=True,Edc=True,ac=True)

# I = 0 #W/m^2
# E = 0 #V/m
# B = 39.6e-4 #T
# #B = 39.4e-4
# H = H0+Hz*B+Hdc*E+Hac*I 

# energies, states = eigh(H)
# sigma_m_tf, sigma_p_tf,pi_tf,D_m,D_p,D_z = transition_calculation(energies,states,2,Nmax,Na23Cs133['I1'],Na23Cs133['I2'],Offset=3471.295,prefactor=1e-6, maxf=0.8)
# pyplot.vlines(mf_4,ymin = 0,ymax = 1, colors = 'green',label = "sigma+ data")
# pyplot.vlines(D_p[1:4,0],ymin = 0,ymax = 1, colors = 'green',ls = '--',label ="sigma+ fitted" )
# #pyplot.vlines(D_p[:,0],ymin = 0,ymax = 1, colors = 'green',label ="sigma+ calculated" )
# pyplot.title("At 39.6 G")
# pyplot.legend()
# pyplot.show()
# pyplot.vlines(mf_2,ymin = 0,ymax = 1, colors = 'red',label = 'sigma- data')
# pyplot.vlines([D_m[0,0],D_m[1,0],D_m[5,0],D_m[7,0]],ymin = 0,ymax = 1, colors = 'red',ls = '--',label = 'sigma- fitted')
# #pyplot.vlines(D_m[:,0],ymin = 0,ymax = 1, colors = 'blue',label = 'sigma- calculated')

# pyplot.title("At 39.6 G")
# pyplot.legend()
# pyplot.show()

# pyplot.vlines(mf_3,ymin = 0,ymax = 1, colors = 'blue',label = 'pi data')
# pyplot.vlines([D_z[2,0],D_z[4,0],D_z[7,0]],ymin = 0,ymax = 1, colors = 'blue',ls = '--',label = 'pi fitted')
# #pyplot.vlines(D_z[:,0],ymin = 0,ymax = 1, colors = 'blue',label = 'pi calculated')
# pyplot.title("At 39.6 G")
# pyplot.legend()
# pyplot.show()

# my_data = numpy.genfromtxt('39.6_G_sigma_+.csv', delimiter=',')
# DDS,nacs = my_data[1:,0],my_data[1:,1]
# pyplot.scatter(DDS,nacs,c ='green')
# pyplot.plot(DDS,nacs,c='green')

# my_data = numpy.genfromtxt('39.6G_sigma+_-.csv', delimiter=',')
# DDS,nacs = my_data[1:,0],my_data[1:,1]
# pyplot.scatter(DDS,nacs,c ='orange')
# pyplot.plot(DDS,nacs, c= 'orange')

# sigma_m_tf, sigma_p_tf,pi_tf,D_m,D_p,D_z = transition_calculation(energies,states,2,Nmax,Na23Cs133['I1'],Na23Cs133['I2'],Offset=3471.295,prefactor=1e-6, maxf=0.8)
# pyplot.vlines(mf_4,ymin = 0,ymax = 4, colors = 'green',label = "sigma+ data")
# pyplot.vlines(D_p[1:4,0],ymin = 0,ymax = 4, colors = 'green',ls = '--',label ="sigma+ fitted" )
# #pyplot.vlines(D_p[:,0],ymin = 0,ymax = 1, colors = 'green',label ="sigma+ calculated" )

# pyplot.vlines(mf_2,ymin = 0,ymax = 4, colors = 'red',label = 'sigma- data')
# pyplot.vlines([D_m[0,0],D_m[1,0],D_m[5,0],D_m[7,0]],ymin = 0,ymax = 4, colors = 'red',ls = '--',label = 'sigma- fitted')
# #pyplot.vlines(D_m[:,0],ymin = 0,ymax = 1, colors = 'blue',label = 'sigma- calculated')

# # pyplot.vlines(mf_3,ymin = 0,ymax = 4, colors = 'blue',label = 'pi data')
# # pyplot.vlines([D_z[2,0],D_z[4,0],D_z[7,0]],ymin = 0,ymax = 4, colors = 'blue',ls = '--',label = 'pi fitted')
# #pyplot.vlines(D_z[:,0],ymin = 0,ymax = 1, colors = 'blue',label = 'pi calculated')
# pyplot.title("At 39.6 G")
# pyplot.legend()
# pyplot.show()







# print(D_m)

# print(sigma_p_tf)
# print(pi_tf)
# pyplot.vlines(sigma_m_tf,ymin = 0,ymax = 1, colors = 'red')
# D_p.sort(axis=0)


mf_2 = [3471.304,3471.32,3471.37]
def optimization_object(x):
    Na23Cs133 = {"I1":1.5,
                "I2":3.5,
                "d0":4.6*DebyeSI,
                "binding":114268135.25e6*h*0, ### need to figure out the true value
                #"Brot":1735.6594e6*h,
                "Brot":x[0]*h,   #fit
                "Drot":0*h,
                #"Q1":-50e3*h,
                "Q1":x[1]*h,       #fit   -90
                "Q2":x[2]*h,       #fit   152
                "C1":14.2*h,            
                "C2":854.5*h,
                "C3":105.6*h,
                "C4":3941*h,      #fit
                #"C4":x[3]*h,
                "MuN":x[3]*muN,   #fit
                "Mu1":1.478*(1-639.2e-6)*muN, 
                "Mu2":0.738*(1-6278.7e-6)*muN,
                "a0":0, #1064nm
                "a2":0, #1064nm
                "Beta":0}
    Nmax=2
    H0,Hz,Hdc,Hac = \
        hamiltonian.build_hamiltonians(Nmax,Na23Cs133,zeeman=True,Edc=True,ac=True)

    I = 0 #W/m^2
    E = 0 #V/m
    #B = 181.5e-4 #T
   # B = 39.6e-4
    B = 864e-4
    H = H0+Hz*B+Hdc*E+Hac*I 

    energies, states = eigh(H)
    sigma_m_tf, sigma_p_tf,pi_tf,D_m,D_p,D_z = transition_calculation(energies,states,2,Nmax,Na23Cs133['I1'],Na23Cs133['I2'],Offset=3471.295,prefactor=1e-6, maxf=0.8)
    

    
    
    diff_m = []
    diff_p = []
    diff_z = []
    for s_p in sigma_p_tf:
        diff_p.append((s_p-sigma_p)**2)
    for s_m in  sigma_m_tf:
        diff_m.append((s_m-sigma_m)**2)
    for s_z in pi_tf:
        diff_z.append((s_z-pi)**2)
    cost_h = min(diff_p)+min(diff_z)+min(diff_m)
    #### for low field
    # cost_p = 0
    # for mf4 in mf_4:
    #     diff_list = []
    #     for sigma_p in sigma_p_tf:
    #         diff_list.append(abs(sigma_p-mf4))
    #     cost_p = cost_p+min(diff_list)
    # cost_m = 0
    # for mf2 in mf_2:
    #     diff_list = []
    #     for sigma_m in sigma_m_tf:
    #         diff_list.append(abs(sigma_m-mf2))
    #     cost_m = cost_m+min(diff_list)
    # cost = cost_m+cost_p
    
    ##### for high field
    # print(D_m)
    # print(D_p)
    # print(D_z)
    # tf = []
    # for D_1 in (D_m,D_p,D_z):
    #     TDM = D_1[:,1]
    #     id_max = numpy.argmax(TDM)
    #     tf.append(D_1[id_max,0])
    # cost_h = abs(tf[0]-sigma_m)+abs(tf[1]-sigma_p)+abs(tf[2]-pi)
    
    
    B = 39.6e-4
    H = H0+Hz*B+Hdc*E+Hac*I 

    energies, states = eigh(H)
    sigma_m_tf, sigma_p_tf,pi_tf,D_m,D_p,D_z = transition_calculation(energies,states,2,Nmax,Na23Cs133['I1'],Na23Cs133['I2'],Offset=3471.295,prefactor=1e-6, maxf=0.8)
    
    
    cost_p = 0
    for mf4 in mf_4:
        diff_list = []
        for s_p in sigma_p_tf:
            diff_list.append((s_p-mf4)**2)
        cost_p = cost_p+min(diff_list)
    cost_m = 0
    for mf2 in mf_2:
        diff_list = []
        for s_m in sigma_m_tf:
            diff_list.append((s_m-mf2)**2)
        cost_m = cost_m+min(diff_list)
    cost_low = cost_m+cost_p
    
    


    
    #cost = cost_low
    cost = numpy.sqrt(cost_h+cost_low)
    print(cost)
    return cost





#start = time.time()
#cost_funct = optimization_object(numpy.array([-94e3,150e3,3941.3,0.0073]))
#end = time.time()
#print(end-start)
# resbrute = brute(optimization_object,((1735.6585,1753.6595),(-117e3,-77e3),(130e3,170e3),(3841,4041),(0.0063,0.0083)) , Ns = 10, full_output=True,finish=None)
# print(resbrute[0])
# print(resbrute[1])
# x0 = [1735.6593e6,-97e3,150e3,0.0073]
# bnds = ((1735.6589e6,1735.6599e6),(-110e3,-85e3),(140e3,160e3),(0.006,0.009))
# result = differential_evolution(rosen, bounds)
# res = minimize(optimization_object,x0,method='Powell',bounds = bnds)


# bounds = [(1735.6589e6,1735.6599e6),(-110e3,-85e3),(140e3,160e3),(0.006,0.009)]
# res= differential_evolution(optimization_object, bounds)
# print(res.x)





def dipole(Nmax,I1,I2,d,M):
    ''' Generates the induced dipole moment operator for a Rigid rotor.
    Expanded to cover state  vectors in the uncoupled hyperfine basis.

    Args:
        Nmax (int) - maximum rotational states
        I1,I2 (float) - nuclear spin quantum numbers
        d (float) - permanent dipole moment
        M (float) - index indicating the helicity of the dipole field

    Returns:
        Dmat (numpy.ndarray) - dipole matrix
    '''
    shape = numpy.sum(numpy.array([2*x+1 for x in range(0,int(Nmax+1))]))
    dmat = numpy.zeros((shape,shape),dtype= numpy.complex)
    i =0
    j =0
    for N1 in range(0,int(Nmax+1)):
        for M1 in range(N1,-(N1+1),-1):
            for N2 in range(0,int(Nmax+1)):
                for M2 in range(N2,-(N2+1),-1):
                    dmat[i,j]=d*numpy.sqrt((2*N1+1)*(2*N2+1))*(-1)**(M1)*\
                    wigner_3j(N1,1,N2,-M1,M,M2)*wigner_3j(N1,1,N2,0,0,0)
                    j+=1
            j=0
            i+=1

    shape1 = int(2*I1+1)

    shape2 = int(2*I2+1)

    dmat = numpy.kron(dmat,numpy.kron(numpy.identity(shape1),
                                                    numpy.identity(shape2)))

    return dmat










