from math import gamma
import numpy as np
from copy import deepcopy
class Trajectory_SSHmodel():

    def __init__(self,Nmol,hbar,seed=None):
        self.Nmol = Nmol
        self.Hmol = np.zeros((Nmol,Nmol),complex)
        self.Hmol_dt = np.zeros((Nmol,Nmol),complex)
        self.Cj = np.zeros((Nmol,1),complex)
        self.Hcoupling = np.zeros((Nmol,1),complex)
        self.Hcav = np.zeros((1,1),complex)
        # self.Xj = Xj#position
        self.X_past = np.zeros(Nmol) # 2023_4_18, using to store x(t-dt) for verlet algorithm
        # self.Vj = Vj
        self.hbar = hbar
        #velocity
        self.Rj = np.array(range(Nmol)) +1
        np.random.seed(seed)

    def initialHamiltonian(self,staticCoup,dynamicCoup):    
        self.staticCoup = staticCoup
        self.dynamicCoup = dynamicCoup
        Jmol = np.zeros((self.Nmol,self.Nmol),complex) #construct initial J operator. 2023/09/08


        for j in range(self.Nmol-1):
            self.Hmol[j,j+1] = -self.staticCoup + self.dynamicCoup * (self.Xj[j+1]-self.Xj[j])
            self.Hmol[j+1,j] = -self.staticCoup + self.dynamicCoup * (self.Xj[j+1]-self.Xj[j])
            self.Hmol_dt[j,j+1] = self.dynamicCoup * (self.Vj[j+1]-self.Vj[j])
            self.Hmol_dt[j+1,j] = self.dynamicCoup * (self.Vj[j+1]-self.Vj[j])
            Jmol[j,j+1] = -1j*self.Hmol[j,j+1]
            Jmol[j+1,j] = 1j*self.Hmol[j+1,j]
            
        self.Hmol[0,-1] = -self.staticCoup + self.dynamicCoup * (self.Xj[0]-self.Xj[-1])
        self.Hmol[-1,0] = -self.staticCoup + self.dynamicCoup * (self.Xj[0]-self.Xj[-1])
        self.Hmol_dt[0,-1] = self.dynamicCoup * (self.Vj[0]-self.Vj[-1])
        self.Hmol_dt[-1,0] = self.dynamicCoup * (self.Vj[0]-self.Vj[-1])
        Jmol[0,-1] = 1j*self.Hmol[0,-1]
        Jmol[-1,0] = -1j*self.Hmol[-1,0]
        
        self.Jt0 = Jmol
        self.Jt_t = Jmol
    def polarization_operator(self):
        self.Polar =np.eye(self.Nmol,dtype=complex)
        for i in range(self.Nmol):
            self.Polar[i][i] = i+1
        
    
    def Current_operator(self):
        J = -1j*(np.dot(self.Polar,self.Hmol) - np.dot(self.Hmol,self.Polar))
        return J
    
    
    def initialHamiltonianCavity(self,couplingStrength,cavityFrequency):
        self.couplingStrength = couplingStrength
        matrix_Jcav = np.zeros((self.Nmol,1),dtype=complex)
        for i in range(self.Nmol):
            self.Hcoupling[i,0] = self.couplingStrength
            matrix_Jcav[i,0] = -1j*self.couplingStrength 
        self.Hcav[0,0] = cavityFrequency
        
        self.Htot = np.vstack((np.hstack((self.Hmol, self.Hcoupling)),
                              np.hstack((self.Hcoupling.T, self.Hcav))))
        
        self.Hint = np.vstack((np.hstack((np.zeros((self.Nmol,self.Nmol),dtype=complex), self.Hcoupling)),
                              np.hstack((self.Hcoupling.T, np.zeros((1,1),dtype=complex)))))
        
        self.Jmol_0 = np.vstack((np.hstack((self.Jt0, np.zeros((self.Nmol,1),dtype=complex))),
                              np.hstack((np.zeros((1,self.Nmol),dtype=complex), np.zeros((1,1),dtype=complex)))))
        
        self.Jcav_0 = np.vstack((np.hstack((np.zeros((self.Nmol,self.Nmol),dtype=complex), matrix_Jcav)),
                              np.hstack((-matrix_Jcav.T, np.zeros((1,1),dtype=complex)))))
        
        self.Jtot_0 = self.Jmol_0 + self.Jcav_0

    def updateDiagonalStaticDisorder(self,DisorderValue):
        self.Hmol = deepcopy(self.Hmol)
        
        for i in range(self.Nmol):
            self.Hmol[i][i] +=np.random.normal(0,np.sqrt(DisorderValue))

    def eigenvalue(self):
        W,V = np.linalg.eigh(self.Htot)
        
        idx = W.argsort()[:]
        W = W[idx]
        
        return W

    def initialGaussian(self,kBT,mass,Kconst):
        self.mass = mass
        self.Kconst = Kconst

        self.Xj = np.random.normal(0.0, np.sqrt(kBT/(self.Kconst)), self.Nmol)#0.836 is conversion from amu*A^2*ps^-2 to cm-1
        self.Vj = np.random.normal(0.0, np.sqrt(kBT/(self.mass)),   self.Nmol)

    def initialState(self,hbar,kBT,most_prob=False):
        """
        Choose the initial state from the set of eigenfunctions based on Boltzman distribution exp(-E_n/kBT)
        """
        W,U = np.linalg.eig(self.Hmol)
        #idx = W.argsort()[::-1]   
        idx = W.argsort()[:]
        W = W[idx]
        U = U[:,idx]

        self.Prob = np.exp(-W*hbar/kBT)
        self.Prob = self.Prob/np.sum(self.Prob)

        rand = np.random.random()
        
        Prob_cum = np.cumsum(self.Prob)
        # initial_state = 0
        # while rand > Prob_cum[initial_state]: #trick for sampling according to  probabilty distribution.
        #     initial_state += 1
        # initial_state -= 1

        # print(rand, Prob_cum[initial_state],Prob_cum[initial_state+1])
        if most_prob:   
            initial_state = np.argmax(self.Prob) # most probable state
        initial_state = int(1-1)
        # self.Cj = U[:,initial_state]
        self.Cj = U[:,initial_state]
        self.Prob = self.Prob[initial_state]
        
    def initialCj_disorder(self):
        Hmol = self.Hmol
        W, U = np.linalg.eigh(Hmol)
        
        idx = W.argsort()[:]
        W = W[idx]
        U = U[:,idx]
        # print('this is eigenvalue from static:',W)
        # print('this is eigenvector from static:',U)
        
        self.Evalue = W
        self.Evec = U
    
    def initialCj_cavity(self):
        Hmol = self.Htot
        W, U = np.linalg.eigh(Hmol)
        
        idx = W.argsort()[:]
        W = W[idx]
        U = U[:,idx]
        # print('this is eigenvalue from static:',W)
        # print('this is eigenvector from static:',U)
        
        self.Evalue = W
        self.Evec = U
        
    def initialCj_disorder_trajectory(self,kBT):
        Hmol = self.Hmol
        W, U = np.linalg.eigh(Hmol)
        
        idx = W.argsort()[:]
        W = W[idx]
        U = U[:,idx]
        # print('this is eigenvalue from dynamic:',W)
        # print('this is eigenvector from dynamic:',U)
        Prob = np.exp(-W*self.hbar/kBT)
        self.Prob = Prob/np.sum(Prob)
        
        self.Cj = U
    
    def initialCj_cavity_trajectory(self,kBT):
        Hmol = self.Htot
        W, U = np.linalg.eigh(Hmol)
        
        idx = W.argsort()[:]
        W = W[idx]
        U = U[:,idx]
        # print('this is eigenvalue from dynamic:',W)
        # print('this is eigenvector from dynamic:',U)
        Prob = np.exp(-W*self.hbar/kBT)
        self.Prob = Prob/np.sum(Prob)
        # print(self.Prob)
        # print(np.sum(self.Prob[:10]))
        
        init = 1 - 1
        self.Cj = U[:,init]
        self.Prob = self.Prob[init]
    
    def initial_densityMatrix(self,kBT): # not necessary here
        Hmol = self.Hmol
        W, U = np.linalg.eigh(Hmol)
        
        idx = W.argsort()[:]
        W = W[idx]
        U = U[:,idx]
        
        Prob = np.exp(-W*self.hbar/kBT)
        Prob = Prob/np.sum(Prob)
        
        for i in range(self.Nmol):
            U[:,i] = U[:,i]*Prob[i]
        
        self.DM = U
    def initialCj_Hmolcav(self):
        Hmolcav = np.zeros(1)
    
    def initialstate_equal(self):
        self.Cj = np.ones(self.Nmol)/np.sqrt(self.Nmol)
        # self.Cj = np.zeros(self.Nmol)
        # self.Cj[0] = 1.0

        # var_list = []
        # for i in range(self.Nmol):
        #     R =  np.abs( np.sum(self.Rj    *np.abs(U[:,i].T)**2) ) 
        #     R2 = np.abs( np.sum((self.Rj-R)**2 *np.abs(U[:,i].T)**2) ) 
        #     var_list.append(R2)
        # print(min(var_list))
        
    
    def old_Aj(self,Ehrenfest=True):  #calculate a(t). update on 2023_4_18
        Aj = -self.Kconst/self.mass * self.Xj
        # print('This is Aj without Ehrenfest')
        # print(Aj)
        
        if Ehrenfest:
            for j in range(1,self.Nmol-1):
                Aj[j] = Aj[j] -self.dynamicCoup/(self.mass)* \
                        ( 2*np.real(np.conj(self.Cj[j])*self.Cj[j-1]) \
                        - 2*np.real(np.conj(self.Cj[j])*self.Cj[j+1]))
            Aj[0] = Aj[0] -self.dynamicCoup/(self.mass)* \
                    ( 2*np.real(np.conj(self.Cj[0])*self.Cj[-1]) \
                    - 2*np.real(np.conj(self.Cj[0])*self.Cj[1]))
            Aj[-1] = Aj[-1] -self.dynamicCoup/(self.mass)* \
                        ( 2*np.real(np.conj(self.Cj[-1])*self.Cj[-2]) \
                        - 2*np.real(np.conj(self.Cj[-1])*self.Cj[0]))
        # print('This is Aj with Ehrenfest')
        # print(Aj)
        self.accelerate = Aj
        # print('this is sum of Ehrenfest:',np.sum(Aj+self.Kconst*self.Xj/self.mass))
        

    def Newton(self,dt,Ehrenfest=True):
        self.Xj = self.Xj + self.Vj*dt #+ 0.5*dt**2*self.accelerate
        Aj = -self.Kconst/self.mass * self.Xj
        if Ehrenfest:
            for j in range(1,self.Nmol-1):
                Aj[j] = Aj[j] -self.dynamicCoup/(self.mass)* \
                        ( 2*np.real(np.conj(self.Cj[j])*self.Cj[j-1]) \
                        - 2*np.real(np.conj(self.Cj[j])*self.Cj[j+1]))
            Aj[0] = Aj[0] -self.dynamicCoup/(self.mass)* \
                    ( 2*np.real(np.conj(self.Cj[0])*self.Cj[-1]) \
                    - 2*np.real(np.conj(self.Cj[0])*self.Cj[1]))
            Aj[-1] = Aj[-1] -self.dynamicCoup/(self.mass)* \
                        ( 2*np.real(np.conj(self.Cj[-1])*self.Cj[-2]) \
                        - 2*np.real(np.conj(self.Cj[-1])*self.Cj[0]))
        self.Vj = self.Vj + dt*(Aj)
    def velocityVerlet(self,dt,Ehrenfest=True):
        """
        We use the algorithm with eliminating the half-step velocity
        https://en.wikipedia    .org/wiki/Verlet_integration
        """
        # velocity verlet oder: 1. calculate x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt**2
        # 2. calculate a(t+dt) = -F/m
        # 3. calculate v(t+dt) = v(t) + 0.5*(a(t+dt)+a(t))*dt
        self.Xj = self.Xj + self.Vj*dt + 0.5*dt**2*self.accelerate
        # 1: calculate Aj(t+dt) 2023_4_18 update
        Aj = -self.Kconst/self.mass * self.Xj
        if Ehrenfest:
            for j in range(1,self.Nmol-1):
                Aj[j] = Aj[j] -self.dynamicCoup/(self.mass)* \
                        ( 2*np.real(np.conj(self.Cj[j])*self.Cj[j-1]) \
                        - 2*np.real(np.conj(self.Cj[j])*self.Cj[j+1]))
            Aj[0] = Aj[0] -self.dynamicCoup/(self.mass)* \
                    ( 2*np.real(np.conj(self.Cj[0])*self.Cj[-1]) \
                    - 2*np.real(np.conj(self.Cj[0])*self.Cj[1]))
            Aj[-1] = Aj[-1] -self.dynamicCoup/(self.mass)* \
                        ( 2*np.real(np.conj(self.Cj[-1])*self.Cj[-2]) \
                        - 2*np.real(np.conj(self.Cj[-1])*self.Cj[0]))
        # 2: calculate Xj(t+dt)
        
        # 3: calculate Aj(t+dt)+Aj(t)
        # Aj = Aj +self.accelerate
        # 4: calculate Vj(t+dt)
        self.Vj = self.Vj + 0.5*dt*(Aj+self.accelerate) #update on 2023_4_18


    def x_dot(self,v): #xdot = \partial H/ \partial p = p/m = v
        return v
    
    def v_dot(self,X,Cj,Ehrenfest):
        Aj = -self.Kconst/self.mass * X
        if Ehrenfest:
            for j in range(1,self.Nmol-1):
                Aj[j] = Aj[j] -self.dynamicCoup/(self.mass)* \
                        ( 2*np.real(np.conj(Cj[j])*Cj[j-1]) \
                        - 2*np.real(np.conj(Cj[j])*Cj[j+1]))
            Aj[0] = Aj[0] -self.dynamicCoup/(self.mass)* \
                    ( 2*np.real(np.conj(Cj[0])*Cj[-1]) \
                    - 2*np.real(np.conj(Cj[0])*Cj[1]))
            Aj[-1] = Aj[-1] -self.dynamicCoup/(self.mass)* \
                        ( 2*np.real(np.conj(Cj[-1])*Cj[-2]) \
                        - 2*np.real(np.conj(Cj[-1])*Cj[0]))
        return Aj
    
    def Cj_dot(self,Xj,Cj):
        Hmol = np.zeros((self.Nmol,self.Nmol),complex)
        for j in range(self.Nmol-1):
            Hmol[j,j+1] = -self.staticCoup + self.dynamicCoup * (Xj[j+1]-Xj[j])
            Hmol[j+1,j] = -self.staticCoup + self.dynamicCoup * (Xj[j+1]-Xj[j])
        Hmol[0,-1] = -self.staticCoup + self.dynamicCoup * (Xj[0]-Xj[-1])
        Hmol[-1,0] = -self.staticCoup + self.dynamicCoup * (Xj[0]-Xj[-1])
        
        return -1j*np.dot(Hmol,Cj)/self.hbar
    
    def RK4(self,dt,Ehrenfest = True):
        kx1 = self.x_dot(self.Vj)
        kv1 = self.v_dot(self.Xj,self.Cj,Ehrenfest)
        kc1 = self.Cj_dot(self.Xj,self.Cj)
        
        kx2 = self.x_dot(self.Vj+dt*0.5*kv1)
        kv2 = self.v_dot(self.Xj+dt*0.5*kx1,self.Cj+dt*0.5*kc1,Ehrenfest)
        kc2 = self.Cj_dot(self.Xj+dt*0.5*kx1,self.Cj+dt*0.5*kc1)
        
        kx3 = self.x_dot(self.Vj+dt*0.5*kv2)
        kv3 = self.v_dot(self.Xj+dt*0.5*kx2,self.Cj+dt*0.5*kc2,Ehrenfest)
        kc3 = self.Cj_dot(self.Xj+dt*0.5*kx2,self.Cj+dt*0.5*kc2)
        
        kx4 = self.x_dot(self.Vj+dt*kv3)
        kv4 = self.v_dot(self.Xj+dt*kx3,self.Cj+dt*kc3,Ehrenfest)
        kc4 = self.Cj_dot(self.Xj+dt*kx3,self.Cj+dt*kc3)
        
        self.Xj +=  dt*(kx1+2*kx2+2*kx3+kx4)/6
        self.Vj +=  dt*(kv1+2*kv2+2*kv3+kv4)/6
        self.Cj +=  dt*(kc1+2*kc2+2*kc3+kc4)/6
        
    def v_dot_shiqiang(self,Xj,Cj):
        Aj = np.zeros(self.Nmol)
        for i in range(1,self.Nmol-1):
            Aj[i] = (-self.Kconst*Xj[i] + self.dynamicCoup*(2*np.real(np.conj(Cj[i+1])*Cj[i])
                                                            -2*np.real(np.conj(Cj[i])*Cj[i-1])) )/self.mass
        Aj[0] = (-self.Kconst*Xj[0] + self.dynamicCoup*(2*np.real(np.conj(Cj[0])*Cj[1])
                                                            -2*np.real(np.conj(Cj[0])*Cj[-1])) )/self.mass
        Aj[-1] = (-self.Kconst*Xj[-1] + self.dynamicCoup*(2*np.real(np.conj(Cj[-1])*Cj[0])
                                                            -2*np.real(np.conj(Cj[-1])*Cj[-2])) )/self.mass
        return Aj
    
    def Cj_dot_shiqiang(self,Xj,Cj):
        C = np.zeros(self.Nmol,complex)
        for i in range(1,self.Nmol-1):
            C[i] = -1j * (-self.staticCoup*(Cj[i+1]+Cj[i-1]) + self.dynamicCoup*((Xj[i]-Xj[i-1])*Cj[i-1] 
                                                                           +(Xj[i+1]-Xj[i])*Cj[i+1]))/self.hbar
            
        C[0] = -1j * (-self.staticCoup*(Cj[1]+Cj[-1]) + self.dynamicCoup*((Xj[0]-Xj[-1])*Cj[-1] 
                                                                           +(Xj[1]-Xj[0])*Cj[1]))/self.hbar
        C[-1] = -1j * (-self.staticCoup*(Cj[0]+Cj[-2]) + self.dynamicCoup*((Xj[-1]-Xj[-2])*Cj[-2] 
                                                                           +(Xj[0]-Xj[-1])*Cj[0]))/self.hbar
        return C
    
    def RK4_shiqiang(self,dt):
        kx1 = self.x_dot(self.Vj)
        kv1 = self.v_dot_shiqiang(self.Xj,self.Cj)
        kc1 = self.Cj_dot_shiqiang(self.Xj,self.Cj)
        
        kx2 = self.x_dot(self.Vj + 0.5*dt*kv1)
        kv2 = self.v_dot_shiqiang(self.Xj+0.5*dt*kx1, self.Cj+0.5*dt*kc1)
        kc2 = self.Cj_dot_shiqiang(self.Xj+0.5*dt*kx1, self.Cj+0.5*dt*kc1)
        
        kx3 = self.x_dot(self.Vj + 0.5*dt*kv2)
        kv3 = self.v_dot_shiqiang(self.Xj+0.5*dt*kx2, self.Cj+0.5*dt*kc2)
        kc3 = self.Cj_dot_shiqiang(self.Xj+0.5*dt*kx2, self.Cj+0.5*dt*kc2)
        
        kx4 = self.x_dot(self.Vj + dt*kv3)
        kv4 = self.v_dot_shiqiang(self.Xj+ dt*kx3, self.Cj+dt*kc3)
        kc4 = self.Cj_dot_shiqiang(self.Xj+dt*kx3, self.Cj+dt*kc3)
        
        self.Xj += dt*(kx1 + 2*kx2 + 2*kx3 + kx4)/6
        self.Vj += dt*(kv1 + 2*kv2 + 2*kv3 + kv4)/6
        self.Cj += dt*(kc1 + 2*kc2 + 2*kc3 + kc4)/6
    
    def RK4_densityMatrix(self,dt):
        k1 = -1j*np.dot(self.Hmol,self.DM)
        k2 = -1j*np.dot(self.Hmol,self.DM + dt*k1/2)
        k3 = -1j*np.dot(self.Hmol,self.DM + dt*k2/2)
        k4 = -1j*np.dot(self.Hmol,self.DM + dt*k3)
        
        self.DM += (k1 + 2*k2 + 2*k3 + k4)*dt/6
        
    def RK4_Jmol(self,dt):
        
        k1 = -1j*np.dot(self.Hmol,self.Jt_t)
        k2 = -1j*np.dot(self.Hmol,self.Jt_t + dt*k1/2)
        k3 = -1j*np.dot(self.Hmol,self.Jt_t + dt*k2/2)
        k4 = -1j*np.dot(self.Hmol,self.Jt_t + dt*k3)
        
        self.Jt_t += (k1 + 2*k2 + 2*k3 + k4)*dt/6
    
    def RK4_Cj_trajectory(self,dt,Cj):
        
        k1 = -1j*np.dot(self.Hmol,Cj)
        k2 = -1j*np.dot(self.Hmol,Cj + dt*k1/2)
        k3 = -1j*np.dot(self.Hmol,Cj + dt*k2/2)
        k4 = -1j*np.dot(self.Hmol,Cj + dt*k3)
        
        Cj += (k1 + 2*k2 + 2*k3 + k4)*dt/6
        
        return Cj
    
    def RK4_Cj_cavity_trajectory(self,dt):
        
        k1 = -1j*np.dot(self.Htot,self.Cj)
        k2 = -1j*np.dot(self.Htot,self.Cj + dt*k1/2)
        k3 = -1j*np.dot(self.Htot,self.Cj + dt*k2/2)
        k4 = -1j*np.dot(self.Htot,self.Cj + dt*k3)
        
        self.Cj += (k1 + 2*k2 + 2*k3 + k4)*dt/6

    def updateHmol(self):
        for j in range(self.Nmol-1):
            self.Hmol[j,j+1] = -self.staticCoup + self.dynamicCoup * (self.Xj[j+1]-self.Xj[j])
            self.Hmol[j+1,j] = -self.staticCoup + self.dynamicCoup * (self.Xj[j+1]-self.Xj[j])
            self.Hmol_dt[j,j+1] = self.dynamicCoup * (self.Vj[j+1]-self.Vj[j])
            self.Hmol_dt[j+1,j] = self.dynamicCoup * (self.Vj[j+1]-self.Vj[j])

        self.Hmol[0,-1] = -self.staticCoup + self.dynamicCoup * (self.Xj[0]-self.Xj[-1])
        self.Hmol[-1,0] = -self.staticCoup + self.dynamicCoup * (self.Xj[0]-self.Xj[-1])
        self.Hmol_dt[0,-1] = self.dynamicCoup * (self.Vj[0]-self.Vj[-1])
        self.Hmol_dt[-1,0] = self.dynamicCoup * (self.Vj[0]-self.Vj[-1])
        
    def updateHamiltonianCavity(self):
        self.Htot = np.vstack((np.hstack((self.Hmol, self.Hcoupling)),
                              np.hstack((self.Hcoupling.T, self.Hcav))))


    def propagateCj(self,dt):
        # print('This is Cj before propogation',self.Cj)
        # print('This is norm of Cj before propogation:',np.sum(np.abs(self.Cj)**2))
        self.Cj = self.Cj - 1j*dt*np.dot(self.Hmol,self.Cj)/self.hbar \
                  -0.5*dt**2*np.dot(self.Hmol,np.dot(self.Hmol,self.Cj))/self.hbar**2 \
                  -0.5*1j*dt**2*np.dot(self.Hmol_dt,self.Cj)/self.hbar
        # print('This is Cj after propogation',self.Cj)
        # print('This is norm of Cj after propogation:',np.sum(np.abs(self.Cj)**2))

    
    def shift_Rj(self):
        R_max = self.Rj[np.argmax((np.abs(self.Cj.T)**2))]
        self.Rj = ((self.Rj - R_max + self.Nmol/2)%self.Nmol) +1
        # print(R_max)
        # print(self.Rj)
        
    def getEnergy(self):# 2023_4_19 revise vibrorational +electronic part
        E_elec = np.real(np.dot(np.conj(self.Cj.T),np.dot(self.Hmol,self.Cj))) #electronic energy
        E_vib = 0.5*self.mass*np.linalg.norm(self.Vj)**2 + 0.5*self.Kconst*np.linalg.norm(self.Xj)**2 #nuclear energy
        return E_elec,E_vib

    def getDisplacement(self):
        # print(np.sum(np.abs(self.Cj.T)**2))
        # R2 = np.abs( np.sum(self.Rj**2 *np.abs(self.Cj.T)**2) ) 
        R =  np.sum(self.Rj    *np.abs(self.Cj.T)**2)
        R2 = np.sum((self.Rj-R)**2 *np.abs(self.Cj.T)**2)
        return R2
    
    def getCurrentCorrelation(self):
        self.Imol = 0
        if hasattr(self, 'J0Cj'):
            # self.Jt = deepcopy(self.Ht)
            self.Jt = np.zeros_like(self.Hmol)
            for j in range(self.Nmol-1): 
                self.Jt[self.Imol+j,   self.Imol+j+1] = -self.Hmol[self.Imol+j,   self.Imol+j+1]*1j
                self.Jt[self.Imol+j+1, self.Imol+j]   = self.Hmol[self.Imol+j+1, self.Imol+j]*1j   
            
            self.Jt[self.Imol,self.Imol+self.Nmol-1] = self.Hmol[self.Imol,self.Imol+self.Nmol-1]*1j
            self.Jt[self.Imol+self.Nmol-1,self.Imol] = -self.Hmol[self.Imol+self.Nmol-1,self.Imol]*1j 
            # Here Cj is at time t
            # self.JtCj = np.dot(self.Jt,self.Cj)
        else: #first step only 
            self.J0Cj = np.dot(self.Jt0,self.Cj)
            self.Jt = deepcopy(self.Jt0)

        CJJ = np.dot(np.conj(self.Cj).T,np.dot(self.Jt,self.J0Cj))
        Javg = np.dot(np.conj(self.Cj).T,np.dot(self.Jt,self.Cj))
        return Javg, CJJ
    
    def getCurrentCorrelation_cavity(self):
        self.Ngroundstate = 0
        if hasattr(self, 'Jmol0Cj'):
            # self.Jt = deepcopy(self.Ht)
            self.Jmol_t = np.zeros_like(self.Htot)
            for j in range(self.Nmol-1): 
                self.Jmol_t[self.Ngroundstate+j, self.Ngroundstate+j+1] = -self.Hmol[self.Ngroundstate+j, self.Ngroundstate+j+1] *1j
                self.Jmol_t[self.Ngroundstate+j+1, self.Ngroundstate+j]   = self.Hmol[self.Ngroundstate+j+1, self.Ngroundstate+j] *1j   
            
            self.Jmol_t[self.Ngroundstate, self.Ngroundstate+self.Nmol-1] = self.Hmol[self.Ngroundstate, self.Ngroundstate +self.Nmol-1]*1j
            self.Jmol_t[self.Ngroundstate +self.Nmol-1, self.Ngroundstate] = -self.Hmol[self.Ngroundstate+self.Nmol-1, self.Ngroundstate]*1j 
            self.Jcav_t = deepcopy(self.Jcav_0)
            
            self.Jtot_t = self.Jmol_t + self.Jcav_t
            # Here Cj is at time t
            # self.JtCj = np.dot(self.Jt,self.Cj
        else: #first step only 
            self.Jmol0Cj = np.dot(self.Jmol_0,self.Cj)
            self.Jtot0Cj = np.dot(self.Jtot_0,self.Cj)
            self.Jcav0Cj = np.dot(self.Jcav_0,self.Cj)
            
            self.Jtot_t = deepcopy(self.Jtot_0)
            self.Jmol_t = deepcopy(self.Jmol_0)
            self.Jcav_t = deepcopy(self.Jcav_0)

        CJJmol = np.dot(np.conj(self.Cj).T,np.dot(self.Jmol_t,self.Jmol0Cj))
        CJJcav = np.dot(np.conj(self.Cj).T,np.dot(self.Jcav_t,self.Jcav0Cj))
        CJJtot = np.dot(np.conj(self.Cj).T,np.dot(self.Jtot_t,self.Jtot0Cj))
        
        return CJJmol, CJJcav, CJJtot
    
    def getCurrentCorrelation_static(self,dt,Ntimes,kBT,Vcav,J_cav=True,J_mol=True):
        times = np.arange(Ntimes) * dt
        if Vcav == 0.0:
            Nsize = self.Nmol
            Jmol = self.Jt0
        elif J_cav:
            Nsize = self.Nmol + 1
            Jmol = self.Jcav_0
        elif J_mol:
            Nsize = self.Nmol + 1
            Jmol = self.Jmol_0
        else:
            Nsize = self.Nmol + 1
            Jmol = self.Jtot_0
        
        J0kl = np.dot(np.conj(self.Evec).T,np.dot(Jmol,self.Evec))
        
        CJJ_avg = np.zeros(Ntimes,complex)
        Partition = 0.0
        # CJJ_analytical = np.zeros(Ntimes,complex)
        for k in range(Nsize):
            CJJ_analytical = np.zeros(Ntimes,complex)
            for l in range(Nsize):
                CJJ_analytical = CJJ_analytical + np.exp(1j* (self.Evalue[k]-self.Evalue[l])* times) * np.abs(J0kl[k,l])**2
            
            CJJ_avg = CJJ_avg + CJJ_analytical * np.exp(-self.Evalue[k]/kBT)
            Partition = Partition + np.exp(-self.Evalue[k]/kBT)
        CJJ_avg = CJJ_avg / Partition  #ensemble average
        # print(Partition)
        
        return CJJ_avg
    
    def getCurrentCorrelation_DM(self):
        
        CJJ = np.trace(np.dot(self.DM,np.dot(self.Jt_t,self.Jt0)))
        return CJJ
        
    
    def getCurrentCorrelation_disorder_trajectory(self,Cj,J0Cj):
        self.Imol = 0
        
            # self.Jt = deepcopy(self.Ht)
        self.Jt = np.zeros_like(self.Hmol)
        for j in range(self.Nmol-1): 
            self.Jt[self.Imol+j,   self.Imol+j+1] = -self.Hmol[self.Imol+j,   self.Imol+j+1]*1j
            self.Jt[self.Imol+j+1, self.Imol+j]   = self.Hmol[self.Imol+j+1, self.Imol+j]*1j   
        
        self.Jt[self.Imol,self.Imol+self.Nmol-1] = self.Hmol[self.Imol,self.Imol+self.Nmol-1]*1j
        self.Jt[self.Imol+self.Nmol-1,self.Imol] = -self.Hmol[self.Imol+self.Nmol-1,self.Imol]*1j 
            # Here Cj is at time t
            # self.JtCj = np.dot(self.Jt,self.Cj)
        #first step only 
            # self.J0Cj = np.dot(self.Jt0,Cj) 
            # self.Jt = deepcopy(self.Jt0)

        CJJ = np.dot(np.conj(Cj).T,np.dot(self.Jt,J0Cj))
        # Javg = np.dot(np.conj(Cj).T,np.dot(self.Jt,self.Cj))
        return CJJ
    
    def getCurrentCorrelation_cavity_trajectory(self,Cj,J0Cj):
        
        self.Jtot_t = deepcopy(self.Jtot_0)
        for j in range(self.Nmol-1): 
            self.Jtot_t[j, j+1] = -self.Hmol[j, j+1]*1j
            self.Jtot_t[j+1, j] = self.Hmol[j+1, j]*1j   
        
        self.Jtot_t[0,self.Nmol-1] = self.Hmol[0, self.Nmol-1]*1j
        self.Jtot_t[self.Nmol-1,0] = -self.Hmol[self.Nmol-1, 0]*1j 
        
        CJJ = np.dot(np.conj(Cj).T,np.dot(self.Jtot_t,J0Cj))
        
        return CJJ
    
    def propagateJ0Cj_RK4_trajectory(self,dt,J0Cj,Vcav):
        ### RK4 propagation 
        if Vcav==0:
            K1 = -1j*np.dot(self.Hmol,J0Cj)
            K2 = -1j*np.dot(self.Hmol,J0Cj+dt*K1/2)
            K3 = -1j*np.dot(self.Hmol,J0Cj+dt*K2/2)
            K4 = -1j*np.dot(self.Hmol,J0Cj+dt*K3)
        else:
            K1 = -1j*np.dot(self.Htot,J0Cj)
            K2 = -1j*np.dot(self.Htot,J0Cj+dt*K1/2)
            K3 = -1j*np.dot(self.Htot,J0Cj+dt*K2/2)
            K4 = -1j*np.dot(self.Htot,J0Cj+dt*K3)
        
        J0Cj += (K1+2*K2+2*K3+K4)*dt/6
        return J0Cj
    
    def propagateJ0Cj_RK4(self,dt):
        ### RK4 propagation 
        K1 = -1j*np.dot(self.Hmol,self.J0Cj)
        K2 = -1j*np.dot(self.Hmol,self.J0Cj+dt*K1/2)
        K3 = -1j*np.dot(self.Hmol,self.J0Cj+dt*K2/2)
        K4 = -1j*np.dot(self.Hmol,self.J0Cj+dt*K3)
        self.J0Cj += (K1+2*K2+2*K3+K4)*dt/6
    
    def propagateJmol0Cj_RK4(self,dt):
        ### RK4 propagation 
        K1 = -1j*np.dot(self.Htot,self.Jmol0Cj)
        K2 = -1j*np.dot(self.Htot,self.Jmol0Cj+dt*K1/2)
        K3 = -1j*np.dot(self.Htot,self.Jmol0Cj+dt*K2/2)
        K4 = -1j*np.dot(self.Htot,self.Jmol0Cj+dt*K3)
        self.Jmol0Cj += (K1+2*K2+2*K3+K4)*dt/6
    
    def propagateJcav0Cj_RK4(self,dt):
        ### RK4 propagation 
        K1 = -1j*np.dot(self.Htot,self.Jcav0Cj)
        K2 = -1j*np.dot(self.Htot,self.Jcav0Cj+dt*K1/2)
        K3 = -1j*np.dot(self.Htot,self.Jcav0Cj+dt*K2/2)
        K4 = -1j*np.dot(self.Htot,self.Jcav0Cj+dt*K3)
        self.Jcav0Cj += (K1+2*K2+2*K3+K4)*dt/6
    
    def propagateJtot0Cj_RK4(self,dt):
        ### RK4 propagation 
        K1 = -1j*np.dot(self.Htot,self.Jtot0Cj)
        K2 = -1j*np.dot(self.Htot,self.Jtot0Cj+dt*K1/2)
        K3 = -1j*np.dot(self.Htot,self.Jtot0Cj+dt*K2/2)
        K4 = -1j*np.dot(self.Htot,self.Jtot0Cj+dt*K3)
        self.Jtot0Cj += (K1+2*K2+2*K3+K4)*dt/6

class SingleExcitationWithCollectiveCoupling():

    def __init__(self,Nmol,Nrad,seed=None):
        self.Nmol = Nmol
        self.Nrad = Nrad
        np.random.seed(seed)

    def initialHamiltonian_Radiation(self,Wgrd,Wmol,Vndd,Vrad,Wmax,damp,useQmatrix=False):
        """
        Construct the Hamiltonian in the form of 
        Ht0 = 
            | grd     | mol     | rad 
        grd | Hgrd    |         |         
        mol | Vmolgrd | Hmol    |
        rad | Vradgrd | Vradmol | Hrad
        """
        self.Wmol = Wmol
        self.useQmatrix = useQmatrix
        self.damp = damp
        self.Erad = np.zeros(self.Nrad)

        Hgrd = np.eye(1) * Wgrd
        Hmol = np.eye(self.Nmol) * Wmol
        Hrad = np.zeros((self.Nrad,self.Nrad),complex)

        Vradmol = np.zeros((self.Nrad,self.Nmol),complex)
        Vmolgrd = np.ones((self.Nmol,1),complex)
        Vradgrd = np.zeros((self.Nrad,1),complex)


        # Construct the molecule-radiation coupling
        Gamma = 0.0
        for j in range(self.Nrad):
            self.Erad[j] = ( j - (self.Nrad-1)/2 ) * Wmax *2.0/(self.Nrad-1)
            Hrad[j,j] = self.Erad[j] - 1j*self.damp
            for i in range(self.Nmol):
                Vradmol[j,i] = Vrad # * (Wrad_width**2) / ( Erad[j]**2 + Wrad_width**2 )
            Gamma += -2.0*1j*(Vradmol[j,0]**2)/Hrad[j,j] # set to be the same: 0
        #Gamma = 1j*Gamma*(Vrad**2)
        self.Gamma = np.real(Gamma)
        # print(self.Gamma)

        # Construct the nearest dipole-dipole coupling
        for j in range(self.Nmol-1): 
            Hmol[j,   j+1] = Vndd
            Hmol[j+1, j  ] = Vndd
        Hmol[0,-1] = Vndd
        Hmol[-1,0] = Vndd
        
        drive = 0.0
        if useQmatrix:
            Qmol = np.ones((self.Nmol,self.Nmol))
            self.Ht0 = np.vstack(( np.hstack(( Hgrd,          Vmolgrd.T*drive               )),
                                   np.hstack(( Vmolgrd*drive, Hmol - 1j*(self.Gamma/2)*Qmol )) ))        
        else:
            self.Ht0 = np.vstack((  np.hstack(( Hgrd,          Vmolgrd.T*drive,    Vradgrd.T    )),
                                    np.hstack(( Vmolgrd*drive, Hmol,               Vradmol.T    )),
                                    np.hstack(( Vradgrd,       Vradmol,            Hrad         )) ))
        self.Ht = deepcopy(self.Ht0)

        self.Imol = 1
        self.Irad = self.Nmol+1

    def initialHamiltonian_Cavity_Radiation(self,Wgrd,Wcav,Wmol,Vndd,Vcav,Vrad,Wmax,damp,useQmatrix=False):
        """
        Construct the Hamiltonian in the form of 
        Ht0 = 
            | grd     | cav     | mol     | rad 
        grd | Hgrd    |         |         |
        cav | Vcavgrd | Hcav    |         |  
        mol | Vmolgrd | Vmolcav | Hmol    |
        rad | Vradgrd | Vradcav | Vradmol | Hrad
        """
        self.Wmol = Wmol
        self.useQmatrix = useQmatrix
        self.damp = damp
        self.Erad = np.zeros(self.Nrad)

        Hgrd = np.eye(1) * Wgrd
        Hcav = np.eye(1) * Wcav
        Hmol = np.eye(self.Nmol) * Wmol
        Hrad = np.zeros((self.Nrad,self.Nrad),complex)

        Vradmol = np.zeros((self.Nrad,self.Nmol),complex)
        Vmolgrd = np.ones((self.Nmol,1),complex)
        Vradgrd = np.zeros((self.Nrad,1),complex)
        Vcavgrd = np.zeros((1,1),complex)
        Vmolcav = np.ones((self.Nmol,1),complex) * Vcav
        Vradcav = np.zeros((self.Nrad,1),complex)

        # Construct the molecule-radiation coupling
        Gamma = 0.0
        for j in range(self.Nrad):
            self.Erad[j] = ( j - (self.Nrad-1)/2 ) * Wmax *2.0/(self.Nrad-1)
            Hrad[j,j] = self.Erad[j] - 1j*self.damp
            for i in range(self.Nmol):
                Vradmol[j,i] = Vrad # * (Wrad_width**2) / ( Erad[j]**2 + Wrad_width**2 )
            Gamma += -2.0*1j*(Vradmol[j,0]**2)/Hrad[j,j] # set to be the same: 0
        #Gamma = 1j*Gamma*(Vrad**2)
        self.Gamma = np.real(Gamma)
        # print(self.Gamma)
        
        # Construct the nearest dipole-dipole coupling
        for j in range(self.Nmol-1): 
            Hmol[j,   j+1] = Vndd
            Hmol[j+1, j  ] = Vndd
        Hmol[0,-1] = Vndd
        Hmol[-1,0] = Vndd

        drive = 0.0
        if useQmatrix:
            Qmol = np.ones((self.Nmol,self.Nmol))
            self.Ht0 = np.vstack((  np.hstack(( Hgrd,          Vcavgrd.T,     Vmolgrd.T*drive               )),
                                    np.hstack(( Vcavgrd,       Hcav,          Vmolcav.T                     )),
                                    np.hstack(( Vmolgrd*drive, Vmolcav,       Hmol - 1j*(self.Gamma/2)*Qmol )) ))
        else:
            self.Ht0 = np.vstack((  np.hstack(( Hgrd,          Vcavgrd.T,     Vmolgrd.T*drive,    Vradgrd.T )),
                                    np.hstack(( Vcavgrd,       Hcav,          Vmolcav.T,          Vradcav.T )),
                                    np.hstack(( Vmolgrd*drive, Vmolcav,       Hmol,               Vradmol.T )),
                                    np.hstack(( Vradgrd,       Vradcav,       Vradmol,            Hrad      )) ))
        self.Ht = deepcopy(self.Ht0)
        
        self.Icav = 1
        self.Imol = 2
        self.Irad = self.Nmol+2

    def initialHamiltonian_Cavity_nonHermitian(self,Wgrd,Wcav,Wmol,Vndd,Vcav,Kcav,Gamma=0.0):
        """
        Construct the Hamiltonian in the form of 
        Ht0 = 
            | grd     | cav     | mol   
        grd | Hgrd    |         |       
        cav | Vcavgrd | Hcav    |         
        mol | Vmolgrd | Vmolcav | Hmol  
        """
        self.useQmatrix = True #Just to eliminate the rad part 
        self.Wmol = Wmol
        self.Gamma = Gamma

        Hgrd = np.eye(1) * Wgrd
        Hcav = np.eye(1) * Wcav
        Hmol = np.eye(self.Nmol) * Wmol
        Jmol = np.zeros((self.Nmol,self.Nmol),complex)

        Vmolgrd = np.ones((self.Nmol,1),complex)
        Vcavgrd = np.zeros((1,1),complex)
        Vmolcav = np.ones((self.Nmol,1),complex) * Vcav
        if not Kcav==0:
            for j in range(self.Nmol):
                # Vmolcav[j,0] = Vcav*np.sin(Kcav*np.pi*j/self.Nmol)
                Vmolcav[j,0] = Vcav*np.exp(-1j*(Kcav*np.pi*j/self.Nmol))
        
        # Construct the nearest dipole-dipole coupling
        for j in range(self.Nmol-1): 
            Hmol[j,   j+1] = Vndd
            Hmol[j+1, j  ] = Vndd
            Jmol[j,   j+1] = Vndd*1j
            Jmol[j+1, j  ] =-Vndd*1j
        Hmol[0,-1] = Vndd
        Hmol[-1,0] = Vndd
        Jmol[0,-1] =-Vndd*1j
        Jmol[-1,0] = Vndd*1j

        drive = 0.0
        Qmol = np.ones((self.Nmol,self.Nmol))
        self.Ht0 = np.vstack((  np.hstack(( Hgrd,          Vcavgrd.T,     Vmolgrd.T*drive   )),
                                np.hstack(( Vcavgrd,       Hcav,          np.conj(Vmolcav).T)),
                                np.hstack(( Vmolgrd*drive, Vmolcav,       Hmol -1j*(self.Gamma/2)*Qmol   )) ))
        
        self.Qmat = np.vstack((  np.hstack(( Hgrd*0.0,      Vcavgrd.T*0.0,   Vmolgrd.T*0.0   )),
                                np.hstack(( Vcavgrd*0.0,   Hcav*0.0,        Vmolcav.T*0.0   )),
                                np.hstack(( Vmolgrd*0.0,   Vmolcav*0.0,     -1j*(self.Gamma/2)*Qmol )) ))

        self.Jt0 = np.vstack((  np.hstack(( Hgrd*0.0,      Vcavgrd.T*0.0,   Vmolgrd.T*0.0   )),
                                np.hstack(( Vcavgrd*0.0,   Hcav*0.0,        Vmolcav.T*0.0   )),
                                np.hstack(( Vmolgrd*0.0,   Vmolcav*0.0,     Jmol )) ))
        self.Jt = deepcopy(self.Jt0)

        self.Ht = deepcopy(self.Ht0)        
        self.Icav = 1
        self.Imol = 2

    def updateDiagonalStaticDisorder(self,Delta):
        self.Ht = deepcopy(self.Ht0)

        self.Wstc = np.random.normal(0.0,Delta,self.Nmol) + self.Wmol
        for j in range(self.Nmol): 
            self.Ht[self.Imol+j,self.Imol+j] += self.Wstc[j]

    def updateDiagonalDynamicDisorder(self,Delta,TauC,dt):
        # simulate Gaussian process
        # cf. George B. Rybicki's note
        # https://www.lanl.gov/DLDSTP/fast/OU_process.pdf
        self.Ht = deepcopy(self.Ht0)

        if not hasattr(self, 'Wdyn'):
            self.Wdyn = np.random.normal(0.0,Delta,self.Nmol) + self.Wmol
        else:
            ri = np.exp(-dt/TauC) * (TauC>0.0)
            mean_it = ri*self.Wdyn
            sigma_it = Delta*np.sqrt(1.0-ri**2)
            self.Wdyn = np.random.normal(mean_it,sigma_it,self.Nmol) + self.Wmol
        
        for j in range(self.Nmol): 
            self.Ht[self.Imol+j,self.Imol+j] += self.Wdyn[j]

    def updateNeighborStaticDisorder(self,Delta):
        self.Ht = deepcopy(self.Ht0)

        if not hasattr(self, 'Vstc'):
            self.Vstc = np.random.normal(0.0,Delta,self.Nmol)
        
        for j in range(self.Nmol-1): 
            self.Ht[self.Imol+j,   self.Imol+j+1] += self.Vstc[j]
            self.Ht[self.Imol+j+1, self.Imol+j]   += self.Vstc[j]
        
        self.Ht[self.Imol,self.Imol+self.Nmol-1] += self.Vstc[-1]
        self.Ht[self.Imol+self.Nmol-1,self.Imol] += self.Vstc[-1]

    def updateNeighborDynamicDisorder(self,Delta,TauC,dt):
        # simulate Gaussian process
        # cf. George B. Rybicki's note
        # https://www.lanl.gov/DLDSTP/fast/OU_process.pdf
        self.Ht = deepcopy(self.Ht0)

        # if not hasattr(self, 'Vdyn'):
        #     self.Vdyn = np.random.normal(0.0,Delta,self.Nmol)
        # else:
        #     ri = np.exp(-dt/TauC) * (TauC>0.0)
        #     mean_it = ri*self.Vdyn
        #     sigma_it = Delta*np.sqrt(1.0-ri**2)
        #     self.Vdyn = np.random.normal(mean_it,sigma_it,self.Nmol)
        
        # for j in range(self.Nmol-1): 
        #     self.Ht[self.Imol+j,   self.Imol+j+1] += self.Vdyn[j]
        #     self.Ht[self.Imol+j+1, self.Imol+j]   += self.Vdyn[j]
        
        # self.Ht[self.Imol,self.Imol+self.Nmol-1] += self.Vdyn[-1]
        # self.Ht[self.Imol+self.Nmol-1,self.Imol] += self.Vdyn[-1]

        if not hasattr(self, 'Xdyn'):
            self.Xdyn = np.random.normal(0.0,1.0,self.Nmol)
        else:
            ri = np.exp(-dt/TauC) * (TauC>0.0)
            mean_it = ri*self.Xdyn
            sigma_it = np.sqrt(1.0-ri**2)
            self.Xdyn = np.random.normal(mean_it,sigma_it,self.Nmol)
        
        for j in range(self.Nmol-1): 
            self.Ht[self.Imol+j,   self.Imol+j+1] += Delta*(self.Xdyn[j+1]-self.Xdyn[j])
            self.Ht[self.Imol+j+1, self.Imol+j]   += Delta*(self.Xdyn[j+1]-self.Xdyn[j])
        
        self.Ht[self.Imol,self.Imol+self.Nmol-1] += Delta*(self.Xdyn[0]-self.Xdyn[-1])
        self.Ht[self.Imol+self.Nmol-1,self.Imol] += Delta*(self.Xdyn[0]-self.Xdyn[-1])

    def updateNeighborHarmonicOscillator(self,staticCoup,dynamicCoup):
        self.Ht = deepcopy(self.Ht0)

        if not hasattr(self, 'dHdt'):
            self.dHdt = np.zeros_like(self.Ht0)

        self.staticCoup = staticCoup
        self.dynamicCoup = dynamicCoup

        for j in range(self.Nmol-1):
            self.Ht[self.Imol+j,   self.Imol+j+1] += -self.staticCoup + self.dynamicCoup * (self.Xj[j+1]-self.Xj[j])
            self.Ht[self.Imol+j+1, self.Imol+j]   += -self.staticCoup + self.dynamicCoup * (self.Xj[j+1]-self.Xj[j])
            self.dHdt[self.Imol+j,   self.Imol+j+1] = self.dynamicCoup * (self.Vj[j+1]-self.Vj[j])
            self.dHdt[self.Imol+j+1, self.Imol+j]   = self.dynamicCoup * (self.Vj[j+1]-self.Vj[j])

        self.Ht[self.Imol,self.Imol+self.Nmol-1] += -self.staticCoup + self.dynamicCoup * (self.Xj[0]-self.Xj[-1])
        self.Ht[self.Imol+self.Nmol-1,self.Imol] += -self.staticCoup + self.dynamicCoup * (self.Xj[0]-self.Xj[-1])
        self.dHdt[self.Imol,self.Imol+self.Nmol-1] = self.dynamicCoup * (self.Vj[0]-self.Vj[-1])
        self.dHdt[self.Imol+self.Nmol-1,self.Imol] = self.dynamicCoup * (self.Vj[0]-self.Vj[-1])

    def initialCj_Cavity(self):
        self.Cj = np.zeros((self.Nmol,1),complex)
        if hasattr(self, 'Icav'):
            self.Cj = np.vstack( (np.zeros((1,1),complex),    #grd
                                    np.ones((1,1),complex),  #cav
                                    self.Cj) )                #mol
        else:
            print("cannot initial Cj in the cavity state when there is no cavity")
            exit()
        if not self.useQmatrix:
            self.Cj = np.vstack( (self.Cj,np.zeros((self.Nrad,1),complex)) )

    def initialCj_Bright(self):
        self.Cj = np.ones((self.Nmol,1),complex)/np.sqrt(self.Nmol)
        if hasattr(self, 'Icav'):
            self.Cj = np.vstack( (np.zeros((1,1),complex),    #grd
                                    np.zeros((1,1),complex),  #cav
                                    self.Cj) )                #mol
        else:
            self.Cj = np.vstack( (np.zeros((1,1),complex),    #grd
                                    self.Cj) )                #mol
        if not self.useQmatrix:
            self.Cj = np.vstack( (self.Cj,np.zeros((self.Nrad,1),complex)) )

    def initialCj_Ground(self):
        self.Cj = np.zeros((self.Nmol,1),complex)
        if hasattr(self, 'Icav'):
            self.Cj = np.vstack( (np.ones((1,1),complex),   #grd
                                    np.zeros((1,1),complex),  #cav
                                    self.Cj) )                #mol
        else:  
            self.Cj = np.vstack( (np.ones((1,1),complex),   #grd
                                    self.Cj) )                #mol          
        if not self.useQmatrix:
            self.Cj = np.vstack( (self.Cj,np.zeros((self.Nrad,1),complex)) )

    def initialCj_Random(self):      
        self.Cj = np.ones((self.Nmol,1),complex)/np.sqrt(self.Nmol)*np.exp(1j*2*np.pi*np.random.rand(self.Nmol,1))
        if hasattr(self, 'Icav'):
            self.Cj = np.vstack( (np.zeros((1,1),complex),  #grd
                                    np.zeros((1,1),complex),  #cav 
                                    self.Cj) )                #mol
        else:
            self.Cj = np.vstack( (np.zeros((1,1),complex),  #grd
                                    self.Cj) )                #mol
        if not self.useQmatrix:
            self.Cj = np.vstack( (self.Cj,np.zeros((self.Nrad,1),complex)) )

    def initialCj_middle(self):
        """
        choose the initial Cj as a single exictation at the middle of the chain
        """
        self.Cj = np.zeros((self.Nmol,1),complex)
        self.Cj[int(self.Nmol/2)] = 1.0
        # j0 = int(self.Nmol/2)
        # width = 1
        # for j in range(self.Nmol):
        #     self.Cj[j,0] = np.exp(-(j-j0)**2/width**2/2)            
        # self.Cj = self.Cj/np.sqrt(np.sum(np.abs(self.Cj)**2))
        
        if hasattr(self, 'Icav'):
            self.Cj = np.vstack( (np.zeros((1,1),complex),    #grd
                                    np.zeros((1,1),complex),  #cav
                                    self.Cj) )                #mol
        else:
            self.Cj = np.vstack( (np.zeros((1,1),complex),    #grd
                                    self.Cj) )                #mol
        if not self.useQmatrix:
            self.Cj = np.vstack( (self.Cj,np.zeros((self.Nrad,1),complex)) )

    def initialCj_Gaussian(self,width):
        """
        Initialize Cj as a Gaussian distribution centered at the middle of the chain
        """
        self.Cj = np.zeros((self.Nmol,1),complex)
        middle = int(self.Nmol/2)
        for j in range(self.Nmol):
            self.Cj[j] = np.exp(-(j-middle)**2/2/width**2)/np.sqrt(np.sqrt(np.pi)*width)
            # self.Cj[j] = np.exp(-(j-middle)**2/2/width**2)/np.sqrt(np.sqrt(np.pi)*width) * np.exp(1j*1.0*j) #WITH AN INITIAL MOMENTUM
        print(np.linalg.norm(self.Cj)**2)

        if hasattr(self, 'Icav'):
            self.Cj = np.vstack( (np.zeros((1,1),complex),    #grd
                                    np.zeros((1,1),complex),  #cav
                                    self.Cj) )                #mol
        else:
            self.Cj = np.vstack( (np.zeros((1,1),complex),    #grd
                                    self.Cj) )                #mol
        if not self.useQmatrix:
            self.Cj = np.vstack( (self.Cj,np.zeros((self.Nrad,1),complex)) )

    def initialCj_Eigenstate_Forward(self,Wmol,Vndd,initial_state=0):
        """
        Choose the initial state to be one of the eigenstate of the forward drift matrix
        """
        Amol = np.eye(self.Nmol) * Wmol
        for j in range(self.Nmol-1): 
            Amol[j,   j+1] = Vndd
            # Amol[j+1, j  ] = Vndd
        # Amol[0,-1] = Vndd
        Amol[-1,0] = Vndd
        W,U = np.linalg.eig(Amol)
        #idx = W.argsort()[::-1]   
        idx = W.argsort()[:]
        W = W[idx]
        U = U[:,idx]

        # Initialize state vector
        self.Cj = U.T[initial_state]
        self.Cj = self.Cj[..., None] 
        if hasattr(self, 'Icav'):
            self.Cj = np.vstack( (np.zeros((1,1),complex),  #grd
                                  np.zeros((1,1),complex),  #cav
                                  self.Cj) )                #mol
        else:
            self.Cj = np.vstack( (np.zeros((1,1),complex),  #grd
                                  self.Cj) )                #mol
        if not self.useQmatrix:
            self.Cj = np.vstack( (self.Cj,np.zeros((self.Nrad,1),complex)) )

        return W

    def initialCj_Eigenstate_Hmol(self,initial_state=0):
        """
        Choose the initial state to be one of the eigenstate of Hmol
        """
        # Use the updated Hamiltonian with this initial intermolecular coupling
        Hmol = self.Ht[self.Imol:self.Imol+self.Nmol,self.Imol:self.Imol+self.Nmol] ###NOTE: INCORRECT! THIS INCLUCE Qmat!!! 

        W,U = np.linalg.eigh(Hmol)
        #idx = W.argsort()[::-1]   
        idx = W.argsort()[:]
        W = W[idx]
        U = U[:,idx]

        # Initialize state vector
        self.Cj = U.T[initial_state]
        self.Cj = self.Cj[..., None] 
        if hasattr(self, 'Icav'):
            self.Cj = np.vstack( (np.zeros((1,1),complex),  #grd
                                  np.zeros((1,1),complex),  #cav
                                  self.Cj) )                #mol
        else:
            self.Cj = np.vstack( (np.zeros((1,1),complex),  #grd
                                  self.Cj) )                #mol
        if not self.useQmatrix:
            self.Cj = np.vstack( (self.Cj,np.zeros((self.Nrad,1),complex)) )

        return W, U

    def initialCj_Eigenstate_Hcavmol(self,initial_state):
        """
        Choose the initial state to be one of the eigenstate of Hmol+Hcav
        """
        # Use the updated Hamiltonian with this initial intermolecular coupling
        Hcavmol = self.Ht[self.Icav:self.Imol+self.Nmol,self.Icav:self.Imol+self.Nmol]
        # Hcavmol = self.Ht0[self.Icav:self.Imol+self.Nmol,self.Icav:self.Imol+self.Nmol]

        W,U = np.linalg.eigh(Hcavmol)
        #idx = W.argsort()[::-1]   
        idx = W.argsort()[:]
        W = W[idx]
        U = U[:,idx]
        
        # Initialize state vector
        self.Cj = U.T[initial_state]
        self.Cj = self.Cj[..., None] 

        self.Cj = np.vstack( (np.zeros((1,1),complex),  #grd
                              self.Cj) )                #mol
        if not self.useQmatrix:
            self.Cj = np.vstack( (self.Cj,np.zeros((self.Nrad,1),complex)) )

        return W, U

    def initialCj_Boltzman(self,hbar,kBT,most_prob=False):
        """
        Choose the initial state from the set of eigenfunctions based on Boltzman distribution exp(-E_n/kBT)
        """
        # Use the updated Hamiltonian with this initial intermolecular coupling
        Hmol = self.Ht[self.Imol:self.Imol+self.Nmol,self.Imol:self.Imol+self.Nmol] ###NOTE: INCORRECT! THIS INCLUCE Qmat!!! 

        W,U = np.linalg.eigh(Hmol)
        #idx = W.argsort()[::-1]   
        idx = W.argsort()[:]
        W = W[idx]
        U = U[:,idx]

        self.Prob = np.exp(-W*hbar/kBT)
        self.Prob = self.Prob/np.sum(self.Prob)

        rand = np.random.random()
        
        Prob_cum = np.cumsum(self.Prob)
        initial_state = 0
        while rand > Prob_cum[initial_state]:
            initial_state += 1
        initial_state -= 1

        # print(rand, Prob_cum[initial_state],Prob_cum[initial_state+1])
        if most_prob:   
            initial_state = np.argmax(self.Prob) # most probable state
        
        self.Prob = self.Prob[initial_state]

        # Initialize state vector
        self.Cj = U.T[initial_state]
        self.Cj = self.Cj[..., None] 
        if hasattr(self, 'Icav'):
            self.Cj = np.vstack( (np.zeros((1,1),complex),  #grd
                                  np.zeros((1,1),complex),  #cav
                                  self.Cj) )                #mol
        else:
            self.Cj = np.vstack( (np.zeros((1,1),complex),  #grd
                                  self.Cj) )                #mol
        if not self.useQmatrix:
            self.Cj = np.vstack( (self.Cj,np.zeros((self.Nrad,1),complex)) )

    def initialCj_Polariton(self,initial_state):
        """
        Choose the initial state as the upper/lower polariton
        """
        # Use the updated Hamiltonian with this initial intermolecular coupling
        Hmol = (self.Ht-self.Qmat)[self.Icav:self.Imol+self.Nmol,self.Icav:self.Imol+self.Nmol]

        W,U = np.linalg.eigh(Hmol)
        #idx = W.argsort()[::-1]   
        idx = W.argsort()[:]
        W = W[idx]
        U = U[:,idx]
        
        # Initialize state vector
        self.Cj = U.T[initial_state]
        self.Cj = self.Cj[..., None] 

        self.Cj = np.vstack( (np.zeros((1,1),complex),  #grd
                              self.Cj) )                #mol
        if not self.useQmatrix:
            self.Cj = np.vstack( (self.Cj,np.zeros((self.Nrad,1),complex)) )

    def propagateCj_RK4(self,dt):
        ### RK4 propagation 
        K1 = -1j*np.dot(self.Ht,self.Cj)
        K2 = -1j*np.dot(self.Ht,self.Cj+dt*K1/2)
        K3 = -1j*np.dot(self.Ht,self.Cj+dt*K2/2)
        K4 = -1j*np.dot(self.Ht,self.Cj+dt*K3)
        self.Cj += (K1+2*K2+2*K3+K4)*dt/6

    def propagateCj_dHdt(self,dt):
        if not hasattr(self, 'dHdt'):
            self.dHdt = np.zeros_like(self.Ht0)
        self.Cj = self.Cj - 1j*dt*np.dot(self.Ht,self.Cj) \
                   -0.5*dt**2*np.dot(self.Ht,np.dot(self.Ht,self.Cj)) \
                   -0.5*1j*dt**2*np.dot(self.dHdt,self.Cj)

    def initialXjVj_Gaussian(self,kBT,mass,Kconst):
        self.kBT = kBT
        self.mass = mass
        self.Kconst = Kconst

        self.Xj = np.random.normal(0.0, self.kBT/self.Kconst, self.Nmol)#revise later
        self.Vj = np.random.normal(0.0, self.kBT/self.mass,   self.Nmol)
        
    def propagateXjVj_velocityVerlet(self,dt):
        """
        We use the algorithm with eliminating the half-step velocity
        https://en.wikipedia.org/wiki/Verlet_integration
        """
        # 1: calculate Aj(t)
        Aj = -self.Kconst/self.mass * self.Xj
        for j in range(1,self.Nmol-1):
            Aj[j] = Aj[j] -self.dynamicCoup/self.mass* \
                    ( 2*np.real(np.conj(self.Cj[self.Imol+j])*self.Cj[self.Imol+j-1]) \
                    - 2*np.real(np.conj(self.Cj[self.Imol+j])*self.Cj[self.Imol+j+1]))
        Aj[0] = Aj[0] -self.dynamicCoup/self.mass* \
                ( 2*np.real(np.conj(self.Cj[self.Imol+0])*self.Cj[self.Imol+self.Nmol-1]) \
                - 2*np.real(np.conj(self.Cj[self.Imol+0])*self.Cj[self.Imol+1]))
        Aj[-1] = Aj[-1] -self.dynamicCoup/self.mass* \
                    ( 2*np.real(np.conj(self.Cj[self.Imol+self.Nmol-1])*self.Cj[self.Imol+self.Nmol-2]) \
                    - 2*np.real(np.conj(self.Cj[self.Imol+self.Nmol-1])*self.Cj[self.Imol]))
        # 2: calculate Xj(t+dt)
        self.Xj = self.Xj + self.Vj*dt + 0.5*dt**2*Aj
        # 3: calculate Aj(t+dt)+Aj(t)
        Aj = Aj -self.Kconst/self.mass * self.Xj #a(t+Δt) + a(t)
        # 4: calculate Vj(t+dt)
        self.Vj = self.Vj + 0.5*dt*Aj

    def getPopulation_system(self):
        return np.linalg.norm(self.Cj[self.Imol:self.Imol+self.Nmol])**2

    def getPopulation_radiation(self):
        if not self.useQmatrix:
            return np.linalg.norm(self.Cj[self.Irad:])**2
        else:
            return 0.0
            
    def getPopulation_cavity(self):
        if hasattr(self, 'Icav'):
            return np.abs(self.Cj[self.Icav])**2
        else:
            return 0.0

    def getIPR(self):
        return np.linalg.norm(self.Cj[self.Imol:self.Imol+self.Nmol])**4 \
                / np.sum(np.abs(self.Cj[self.Imol:self.Imol+self.Nmol])**4)

    def getEnergy(self):
        return 0.5*self.mass*np.linalg.norm(self.Vj)**2 + 0.5*self.Kconst*np.linalg.norm(self.Xj)**2

    def getDisplacement(self):
        # print(np.sum(np.abs(self.Cj.T)**2))
        # R2 = np.abs( np.sum(self.Rj**2 *np.abs(self.Cj.T)**2) ) 
        Rj = np.array(range(self.Nmol))
        R =  np.abs( np.sum( Rj       *np.abs(self.Cj[self.Imol:self.Imol+self.Nmol].T)**2) ) 
        R2 = np.abs( np.sum((Rj-R)**2 *np.abs(self.Cj[self.Imol:self.Imol+self.Nmol].T)**2) ) 
        return R2

    def getCurrentCorrelation(self):
        if hasattr(self, 'J0Cj'):
            # self.Jt = deepcopy(self.Ht)
            self.Jt = np.zeros_like(self.Ht)
            for j in range(self.Nmol-1): 
                self.Jt[self.Imol+j,   self.Imol+j+1] = self.Ht[self.Imol+j,   self.Imol+j+1]*1j
                self.Jt[self.Imol+j+1, self.Imol+j]   =-self.Ht[self.Imol+j+1, self.Imol+j]*1j   
            
            self.Jt[self.Imol,self.Imol+self.Nmol-1] =-self.Ht[self.Imol,self.Imol+self.Nmol-1]*1j
            self.Jt[self.Imol+self.Nmol-1,self.Imol] = self.Ht[self.Imol+self.Nmol-1,self.Imol]*1j 
            # Here Cj is at time t
            # self.JtCj = np.dot(self.Jt,self.Cj)
        else: #first step only 
            self.J0Cj = np.dot(self.Jt0,self.Cj)
            self.Jt = deepcopy(self.Jt0)

        CJJ = np.dot(np.conj(self.Cj).T,np.dot(self.Jt,self.J0Cj))
        Javg = np.dot(np.conj(self.Cj).T,np.dot(self.Jt,self.Cj))
        return Javg[0,0], CJJ[0,0]

    def propagateJ0Cj_RK4(self,dt):
        ### RK4 propagation 
        K1 = -1j*np.dot(self.Ht,self.J0Cj)
        K2 = -1j*np.dot(self.Ht,self.J0Cj+dt*K1/2)
        K3 = -1j*np.dot(self.Ht,self.J0Cj+dt*K2/2)
        K4 = -1j*np.dot(self.Ht,self.J0Cj+dt*K3)
        self.J0Cj += (K1+2*K2+2*K3+K4)*dt/6
