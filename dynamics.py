# -*- coding: utf-8 -*-
"""
Author: Alexandre Barbosa
Contact: alexandre.barbosa@tecnico.ulisboa.pt
Last Updated: 21-10-2023
"""

""" 
TODO: Add polarization degree, other pulse shapes?

"""

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from pde import FieldCollection, PDEBase, ScalarField
from scipy.special import spherical_jn
from geometry import grid, ngrid

class CollectiveSpinsPDE(PDEBase):
    
    """ A system of coupled Heisenberg-Bloch-Langevin equations describing the
    dynamics of collective states of alkali-metal and noble-gas spin ensembles
    and the light field throughout an optical quantum memory protocol. [1]
    
    [1] O. Firstenberg et al., Optical quantum memory for noble-gas spins based 
        on spin-exchange collisions,  Phys. Rev. A 105, 042606 (2022).      """

    def __init__(self, bca="dirichlet", bcb="neumann", gamma_p=1, gamma_k=0, 
                 gamma_s=4.03e-7-9.35e-9*3.14*3.14, pa=1, pb=1, na=5.2e14, nb=5.4e19, detuning=0,
                 delta_s=0, delta_k=1.11e-3, J=1.85e-5, g=1.239e-7*np.sqrt(4/3*3.14), kappa=0.90371, 
                 Da=9.35e-9, Db=1.87e-8, omega=0.253*np.sqrt(4/3*3.14), control_shape="square",
                 control_duration=1, pulse_shape="exp", T=593.7,    
                 t0=1e4, tdark=7.5e4, protocol="sequential", afc=False): 
        
        super().__init__()
        self.complex_valued = True # collective spin operators have complex average values
        self.bca = bca # Dirichlet or Robin, depending on the quality of the antirrelaxation coating
        self.bcb = bcb # von Neumann Boundary Conditions
        self.gamma_p = gamma_p # decay rate for alkali excited-ground state coherence 
        self.gamma_k = gamma_k # decay rate for noble-gas spin coherence
        self.gamma_s = gamma_s # decay rate for alkali metastable-ground state coherence
        self.pa = pa # degree of polarization of the alkali spin ensemble    (between 0 and 1)
        self.pb = pb # degree of polarization of the noble-gas spin ensemble (between 0 and 1)
        self.na = na # alkali-metal gas density
        self.nb = nb # noble-gas density
        if afc:
            #self.detuning = ScalarField.from_expression(grid, f"1*afc(r)", user_funcs={"afc": lambda r: np.random.choice(AFC()[0], p=AFC()[1])}) 
            self.detuning = ScalarField.from_expression(grid, f"afc(r)", user_funcs={"afc": lambda r: AFC(r)}) 
        else:
            self.detuning = ScalarField.from_expression(grid, f"1*{detuning}")
        self.delta_s = delta_s  # two-photon detuning (alkali atoms)
        self.delta_k = delta_k  # two-photon detuning (noble-gas spins)
        self.Da = Da # diffusion coefficient for alkali-metal spins
        self.Db = Db # diffusion coefficient for noble-gas spins
        self.J = J # alkali-noble gas spin-exchange coupling rate
        self.kappa = kappa # cavity decay rate
        self.pulse_shape = pulse_shape # temporal shape of the input pulse
        self.omega = omega # rabi frequency of the control field
        self.control_shape = control_shape # temporal shape of the control pulses
        self.t0 = t0 # start of the memory protocol
        self.protocol = protocol # memory protocol (sequential or adiabatic)
        self.T = T # duration of the input pulse
        self.tdark = tdark # dark time
        self.g = g #np.sqrt(self.cooperativity*self.gamma_p*self.kappa)
        self.fe = ScalarField.from_expression(grid, "1")
        self.fc = ScalarField.from_expression(grid, "1/sqrt(4/3*pi)")
        self.Omega = omega*ScalarField.from_expression(grid, "1", dtype=complex) # spatial-dependent rabi frequency
        self.G = g * np.sqrt(self.na*self.pa) #np.sqrt(self.cooperativity*self.gamma_p*self.kappa)
        self.cooperativity = self.G*self.G/(gamma_p*kappa) # optical cooperativity
        if self.protocol == 'afc':
            self.cooperativity = 1
        if self.protocol == 'sequential' or self.protocol == 'afc':
            self.tpulse = (np.pi*J-self.gamma_s)/(2*J*J) # transfer pulse timing
        else:
            self.tpulse = 0
        self.tcontrol = (np.pi*omega-gamma_p-self.cooperativity)/(2*omega*omega) * 2.046
        self.control_duration = control_duration # time-scale of control field change
        self.tr = 2*self.tpulse+self.tdark+self.t0 # memory time

    def get_initial_state(self, grid): # generate initial state
        P = ScalarField(grid, 0, label=r"$\langle \hat{\mathcal{P}}^{\dag} \hat{\mathcal{P}}  \rangle$", dtype=complex)
        S = ScalarField(grid, 0, label=r"$\langle \hat{\mathcal{S}}^{\dag} \hat{\mathcal{S}}  \rangle$", dtype=complex)
        K = ScalarField(grid, 0, label=r"$\langle \hat{\mathcal{K}}^{\dag} \hat{\mathcal{K}}  \rangle$", dtype=complex)
        return FieldCollection([P, S, K])
    
    def Input_Field(self, t):
        if self.pulse_shape == "exponential" or self.pulse_shape == "exp":
            return np.sqrt(2/self.T) * np.exp((t-self.t0)/self.T) if t <= self.t0 else 0
        elif self.pulse_shape == "gaussian" or self.pulse_shape == "gauss":
            #sigma = self.T / (2 * np.sqrt(2*np.log(2)))
            #return np.sqrt(1/2)  * 1/(2*np.pi) * 1/(sigma**2) * (np.exp(-(t-(self.t0-self.T))**2/(2*sigma**2)) + np.exp(-(t-(self.t0-4*self.T))**2/(2*sigma**2))) 
            #return np.sqrt(2/self.T) * (2/np.pi)**(1/4) * (np.exp(-4*np.log(2)*(t-self.t0+self.T)**2/(self.T)**2) + np.exp(-4*np.log(2)*(t+4*self.T-self.t0)**2/(self.T)**2))
            return np.sqrt((2*np.sqrt(2*np.log(2)))/(self.T*np.sqrt(np.pi)))  * (np.exp(-4*np.log(2)*(t-self.t0+self.T)**2/(self.T)**2) + np.exp(-4*np.log(2)*(t+4*self.T-self.t0)**2/(self.T)**2))
        elif self.pulse_shape == "hyperbolic-secant" or self.pulse_shape == "sech2":
            return 1/(np.cosh((t-self.t0)/self.T))**2 
        else:
            raise ValueError("An invalid shape for the input shape was provided: supported options are exponential or 'exp', gaussian or 'gauss' and hyperbolic-secant or 'sech2'.")
        
    def Magnetic_Detuning(self, t):
        tr = 2*self.tpulse+self.tdark+self.t0
        t1 = self.tpulse + self.t0
        t2 = self.tdark + self.t0 + self.tpulse
        T = self.control_duration
        
        if self.protocol=="sequential" or self.protocol=="afc":
            if self.control_shape == "square":
                return self.delta_k if (t < self.t0) or (t > t1 and t < t2) or (t > tr) else 0
            elif self.control_shape == "hyperbolic-tangent" or self.control_shape == "tanh":
                return self.delta_k / 2 * ( (1 + np.tanh((t-self.t0)/T)) + (1 + np.tanh((t-t1)/T)) - (1 + np.tanh((t-t2)/T)) + (1- np.tanh((t-tr)/T)) )
            else:
                raise ValueError("An invalid shape for the control pulse shape was provided: supported options are 'square', 'blackman' and hyperbolic-tangent or 'tanh'.")
        elif self.protocol=="adiabatic":
            return self.delta_k if (t >= self.t0 and t <= self.tdark + self.t0) else 0
        
        """
        elif self.protocol=="afc":
            t1 = (np.pi*self.J-self.gamma_s)/(2*self.J*self.J) + self.t0
            t2 = (np.pi*self.J-self.gamma_s)/(2*self.J*self.J) + self.t0 + self.tdark
            return self.delta_k if (t < self.t0 + self.tpulse/2 - self.T/2) or (t > t1 and t < t2) or (t > tr) else 0"""
    
    def Rabi_Frequency(self, t):
        tr = 2*self.tpulse+self.tdark+self.t0
        T = self.control_duration

        if self.protocol =="sequential":
            if self.control_shape == "square":
                return self.omega if (t <= self.t0) or (t >= tr) else 0
            elif self.control_shape == "blackman":
                raise ValueError("Unfortunately, Blackman pulses are not yet supported.")
            elif self.control_shape == "tanh":
                return self.omega - (1 + np.tanh((t-self.t0)/T)) +  (1 + np.tanh((t-tr)/T)) 
            else:
                raise ValueError("An invalid shape for the control pulse shape was provided: supported options are 'square', 'blackman' and hyperbolic-tangent or 'tanh'.")
        elif self.protocol=="adiabatic":
            return self.omega if (t <= self.t0 or t >= self.tdark + self.t0) else 0
        elif self.protocol == "afc":
            if (t > self.t0 - self.T and t < self.t0 + self.tcontrol/2 - self.T/2):#or (t > self.tr and t < self.tr + self.tpulse):
                return self.omega
            elif (t > self.t0 - 3.5* self.T and t < self.t0 + self.tcontrol - 3.5* self.T):
                return self.omega
            elif ((t > tr and t < tr + self.tcontrol) or (t > tr + 3*self.T and t < tr + self.tcontrol + 3*self.T)):
                return self.omega
            else:
                return 0

    def evolution_rate(self, state, t=0):
        P, S, K = state
            
        E_in = self.Input_Field(t)
        
        delta_k = self.Magnetic_Detuning(t)
        omega = self.Rabi_Frequency(t)
        
        self.Omega = omega*self.fc
        
        #cell_volumes = P.grid.cell_volumes
        #grid_volume = P.grid.volume
        #integral = (P.data * np.conj(self.G * self.fe.data) * cell_volumes).sum() / cell_volumes.sum()
        
        dP_dt = -(self.gamma_p * (1 + self.cooperativity) + 1j*self.detuning)*P + 1j*self.Omega*S + 1j*np.sqrt(2/self.kappa)*E_in*self.G#4/3*np.pi*self.G
        dS_dt = -(self.gamma_s + 1j*self.delta_s)*S + self.Da*S.laplace(bc=self.bca) + 1j*np.conj(self.Omega)*P - 1j*self.J*K
        dK_dt = -(self.gamma_k + 1j*delta_k)*K + self.Db*K.laplace(bc=self.bcb) - 1j*self.J*S

        return FieldCollection([dP_dt, dS_dt, dK_dt])
    
    def _make_pde_rhs_numba(self, state): # numba-compiled implementation of the system of PDEs
        laplace_a = state.grid.make_operator("laplace", bc=self.bca)
        laplace_b = state.grid.make_operator("laplace", bc=self.bcb)
        t0 = self.t0
        fe = np.array(self.fe.data, dtype=np.complex128)
        fc = np.array(self.fc.data, dtype=np.complex128)
        J = self.J
        tpulse = self.tpulse
        tdark = self.tdark
        tr = 2*self.tpulse+self.tdark+self.t0
        gamma_s = self.gamma_s
        gamma_p = self.gamma_p
        detuning = self.detuning.data
        G = self.G 
        delta_s = self.delta_s
        magnetic_detuning = self.delta_k
        Da = self.Da
        Db = self.Db
        gamma_k = self.gamma_k
        kappa = self.kappa 
        cell_volumes = grid.cell_volumes
        T = self.T
        rabi_frequency = self.omega
        pulse_shape = self.pulse_shape
        protocol = self.protocol
        cooperativity = self.cooperativity
        tcontrol = self.tcontrol

        @nb.njit
        def pde_rhs(data, t):
            P = data[0]
            S = data[1]
            K = data[2]
            rhs = np.empty_like(data)
            
            """ Input Field """
            
            if pulse_shape == "exponential" or pulse_shape == "exp":
                E_in = np.sqrt(2/T) * np.exp((t-t0)/T) if t < t0 else 0 
            elif pulse_shape == "gaussian" or pulse_shape == "gauss":
                #E_in = np.sqrt(2/T) * (2/np.pi)**(1/4) * (np.exp(-4*np.log(2)*(t-t0+T)**2/(T)**2) + np.exp(-4*np.log(2)*(t+4*T-t0)**2/(T)**2))
                #E_in = np.sqrt(np.log(16)/2*np.pi*T) * (np.exp(-4*np.log(2)*(t-t0+T)**2/(T)**2) + np.exp(-4*np.log(2)*(t+4*T-t0)**2/(T)**2))
                #E_in = np.sqrt((np.sqrt(2*np.log(2)))/(T*np.sqrt(np.pi))) * (np.exp(-4*np.log(2)*(t-t0+T)**2/(T)**2) + np.exp(-4*np.log(2)*(t+4*T-t0)**2/(T)**2))
                E_in = np.sqrt(np.sqrt(8*np.log(2))/(T*np.sqrt(np.pi))) * (np.exp(-4*np.log(2)*(t-t0+T)**2/(T)**2) + np.exp(-4*np.log(2)*(t+4*T-t0)**2/(T)**2))
                #E_in = np.sqrt((np.sqrt()))
                #E_in = np.sqrt((np.sqrt(2*np.log(2)))/(T*np.sqrt(np.pi))) * (np.exp(-4*np.log(2)*(t-t0+T)**2/(T)**2) + np.exp(-4*np.log(2)*(t+4*T-t0)**2/(T)**2))
                #sigma = T / (2 * np.sqrt(2*np.log(2)))
                #CHECK THIS!1/1.52 *
                #E_in = np.sqrt(1/2) * 1/(2*np.pi) * 1/(sigma**2) * (np.exp(-(t-(t0-T))**2/(2*sigma**2)) + np.exp(-(t-(t0-4*T))**2/(2*sigma**2))) 
            else:
                raise ValueError("An invalid shape for the input shape was provided: supported options are exponential or 'exp', gaussian or 'gauss' and hyperbolic-secant or 'sech2'.")
                
            """ Magnetic Detuning """
                 
            if protocol == "sequential" or protocol == "afc":
                if (t < t0) or (t > tpulse + t0 and t < tdark + t0 + tpulse) or (t > tr):
                    delta_k = magnetic_detuning
                else:
                    delta_k = 0
        
            elif protocol == "adiabatic":
                if (t > t0 and t < tdark + t0):
                    delta_k = magnetic_detuning
                else:
                    delta_k = 0
                    
            #elif protocol == "afc":
                #tpulse1 = (np.pi*J-gamma_s)/(2*J*J)
                #if (t < t0 + tpulse/2 - T/2) or (t > tpulse + t0 and t < tdark + t0 + tpulse) or (t > tr):
                    #delta_k = magnetic_detuning
                #else:
                    #delta_k = 0
                
            """ Control Field """

            if protocol == "sequential":
                if (t < t0) or (t > tr):
                    omega = rabi_frequency
                else:
                    omega = 0
                
            elif protocol == "adiabatic":
                if (t < t0) or (t > tdark + t0):
                    omega = rabi_frequency
                else:
                    omega = 0
                    
            elif protocol == "afc":
                if (t > t0 - T/2 and t < t0 + tcontrol/2 - T/2):  # (t > tr and t < tr + tpulse):
                    omega = rabi_frequency
                elif (t > t0 - 3.5*T and t < t0 + tcontrol - 3.5*T):
                    omega = rabi_frequency
                elif ((t > tr + T/2 and t < tr + tcontrol/2 + T/2) or (t > tr + 3.5*T and t < tr + tcontrol + 3.5*T)):
                    omega = rabi_frequency
                else:
                    omega = 0
            
            """ Polarization """
            
            #integral = 0
            #for i in range(data[0].size):
                #integral += P[i]*np.conj(G*fe[i])*cell_volumes[i]
            #integral/= cell_volumes.sum()
            
            """ PDE RHS """
            
            rhs[0] = -(gamma_p * (1 + cooperativity) + 1j*detuning)*P + 1j*omega*fc*S + 1j*np.sqrt(2/kappa)*E_in*G#4/3*np.pi*G
            rhs[1] = -(gamma_s + 1j*delta_s)*S + Da*laplace_a(S) + 1j*np.conj(omega*fc)*P - 1j*J*K
            rhs[2] = -(gamma_k + 1j*delta_k)*K + Db*laplace_b(K) - 1j*J*S
        
            return rhs
        
        return pde_rhs
"""  
def AFC(separation=2.67, peak_width=0.13, overall_width=3*(2.67+0.13), num_peaks=2, M=10000):
    detunings = np.linspace(-0.5*overall_width, 0.5*overall_width, M)
    n =  []
    for delta in detunings:
        s = np.exp(-(delta)**2/(2*peak_width**2))
        for m in range(1, num_peaks):
            s += np.exp(-(delta-m*separation)**2/(2*peak_width**2))
            s += np.exp(-(delta+m*separation)**2/(2*peak_width**2))
        n.append(np.exp(-delta**2/(2*overall_width**2)) * s)
    
    n /= sum(n) # normalization
    
    print(detunings)
    print(n)
        
    return detunings, n"""

def AFC(r, separation=2.23, num_peaks=15):
    
    # k = np.random.randint(0, 1) 
    Gamma = (num_peaks) * separation #* 2
    j = np.random.randint(-num_peaks, num_peaks+1, ngrid) #* (-1) ** np.random.randint(0, 1) 
    
    peaks = np.arange(-num_peaks+1, num_peaks)
    
    weights = [np.exp(-(peak*separation)**2/(2*Gamma**2)) for peak in peaks ]
    
    #weights[num_peaks-1] = 2 * weights[num_peaks-1]
    
    j = np.random.choice(peaks, ngrid, p = weights/np.sum(weights))
    
    fig, ax = plt.subplots(figsize =(10, 7))
    ax.hist(j)
    plt.show()
    
    #print(j*separation)
    
    return j*separation