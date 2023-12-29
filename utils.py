# -*- coding: utf-8 -*-
"""
Author: Alexandre Barbosa
Contact: alexandre.barbosa@tecnico.ulisboa.pt
Last Updated: 05-11-2023
"""
import scipy
import numpy as np
import csv
from dynamics import CollectiveSpinsPDE

from pde import FileStorage, ScalarField

adiabatic = CollectiveSpinsPDE(protocol='adiabatic', T=5.937e5, t0=1e7, tdark=1e7, omega=0.17898, delta_k=1.85e-5*200)

def write_magnitudes(read_filename, write_filename="magnitudes.csv"):
    
    reader = FileStorage(read_filename, write_mode="read_only")
    
    f = open(write_filename, "a") # open file 
    
    f.write("t, P, S, K \n")
    
    for time, collection in reader.items():
        P, S, K = collection.fields
        f.write(f"{time}, {P.magnitude*P.magnitude}, {S.magnitude*S.magnitude}, {K.magnitude*K.magnitude} \n")
        
    f.close() # close opened file
    
def write_parameters(simulation, write_filename="parameters.txt"):
    
    attrs = vars(simulation)
    
    f = open(write_filename, "a")
    
    f.write(', '.join("%s: %s" % item for item in attrs.items()))
    
    f.close()
     
def output_field(read_filename, params, write_filename):
    
    f = open(write_filename, "a") # open file
    
    with open(read_filename,'r') as csvfile:
        magnitudes = csv.reader(csvfile, delimiter = ',')
        headers = next(magnitudes)
        for row in magnitudes:
            try: 
                if params.protocol != 'afc':
                    output_field = np.sqrt(2/params.kappa)*np.sqrt(float(row[1])) * params.G * np.sqrt(params.T/2)
                    output_field = output_field**2
                    f.write(f"{float(row[0])}, {output_field} \n")
                elif params.protocol == 'afc':
                    output_field = np.sqrt(2/params.kappa)*np.sqrt(float(row[1])) * params.G / np.sqrt(np.sqrt(4*np.log(2))/(params.T*np.sqrt(np.pi))) # np.sqrt(2*(2*np.sqrt(2*np.log(2)))/(params.T*np.sqrt(np.pi))) 
                    output_field = output_field**2
                    f.write(f"{float(row[0])}, {output_field} \n")
            except ValueError:
                pass
    
    f.close()

afc = CollectiveSpinsPDE(protocol='afc', t0=1, T=2*0.03745, pulse_shape='gaussian', omega=100*2.04, afc=True, kappa=0.9865) 
#output_field('afc_1024_30.csv', afc, 'afc_1024_30_EOut.csv')
    
def efficiency(filename, params):
    
    efficiency = 0
    with open(filename,'r') as csvfile:
        magnitudes = csv.reader(csvfile, delimiter = ',')
        headers = next(magnitudes)
        for row in magnitudes:
            try: 
                if float(row[0]) >= params.tr:
                    if float(row[1]) > efficiency:
                        efficiency = float(row[1])
            except ValueError:
                pass
            
    print(f"Total Efficiency: {efficiency}") 
            
#efficiency('sequential_thesis_Eout.csv', CollectiveSpinsPDE())
#efficiency('adiabatic_8_Eout.csv', CollectiveSpinsPDE())

def efficiency_2(filename, params):
    
    time = []
    E_out = []
    with open(filename,'r') as csvfile:
        magnitudes = csv.reader(csvfile, delimiter = ',')
        headers = next(magnitudes)
        for row in magnitudes:
            try: 
                if float(row[0]) >= params.tr:
                    time.append(float(row[0]))
                    E_out.append(float(row[1]))
            except ValueError:
                pass
            
    efficiency = scipy.integrate.simpson(E_out, time) * 2/params.T
    
    if params.protocol == 'afc':
        efficiency = scipy.integrate.simpson(E_out, time) * (2*np.sqrt(2*np.log(2)))/(params.T*np.sqrt(np.pi)) / 1.52 * 2
            
    print(f"Total Efficiency: {efficiency}") 
    
#efficiency_2('sequential_thesis_Eout.csv', CollectiveSpinsPDE())
#efficiency_2('adiabatic_8_Eout.csv', adiabatic)
efficiency_2('afc_1024_30_Eout.csv', afc)

def efficiency_out(filename, params):
    
    reader = FileStorage(filename, write_mode="read_only")
    
    SS = []
    PP = []
    #K0 = 0
    t =  []
    
    for time, collection in reader.items(): 
        P, S, K, E = collection.fields
        if (time >= params.t0 + params.tdark + params.tpulse):
            t.append(time)
            SS.append(S.magnitude)
            PP.append(P.magnitude)
            #if (K.magnitude > K0):
                #K0 = K.magnitude
        
    #intS = 2*params.gamma_s*np.trapz(SS, t)
    intS = 2*params.gamma_s*scipy.integrate.simpson(SS, t)
    intP = 2*params.gamma_p*scipy.integrate.simpson(PP, t)
    
    #efficiency = 1 - (intS)/K0
    #efficiency = 1- intS2/K0
    efficiency = 1 - intS + intP
    
    print(efficiency)
    return efficiency

#efficiency_out('sequential_thesis_Eout.csv', CollectiveSpinsPDE())
#efficiency_out('adiabatic_8_Eout.csv', CollectiveSpinsPDE())
    
def fom(filename, simulation, parameter, reader):
    
    E_in = 0 # input field magnitude
    E_out = 0 # output field magnitude
    #K_out = ScalarField.from_expression(grid, "0")
    #S_in = ScalarField.from_expression(grid, "0")
    S0 = 0 # maximum alkali spin excitation (storage)
    S1 = 0 # maximum alkali spin excitation (retrieval)
    K0 = 0 # maximum noble-gas spin excitation (storage)

    for time, collection in reader.items(): 
        P, S, K, E = collection.fields

        if S.magnitude > S0: # alkali storage
            S0 = S.magnitude # average value
            #S_in = S / S.magnitude # spatial function
            E_in = S.magnitude*S.magnitude + K.magnitude*K.magnitude
            
        if S.magnitude > S1 and time > simulation.tr: # retrieval
            S1 = S.magnitude  # average value
            E_out = S.magnitude*S.magnitude + K.magnitude*K.magnitude
        
        """if K.magnitude > K0: # noble-gas storage
            K0 = K.magnitude # average value
            K_out = K # spatial function"""
    
    """ Transfer Fidelity """
    """
    cell_volumes = grid.cell_volumes
    overlap = (K_out.data / S0  * np.conj(S_in.data) * cell_volumes).sum() / (cell_volumes.sum())
    transfer_fidelity = np.abs(overlap**2) """
    
    """ Transfer Efficiency """
    #transfer_efficiency = np.abs(K0*K0/S0*S0)
    
    """ Total Efficiency """
    if E_in > 0:
        efficiency = E_out/E_in
    else:
        efficiency = 0
    
    """ Print in Terminal """
    print(f"{parameter} :  Efficiency = {efficiency} ")
    
    """ Write to File """
    f = open("cooperativity.csv", "a") # open file to write figures of merit
    #f.write(f"{parameter}, {transfer_fidelity}, {transfer_efficiency}, {efficiency} \n")
    f.write(f"{parameter}, {efficiency} \n")
    f.close() # close opened file
    
    E_in = 0 # input field magnitude
    E_out = 0 # output field magnitude
    #K_out = ScalarField.from_expression(grid, "0")
    #S_in = ScalarField.from_expression(grid, "0")
    S0 = 0 # maximum alkali spin excitation (storage)
    S1 = 0 # maximum alkali spin excitation (retrieval)
    K0 = 0 # maximum noble-gas spin excitation (storage)
    
    reader.clear()
    reader.close()