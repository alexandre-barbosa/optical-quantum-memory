# -*- coding: utf-8 -*-
"""
Author: Alexandre Barbosa
Contact: alexandre.barbosa@tecnico.ulisboa.pt
Last Update: 18-07-2023

TODO: Add noise (?), write to folder
"""

import pde
import warnings
import matplotlib.pyplot as plt
import scienceplots
from pde import config, FileStorage
from pde.trackers import ProgressTracker
#from pde.solvers import ScipySolver, ImplicitSolver
from geometry import grid
from utils import write_magnitudes, write_parameters, output_field, efficiency
from plotting import plot_magnitude, plot_kymograph, plot_controls, plot_fields
from dynamics import CollectiveSpinsPDE, AFC
from datetime import datetime
from numba.core.errors import NumbaDeprecationWarning
config["numba.fastmath"] = True # False
warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
startTime = datetime.now()

plt.rcParams.update({'figure.dpi': '600'})
plt.style.use(['science', 'notebook', 'grid'])
plt.rcParams["font.family"] = "serif" #serif
plt.rcParams["mathtext.fontset"] = "cm"

def run_simulation(dt=0.05, filename='fields.hdf5', **args):

    # initialize the instance of PDEBase class, forwarding simulation parameters
    eq = CollectiveSpinsPDE(**args)
    # setup initial state of the system
    state = eq.get_initial_state(grid) 
    
    # write simulation parameters
    write_parameters(eq, "{filename}_parameters.txt".format(filename=filename))
    
    # write values of fields over time to file
    storage = FileStorage('{filename}.hdf5'.format(filename=filename), write_mode="truncate")
    writer =  FileStorage('{filename}.hdf5'.format(filename=filename), write_mode="append")
    progress = ProgressTracker(1000) 
    
    # solve system of pde

    if eq.protocol == "sequential":
        result =  eq.solve(state,  t_range=(0, eq.t0*0.9), dt=0.05, tracker=[storage.tracker(1), progress], adaptive=True, scheme='rk', backend='numba', tolerance=1e-8)
        result2 = eq.solve(result,  t_range=(eq.t0*0.9, eq.t0*1.05), dt=dt, tracker=[writer.tracker(0.1), progress], adaptive=True, scheme='rk', backend='numba', tolerance=1e-8)
        result3 = eq.solve(result2, t_range=(eq.t0*1.05, eq.tr*0.99), dt=1, adaptive=True, tracker=[writer.tracker(100), progress], scheme='rk', backend='numba', tolerance=1e-8)
        result4 = eq.solve(result3, t_range=(eq.tr*0.99, eq.tr*1.05), dt=dt, tracker=[writer.tracker(0.1), progress], adaptive=True,  backend='numba', tolerance=1e-10)
    
    elif eq.protocol == "adiabatic":
        result = eq.solve(state,  t_range=(0, eq.t0*0.99), dt=1, tracker=[storage.tracker(10), progress], adaptive=True, scheme='rk', backend='numba', tolerance =1e-8)
        result2 = eq.solve(result,  t_range=(eq.t0*0.99, eq.tdark+eq.t0*1.01), dt=1, tracker=[writer.tracker(1000), progress], adaptive=True, scheme='rk', backend='numba', tolerance=1e-8)
        result3 = eq.solve(result2, t_range=(eq.tdark+eq.t0*1.01, eq.tdark+2.1*eq.t0), dt=1, tracker=[writer.tracker(10), progress], adaptive=True, scheme='rk', backend='numba', tolerance=1e-8)
        
    elif eq.protocol == "afc":
        result =  eq.solve(state,  t_range=(0, eq.t0*0.9), dt=dt, tracker=[storage.tracker(dt*5), progress], adaptive=True, scheme='rk', backend='numba', tolerance=1e-8)
        result2 = eq.solve(result,  t_range=(eq.t0*0.9, eq.t0*1.5), dt=dt, tracker=[writer.tracker(dt*2), progress], adaptive=True, scheme='rk', backend='numba', tolerance=1e-8)
        result3 = eq.solve(result2, t_range=(eq.t0*1.5, eq.tr*0.999), dt=1, adaptive=True, tracker=[writer.tracker(10), progress], scheme='rk', backend='numba', tolerance=1e-8)
        result4 = eq.solve(result3, t_range=(eq.tr*0.999, eq.tr*1.001), dt=dt, tracker=[writer.tracker(dt*5), progress], adaptive=True,  backend='numba', tolerance=1e-6)
        
    print("Simulation Runtime: {runtime}".format(runtime=datetime.now() - startTime))  # print runtime

    # read the field values over time
    reader = FileStorage('{filename}.hdf5'.format(filename=filename), write_mode="read_only")
    #pde.plot_kymographs(reader, transpose=True) # plot fields as a function of time and space
    pde.plot_magnitudes(reader)  # plot the spatial average of fields over time
    # result.plot(kind="image") # plot the spatial profile of fields at t=tmax
    # pde.movie_multiple(reader, "movie_spins.mov") # movie containing the time evolution of fields
    
    write_magnitudes('{filename}.hdf5'.format(filename=filename),'{filename}.csv'.format(filename=filename))
    plot_magnitude('{filename}.csv'.format(filename=filename), eq, 'P', '{filename}_P.png'.format(filename=filename))
    plot_magnitude('{filename}.csv'.format(filename=filename), eq, 'S', '{filename}_S.png'.format(filename=filename))
    output_field('{filename}.csv'.format(filename=filename), eq, '{filename}_EOut.csv'.format(filename=filename))
    plot_fields('{filename}_EOut.csv'.format(filename=filename), eq, '{filename}_E.png'.format(filename=filename))
    efficiency('{filename}_EOut.csv'.format(filename=filename), eq)
    
    """
    write_magnitudes('{filename}.hdf5'.format(filename=filename),'{filename}.csv'.format(filename=filename))
    output_field('{filename}.csv'.format(filename=filename), eq, '{filename}_EOut.csv'.format(filename=filename))
    plot_magnitude('{filename}.csv'.format(filename=filename), eq, 'S', '{filename}_S.png'.format(filename=filename))
    plot_magnitude('{filename}.csv'.format(filename=filename), eq, 'K', '{filename}_K.png'.format(filename=filename))
    plot_magnitude('{filename}.csv'.format(filename=filename), eq, 'P', '{filename}_P.png'.format(filename=filename))
    plot_fields('{filename}_EOut.csv'.format(filename=filename), eq, '{filename}_E.png'.format(filename=filename))
    plot_controls('{filename}.csv'.format(filename=filename), eq, '{filename}_deltak.png'.format(filename=filename), '{filename}_omega.png'.format(filename=filename))
    plot_kymograph('{filename}.hdf5'.format(filename=filename), eq,'S', '{filename}_kymoS.png'.format(filename=filename))
    plot_kymograph('{filename}.hdf5'.format(filename=filename), eq,'K', '{filename}_kymoK.png'.format(filename=filename))
    efficiency('{filename}_EOut.csv'.format(filename=filename), eq)
    """
    
    print("Total Runtime: {runtime}".format(runtime=datetime.now() - startTime))  # print runtime
    writer.close()
    reader.close()
    
    
""" Run Simulation """
    
    
#run_simulation(dt=.5, filename='adiabatic_8', protocol='adiabatic', T=5.937e5, t0=1e7, tdark=1e7, omega=0.17898, delta_k=1.85e-5*200)

run_simulation(dt=5e-4, filename='afc_tese', protocol='afc', t0=1, T=2*0.03745, pulse_shape='gaussian', omega=100*2.04, afc=True, kappa=2.484) #0.125 #.9991 , tdark=0