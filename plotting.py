# -*- coding: utf-8 -*-
"""
Author: Alexandre Barbosa
Contact: alexandre.barbosa@tecnico.ulisboa.pt
Last Updated: 05-11-2023
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc, rcParams
import scienceplots
import csv 
from dynamics import CollectiveSpinsPDE
from pde import FileStorage, ScalarField

""" Figure Style """

plt.rcParams.update({'figure.dpi': '600'})
plt.style.use(['science', 'notebook', 'grid'])
plt.rcParams["font.family"] = "serif" #serif
plt.rcParams["mathtext.fontset"] = "cm"

#adiabatic = CollectiveSpinsPDE(protocol='adiabatic', T=5.937e5, t0=1e7, tdark=1e7, omega=0.17898, delta_k=1.85e-5*200)
afc = CollectiveSpinsPDE(protocol='afc', t0=1, T=2*0.03745, pulse_shape='gaussian', omega=100*2.04, afc=True, kappa=0.9865) #0.125
afc_test = CollectiveSpinsPDE(protocol='afc', t0=1, T=2*0.03745, pulse_shape='gaussian', omega=100*2.04, afc=True, kappa=0.9865, tdark=0)

def plot_kymograph(read_filename, sim, operator='S', filename='kymograph.pdf'):
    
    reader = FileStorage(read_filename, write_mode="read_only")
    
    field = []
    t =  []
    
    for time, collection in reader.items(): 
        P, S, K = collection.fields
        
        if operator == 'S':
            field.append(np.abs(S.data)*np.abs(S.data))
            
        elif operator == 'K':
            field.append(np.abs(K.data)*np.abs(K.data))
            
        t.append(time)
        
    """ Spherical Grid """
    npoints = len(field[0])
    r = []
    for i in range(npoints):
        r.append((i+0.5)/npoints)
        
    X, Y = np.meshgrid(t, r)    
    if operator == 'S':
        plt.contourf(X, Y, np.transpose(field), levels=100, cmap='OrRd')
        plt.title(r'$\langle \hat{\mathcal{S}}^{\dag} \hat{\mathcal{S}} \rangle $', fontsize=22, pad=20)
    elif operator == 'K': 
        plt.contourf(X, Y, np.transpose(field), levels=80, cmap='PuBu')
        plt.title(r'$\langle \hat{\mathcal{K}}^{\dag} \hat{\mathcal{K}} \rangle $', fontsize=22, pad=20)
    plt.ylabel(r'$r/R$', fontsize=20)
    
    if sim.protocol == 'sequential':
        plt.xticks([sim.t0 , sim.tpulse + sim.t0, sim.tpulse + sim.t0+ sim.tdark, sim.tr],
                [r'$0$', r'$T_\pi$', r'$T_D + T_\pi$', r'$T_R$'], fontsize=18)
        
    elif sim.protocol == 'adiabatic':
        plt.xticks([sim.t0 , sim.t0+ sim.tdark],
                [r'$0$', r'$T_R$'], fontsize=18)
        
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=18)
    plt.minorticks_off()
    plt.colorbar(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.savefig(filename)
    plt.show()  
    
#plot_kymograph('adiabatic_8.hdf5', adiabatic, 'K', 'adiabatic_kymoK.png')

def plot_magnitude(filename, sim, op, savefile):

    t = []
    P = []    
    S = []
    K = []
    E = []
    
    with open(filename,'r') as csvfile:
        magnitudes = csv.reader(csvfile, delimiter = ',')
        headers = next(magnitudes)
        for row in magnitudes:
            try: 
                t.append(float(row[0]))
                P.append(float(row[1]))
                S.append(float(row[2]))
                K.append(float(row[3]))
            except ValueError:
                pass
            
    S = np.array(S)
    E = np.array(E)
    t = np.array(t)
    
    if op == 'S':
        plt.plot(t, S, color='crimson')
        plt.ylabel(r'$\langle \hat{\mathcal{S}}^{\dag} \hat{\mathcal{S}} \rangle $', fontsize=22)
        plt.fill_between(t, S, alpha=0.25, interpolate=True, color='orangered')
        
    elif op == 'K': 
        plt.plot(t, K, color='blue', label=r'$\langle \mathcal{K}^{\dag} \mathcal{K} \rangle $')
        plt.ylabel(r'$\langle \hat{\mathcal{K}}^{\dag} \hat{\mathcal{K}} \rangle $', fontsize=22)
        plt.fill_between(t, K, alpha=0.25, interpolate=True, color='blue')

    if op != 'P' and sim.protocol == 'sequential':
        plt.xticks([-5000, sim.t0 , sim.tpulse + sim.t0, sim.tpulse + sim.t0+ sim.tdark, sim.tr],
                [r'$-\infty$', '$0$', r'$T_\pi$', r'$T_D + T_\pi$', r'$T_R$'], fontsize=18)
        plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=18)
        plt.minorticks_off()
        plt.savefig(savefile)
        plt.show() 
        
    elif op!= 'P' and sim.protocol == 'afc':
        plt.xticks([sim.t0 , sim.tpulse + sim.t0, sim.tpulse + sim.t0+ sim.tdark, sim.tr],
                [r'$-\infty$$0$', r'$T_\pi$', r'$T_D + T_\pi$', r'$T_R$'], fontsize=18)
        plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0], fontsize=18)
        plt.minorticks_off()
        plt.savefig(savefile)
        plt.show() 
        
    elif op!= 'P' and sim.protocol == 'adiabatic':
        plt.xticks([-5000, sim.t0 , sim.t0+ sim.tdark],
                [r'$-\infty$', '$0$', r'$T_R$'], fontsize=18)
        #plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=18)
        if op == 'S':
            #plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            plt.yticks([0, 0.002, 0.004, 0.006, 0.008, 0.01], fontsize=18)
        plt.minorticks_off()
        plt.savefig(savefile)
        plt.show() 
        
    else:
        plot_polarization(filename, sim, savefile)
        
#plot_magnitude('adiabatic_8.csv', adiabatic, 'S', 'adiabatic_S.png')
#plot_magnitude('afc_1024_30.csv', afc, 'K', 'afc_1024_30_K.png')

    
def plot_controls(filename, params, savefile1='magnetic_detuning.pdf', savefile2='control_field.pdf'):
    tvalues = []
    
    with open(filename,'r') as csvfile:
        magnitudes = csv.reader(csvfile, delimiter = ',')
        headers = next(magnitudes)
        for row in magnitudes:
            try: 
                tvalues.append(float(row[0]))
            except ValueError:
                pass
        
        tvalues = np.array(tvalues)
        
        t1 = params.tpulse + params.t0
        t2 = params.tdark + params.t0 + params.tpulse

        magnetic_detuning =  [params.delta_k/params.J if (t < params.t0) or (t > t1 and t < t2) or (t > params.tr) else 0 for t in tvalues]
        control_field = [params.omega/params.gamma_s if (t <= params.t0) or (t >= params.tr) else 0 for t in tvalues]
        
        # detuning
        plt.plot(tvalues, magnetic_detuning, color='seagreen')
        plt.ylabel(r'$\delta_k / J$', fontsize=22)
        plt.fill_between(tvalues, magnetic_detuning, alpha=0.25, interpolate=True, color='seagreen')
        plt.xticks([params.t0 , params.tpulse + params.t0, params.tpulse + params.t0+ params.tdark, params.tr],
                [r'$0$', r'$T_\pi$', r'$T_D + T_\pi$', r'$T_R$'], fontsize=18)
        plt.yticks([10, 20, 30, 40, 50], fontsize=18)
        plt.ylim(-0.01, 55)
        plt.minorticks_off()
        plt.savefig(savefile1)
        plt.show() 
        
        fig,(ax1,ax2) = plt.subplots(1, 2, sharey=True)
        
        # plot the same data on both axes
        ax1.plot(tvalues, control_field, color='indigo')
        ax2.plot(tvalues, control_field, color='indigo')
            
        ax1.fill_between(tvalues, control_field , alpha=0.2, interpolate=True, color='indigo')
        ax2.fill_between(tvalues, control_field, alpha=0.2, interpolate=True, color='indigo')
        ax1.set_ylabel(r'$\Omega / \gamma_s$', fontsize=22)

        ax1.set_xlim(params.t0/2, params.t0+params.T)
        ax2.set_xlim(params.tr-params.T, params.tr+params.t0/2)
        
        ax1.set_ylim(0, 1.7e6)
        ax2.set_ylim(0, 1.7e6)

        # hide the spines between ax and ax2
        ax1.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax1.yaxis.tick_left()
        ax2.tick_params(labelright='off')
        ax2.yaxis.tick_right()

        d = .015  # how big to make the diagonal lines in axes coordinates
        # arguments to pass plot, just so we don't keep repeating them
        kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
        ax1.plot((1-d, 1+d), (-d, +d), **kwargs)
        ax1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
        
        kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
        ax2.plot((-d, +d), (1-d, 1+d), **kwargs)
        ax2.plot((-d, +d), (-d, +d), **kwargs)
        
        ax1.set_xticks([params.t0/2, params.t0], [r'-$\infty$', '$0$'], fontsize=18)
        #ax1.set_yticks([0, 5e5, 1e6, 1.5e6, 2e6])
        plt.minorticks_off()
        ax2.set_xticks([params.tr, params.tr+params.t0/2], [r'$T_R$', '$+ \infty$'],  fontsize=18)
        #ax2.set_yticks([0, 5e5, 1e6, 1.5e6, 2e6])
        plt.minorticks_off()
        
        plt.savefig(savefile2)
        plt.show()
    

def plot_fields(filename, params, savefile="fields.png"):
    
    tvalues = []
    field_out = []
    
    with open(filename,'r') as csvfile:
        magnitudes = csv.reader(csvfile, delimiter = ',')
        headers = next(magnitudes)
        for row in magnitudes:
            try: 
                tvalues.append(float(row[0]))
                if params.protocol != 'afc':
                    field_out.append(float(row[1]))
                else:
                    field_out.append(float(row[1])/2)
            except ValueError:
                pass
            
    tvalues = np.array(tvalues)
    
    if params.pulse_shape == 'exp' or params.pulse_shape == 'exponential':  
        field_in =  [np.exp (2*(t-params.t0)/params.T) if t < params.t0 else 0 for t in tvalues]
        
    elif params.pulse_shape == 'gaussian' or params.pulse_shape == "gauss":
        #field_in = [np.sqrt(2/params.T) * (2/np.pi)**(1/4) *(np.exp(-4*np.log(2)*(t-params.t0+params.T)**2/(params.T)**2)  + np.exp(-4*np.log(2)*(t+4*params.T-params.t0)**2/(params.T)**2)) for t in tvalues]
        #field_in = [np.sqrt(np.log(16)/2*np.pi*params.T) *(np.exp(-4*np.log(2)*(t-params.t0+params.T)**2/(params.T)**2)  + np.exp(-4*np.log(2)*(t+4*params.T-params.t0)**2/(params.T)**2)) for t in tvalues]
        #field_in = [(2*np.sqrt(2*np.log(2)))/(params.T*np.sqrt(np.pi)) *(np.exp(-4*np.log(2)*(t-params.t0+params.T)**2/(params.T)**2)  + np.exp(-4*np.log(2)*(t+4*params.T-params.t0)**2/(params.T)**2)) for t in tvalues]
        field_in =  [(np.exp(-8*np.log(2)*(t-params.t0+params.T)**2/(params.T)**2) + np.exp(-8*np.log(2)*(t+4*params.T-params.t0)**2/(params.T)**2)) for t in tvalues] #sigma = params.T / (2 * np.sqrt(2*np.log(2)))
        #field_in = [np.sqrt(1/2)  * np.sqrt(1/(2*np.pi)) * 1/sigma * (np.exp(-(t-(params.t0-params.T))**2/(2*sigma**2)) + np.exp(-(t-(params.t0-4*params.T))**2/(2*sigma**2))) for t in tvalues] 
        #field_out = [efficiency* np.exp(-(t-params.tr)/params.T) if t > params.tr else 0 for t in tvalues]
        # (2*np.sqrt(2*np.log(2)))/(params.T*np.sqrt(np.pi))  
        
    fig,(ax1,ax2) = plt.subplots(1, 2, sharey=True)
    
    # plot the same data on both axes
    ax1.plot(tvalues, field_in , color='coral')
    ax2.plot(tvalues, field_out, color='darkorange')
        
    ax1.fill_between(tvalues, field_in , alpha=0.25, interpolate=True, color='coral', label=r'$\langle \hat{\mathcal{E}}_{\mathrm{in}}^{\dagger} \hat{\mathcal{E}}_{\mathrm{in}} \rangle$')
    ax2.fill_between(tvalues, field_out, alpha=0.2, interpolate=True, color='darkorange', label=r'$\langle \hat{\mathcal{E}}_{\mathrm{out}}^{\dagger} \hat{\mathcal{E}}_{\mathrm{out}} \rangle$')
    ax1.set_ylabel(r'$\langle \hat{\mathcal{E}}^{\dag} \hat{\mathcal{E}} \rangle $', fontsize=22)

    if params.protocol == 'sequential':
        ax1.set_xlim(params.t0/2, params.t0+params.T)
        ax2.set_xlim(params.tr-params.T, params.tr+params.t0/2)
        
    elif params.protocol == 'adiabatic':
        ax1.set_xlim(params.t0/2, params.t0+params.T)
        ax2.set_xlim(params.tdark+params.t0-params.T, params.tdark+3*params.t0/2)
        
    elif params.protocol == 'afc':
        ax1.set_xlim(params.t0-6*params.T, params.t0+2*params.T)
        ax2.set_xlim(params.tr-2*params.T, params.tr+6*params.T)
    
    if params.protocol != 'afc':
        ax1.set_ylim(0, 1.05)
        ax2.set_ylim(0, 1.05)
        
    else:
        ax1.set_ylim(0, 1.1)
        ax2.set_ylim(0, 1.1)

    # hide the spines between ax and ax2
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax1.yaxis.tick_left()
    ax2.tick_params(labelright='off')
    ax2.yaxis.tick_right()

    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1-d, 1+d), (-d, +d), **kwargs)
    ax1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
    
    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1-d, 1+d), **kwargs)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    
    ax1.set_xticks([params.t0/2, params.t0], [r'-$\infty$', '$0$'], fontsize=18)
    plt.minorticks_off()
    if params.protocol == 'sequential':
        ax2.set_xticks([params.tr, params.tr+params.t0/2], [r'$T_R$', '$+\infty$'], fontsize=18)
    elif params.protocol == 'adiabatic':
        ax2.set_xticks([params.tdark+params.t0, params.tdark+3*params.t0/2], [r'$T_R$', '$+\infty$'], fontsize=18)
        
    elif params.protocol == 'afc':
        ax2.set_xticks([params.tr+params.T/2, params.tr+8*params.T], [r'$\ T_R + \frac{2 \pi}{\Delta}$', '$+\infty$'], fontsize=18)
    #plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=18)
    plt.minorticks_off()
    
    #ax1.legend()
    #ax2.legend()
    
    plt.savefig(savefile)
    plt.show()
    
#plot_fields('afc_1024_30_EOut.csv', afc, 'afc_1024_30_E.png')

def plot_polarization(filename, params, savefile="polarization.pdf"):
    
    PP = []
    t =  []
    
    with open(filename,'r') as csvfile:
        magnitudes = csv.reader(csvfile, delimiter = ',')
        headers = next(magnitudes)
        for row in magnitudes:
            try: 
                t.append(float(row[0]))
                PP.append(float(row[1]))
            except ValueError:
                pass  
        
    fig,(ax1,ax2) = plt.subplots(1, 2, sharey=True)
    
    # plot the same data on both axes
    ax1.plot(t, PP, color='lightseagreen')
    ax2.plot(t, PP, color='lightseagreen')
        
    ax1.fill_between(t, PP , alpha=0.2, interpolate=True, color='lightseagreen')
    ax2.fill_between(t, PP, alpha=0.2, interpolate=True, color='lightseagreen')
    ax1.set_ylabel(r'$\langle \hat{\mathcal{P}}^{\dag} \hat{\mathcal{P}} \rangle $', fontsize=22)

    #ax1.set_xlim(params.t0/2, params.t0+params.T)
    #ax2.set_xlim(params.tr-params.T, params.tr+params.t0/2-params.T)
    if params.protocol == 'sequential' or params.protocol =='afc':
        ax1.set_xlim(params.t0/2, params.t0+params.T)
        ax2.set_xlim(params.tr-params.T, params.tr+params.t0/2)
        
    elif params.protocol == 'adiabatic':
        ax1.set_xlim(params.t0/2, params.t0+params.T)
        ax2.set_xlim(params.tdark+params.t0-params.T, params.tdark+3*params.t0/2)
    #ax1.set_ylim(0, 1e-6)
    #ax2.set_ylim(0, 1e-6)

    # hide the spines between ax and ax2
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax1.yaxis.tick_left()
    ax2.tick_params(labelright='off')
    ax2.yaxis.tick_right()

    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1-d, 1+d), (-d, +d), **kwargs)
    ax1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
    
    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1-d, 1+d), **kwargs)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    
    ax1.set_xticks([params.t0/2, params.t0], [r'-$\infty$', '$0$'], fontsize=18)
    plt.minorticks_off()
    #ax2.set_xticks([params.tr, params.tr+params.t0/2], [r'$T_R$', '$+\infty$'], fontsize=18)
    if params.protocol == 'sequential':
        ax2.set_xticks([params.tr, params.tr+params.t0/2], [r'$T_R$', '$+\infty$'], fontsize=18)
    elif params.protocol == 'afc':
        ax2.set_xticks([params.tr+params.T/2, params.tr+params.t0/2], [r'$T_R$', '$+\infty$'], fontsize=18)
    elif params.protocol == 'adiabatic':
        ax2.set_xticks([params.tdark+params.t0, params.tdark+3*params.t0/2], [r'$T_R$', '$+\infty$'], fontsize=18)
    #plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=18)
    plt.minorticks_off()
    
    plt.savefig(savefile)
    plt.show()

#plot_polarization('afc_201.csv', afc, 'afc_P.png')
#plot_polarization('afc_testing6.csv', afc_test, 'afc_1024_30_S2.png')