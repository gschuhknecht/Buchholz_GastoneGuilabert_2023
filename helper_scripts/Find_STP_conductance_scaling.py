# HEADER Compute how each EPSP should be scaled as a function of its PPR and spike times during the 100 sets of spike trains
# HEADER Load the Theta parameters and use Equ. (3) and (4) to compute PPRs for arbitrary inter-spike intervals
# HEADER Save this trace as '.../Synaptic_conductance_scaling.npy'


# Header: Import dependencies ==========================================================================================
import numpy as np
import pandas as pd
import sys

# TODO: enter the correct path:
sys.path.append('.../Buchholz_GastoneGuilabert_2023/helper_scripts')
from modules_LIF_simulation import Costa_Functions
import matplotlib.pyplot as plt

def Find_STP_conductance_scaling(global_files,experiment_path, Theta_parameters,input_plotting):

    # Header: import data: spike times, PPRs, EPSPs, Theta values to compute PPRs for different inter-spike intervals ==
    input_spike_times_runs = pd.read_pickle(global_files + r'/input_st_runs.p')
    synapse_parameters = pd.read_pickle(global_files + 'distributions.p')

    ppr = np.flip(synapse_parameters['ppr'])            # flip, because PPR are ranked from weakest to strongest EPSP
    epsp_1 = np.flip(synapse_parameters['epsp_1'])      # flip, because EPSP are ranked from weakest to strongest
    np.min(ppr), np.max(ppr)
    n_inputs = np.shape(ppr)[0]

    plt.scatter(range(270),epsp_1, c = 'cornflowerblue')
    plt.scatter(range(270),ppr, c = 'green')
    plt.title('EPSP-PPR sorted')
    plt.savefig(experiment_path + 'EPSP_PPR_sorted.eps',format='eps')
    plt.show()


    # Header: find the correct scaling factor for each spike time and synapse ==========================================
    # find the theta index that corresponds best to each PPR:
    Theta_index = np.zeros((n_inputs,), dtype=int)
    for i in range(n_inputs):
        Theta_index[i] = (np.abs(Theta_parameters[4] - ppr[i])).argmin(axis = 0)    # find closes PPR value in Theta array

    # compute the maximum error of assigning a correct PPR, to be sure the method works:
    ppr_error = ppr - Theta_parameters[4][Theta_index]      # index 4 = PPR values
    print(np.max(ppr_error))


    # Header: build a conductance trace for each spike train ===========================================================
    # for each spike in each spike train, compute the scaling factor given the time after the previous spike:
    tau_rec = Theta_parameters[0]       # convert to smaller array
    tau_facil = Theta_parameters[1]     # convert to smaller array
    U = Theta_parameters[2]             # convert to smaller array
    f = Theta_parameters[3]             # convert to smaller array

    Scaling_factor_list = []                                    # initialize an empty list
    Syn_conductance_list = []                                   # initialize an empty list for synaptic conductances


    for run in range(np.shape(input_spike_times_runs)[0]):     # Iterate through the 100 runs
        current_run = input_spike_times_runs[run]
        scaling_list = []
        conductance_list = []

        for i in range(n_inputs):                                           # loop through the 270 inputs
            th_idx = Theta_index[i]                                         # get Theta index for the PPR for that input
            scaling_factors = np.ones(np.shape(current_run[i])[0])          # initialize array with 1 for the scaling factors
            syn_cond = np.ones(np.shape(current_run[i])[0])                 # initialize array with 1 for synaptic conductances
            R = 1                                                           # initialize R for the first run
            u = U[th_idx]                                                   # initialize u for the first run

            for spike in range(np.shape(current_run[i])[0]-1):                 # iterate through all the spikes of that input
                ISI = current_run[i][spike+1] - current_run[i][spike]          # compute the delay to the previous spike in [ms]

                # R and u need to over-written by the values from the last run
                R_n, u_n = Costa_Functions(R, u, tau_rec[th_idx], U[th_idx], f[th_idx], tau_facil[th_idx], ISI)
                scaling_factors[spike+1] = (R_n * u_n)/(R*u)                    # compute the relative PPR for that ISI
                syn_cond[spike+1] = syn_cond[spike] * scaling_factors[spike+1]  # derive total syn conductance for that spike

                R = R_n                                          # reset R for next run
                u = u_n                                          # reset u for next run

            scaling_list.append(scaling_factors)          # append the array for that input to the list
            conductance_list.append(syn_cond)

        Scaling_factor_list.append(scaling_list)
        Syn_conductance_list.append(conductance_list)
        print(run)


    # Header Save numpy array and excel data ===========================================================================
    excel_file = np.zeros([270,2])
    excel_file[:,0] = epsp_1
    excel_file[:,1] =ppr
    np.savetxt(experiment_path + 'EPSP_PPR_sorted.csv', excel_file,
               header="random EPSPs, random PPRs",
               delimiter=',', fmt='%s')
    np.save(experiment_path + 'Synaptic_conductance_scaling.npy',Syn_conductance_list)
