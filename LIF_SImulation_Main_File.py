# HEADER        This is the main script for our leaky integrate-and-fire model
# HEADER        Running segments of this script will produce the simulations in the paper
# HEADER        all relevant functions are called from within this script
    # TODO Make sure to change the directory names depending on your file paths
    # TODO Run the script in segments, depending on which simulation you would like to run





# HEADER ===============================================================================================================
# HEADER ===================== PART 1 Load scripts, prepare all files for simulation ===================================
# HEADER ===============================================================================================================


# Header: Import dependencies and all helper functions =================================================================
import os
import numpy as np
import json
import sys

    # TODO: enter the correct path:
sys.path.append('.../Buchholz_GastoneGuilabert_2023/helper_scripts')


# These functions generate the necessary data and execute the simulation, open each function script for more details
from Find_STP_conductance_scaling import Find_STP_conductance_scaling
from Find_synaptic_conductance_LIF_model import Find_synaptic_conductance_LIF_model
from Find_STP_conductance_scaling_SHUFFLED_EPSPs import Find_STP_conductance_scaling_SHUFFLED_EPSPs
from Create_Gsyn_timecourses_BASE_SIM import Create_Gsyn_timecourses_BASE_SIM
from Create_Gsyn_timecourses_SHUFFLED_EPSPs import Create_Gsyn_timecourses_SHUFFLED_EPSPs
from Create_Gsyn_timecourses_NO_STP import Create_Gsyn_timecourses_NO_STP
from Simulate_L23_LIF_model import Simulate_L23_model
from Analysis_Correlation import Analysis_Correlation
from Gain_Analysis import Gain_Analysis


# Header ===============================================================================================================
    # TODO: enter the correct path:
model_data_path = '.../Buchholz_GastoneGuilabert_2023/'


global_files = model_data_path + 'global_files/'    # contains files used by different simulations
experiment_path = model_data_path + 'simulation_1/' # Change this path to run and save a specific eperiment:

if not os.path.exists(experiment_path):     # Create folders for new experiments
    os.mkdir(experiment_path)
    os.mkdir(experiment_path + '0_base_simulation')     # base simulation
    os.mkdir(experiment_path + '1_only_strong')         # only strong inputs, as in Fig. 5 B
    os.mkdir(experiment_path + '2_only_weak')           # only weak inputs, as in Fig. 5 D
    os.mkdir(experiment_path + '3_shuffled_EPSPs')      # EPSPs are randomly assigned across spike trains, as in Fig. 6
    os.mkdir(experiment_path + '4_noSTP_all')           # no STP mechanism, all inputs are active
    os.mkdir(experiment_path + '5_noSTP_only_strong')   # no STP mechanism, only strong inputs, as in Fig. 7 B


# Header: Set the model Parmeters ======================================================================================
    # TODO: set the model parameters here. The current values reflect our base simulation
# To find adequate synapse parameter (t_peak and R_input and tau_m), run 'compare_model_syn_with_exp_syn.py'
# Allows you to play around with these parameters and compare to actual synaptic recordings

params = {
    "E_syn" : 0,            # synaptic reversal potential (mV)
    "R_input" : 100,        # input resistance (MOhm)
    "tau_m" : 20,           # membrane time constant (ms)
    "v_rest" : -70,         # resting membrane potential (mV)
    "t_peak" : 1,           # time constant of synaptic conductance (ms)
    "gmax_peak" : 10,       # synaptic conductance
    "sim_time" : 10000,     # duration of simulation (ms)
    "dt" : 0.1,             # time step of each simulation step (ms)
    "v_thres" : -50,        # action potential threshold (mV)
    "v_peak" : 25,          # action potential peak amplitude (mV)
    "t_refr" : 10,          # refractory period (ms)
    "spike_time" : 20}      #

# Save the parameters used for each experiment as a text file in the respective directory
with open(experiment_path + 'params.txt', 'a') as file:
    json.dump(params, file,indent = 2)


# Header: Set theta parameters by either running script 0 or loading a pre-existing parameter sets =====================
# If you run this for the first time, then run script 0_Find_STP_parameter_set_theta, which saves 'Theta_parameters' in 'global_files'
# Otherwise, use a pre-compiled set of 'Theta_parameters' already saved in 'global_files':

Theta_parameters = np.load(global_files + 'Theta_parameters.npy')


# Header: Compute scaling of input synapses during spike train =========================================================
# Compute how each spike input needs to be scaled depending on PPR and the selected 'Theta_parameters':
# Saves 'Synaptic_conductance_scaling.npy' in 'simulation_n' and has to be re-run whenever Theta_parameters is changed
input_plotting = [0,50,100,150,200,250]

Find_STP_conductance_scaling(global_files,experiment_path,Theta_parameters,input_plotting)
Find_STP_conductance_scaling_SHUFFLED_EPSPs(global_files,experiment_path,Theta_parameters,input_plotting)


# Header: Find the linear transfer function between EPSP and Gsyn ======================================================
# Simulate alpha synapses to find the transfer function between g_max and EPSP amplitude by fitting a line between them

    # TODO: you will have to change a file directory in this script:
m_fit,c_fit = Find_synaptic_conductance_LIF_model(params,experiment_path)


# Header: Create the conductance time courses that go into the model ===================================================
# Create the conductance vector that is fed into the model neuron by summing up the conductances of all individual inputs
# Run separately for each experiment, output is saved in respective experiment folder as 'Conductance_trace_SUMMED.npy'

    # TODO: Set window in which the exponential decay of syn conductance is intergrated:
int_window = 5    # for fast, explorative sim
# int_window = 20     # for slow, high-accuracy sim

Create_Gsyn_timecourses_BASE_SIM(params,global_files,experiment_path,m_fit,c_fit,int_window,input_plotting)

# Specify whether a shuffled EPSP file should be loaded that is available to all experiments in the global folder (which has to be generated first),
# or an EPSP file that is re-shuffled for each experiment:
load_global_EPSPs = True
Create_Gsyn_timecourses_SHUFFLED_EPSPs(params,global_files,experiment_path,m_fit,c_fit,load_global_EPSPs,int_window,input_plotting)
Create_Gsyn_timecourses_NO_STP(params,global_files,experiment_path,m_fit,c_fit,int_window,input_plotting)



# HEADER ===============================================================================================================
# HEADER ================================== PART 2 - Run the simulations ===============================================
# HEADER ===============================================================================================================
# TODO: 1) run the line 'experiment = ...' that corresponds to the respective experimental setup
# TODO: 2) execute the remaining steps to run the simulation and analyze the output data
# TODO: 3) repeat for a different experiment, if desired


# Header: Select which experiment should be simulated =====================================================================
    # TODO - Run '2_only_weak' first, to find by how much Vm should be depolarized in the next experiment
experiment = '2_only_weak'; inputs_range = range(35,270); Vm = params['v_rest']

# baseline simulation --------------------------------------------------------------
# experiment = '0_base_simulation'; inputs_range = range(270); Vm = params['v_rest']
# experiment = '1_only_strong'; inputs_range = range(35); Vm = -63                    # TODO: Manually enter V_rest of only the background inputs

# Shuffled EPSPs -------------------------------------------------------------------
# experiment = '3_shuffled_EPSPs'; inputs_range = range(270); Vm = params['v_rest']

# No short-term plasticity ---------------------------------------------------------
# experiment = '4_noSTP_all'; inputs_range = range(270); Vm = params['v_rest']
# experiment = '5_noSTP_only_strong'; inputs_range = range(35); Vm = params['v_rest']



# Header: Set which run and time window should be plotted in the next two scripts ======================================
plot_run = 7; plot_window = [40000,80000]


# Header: Run the LIF Model ============================================================================================
# Run the actual leaky integrate-and fire model simulation
# Firing rates of model cell for the 100 runs and one example trace will be saved
    # TODO: you will have to change a file directory in this script:
Vm_mean, Vm_median = Simulate_L23_model(Vm,params,experiment_path,experiment,plot_run,plot_window)


# Header: Run the Analysis file to do the correlation analysis =========================================================
# Run the correlation analysis on the spike trains of the inputs and the output of the model cell
# Save Pearson Correlation plot and raw data as excel file in the directory of the experiment
Analysis_Correlation(experiment_path,global_files,experiment,inputs_range,plot_run,plot_window,50,0.5)


# Header: Run the Gain Analysis file for each experiment ================================================================
Gain_Analysis(experiment_path,global_files,experiment,params,inputs_range,50,50)

