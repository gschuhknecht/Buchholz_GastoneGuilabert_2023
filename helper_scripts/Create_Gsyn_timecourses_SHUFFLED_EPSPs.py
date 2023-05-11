# Header Convert the spike times and the synaptic conductance scalings into the final Gsyn traces for the model
# Header Save all these traces as a npy array for use in the simulation

from matplotlib.pyplot import cm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def Create_Gsyn_timecourses_SHUFFLED_EPSPs(params,global_files,experiment_path,m_fit,c_fit,load_global_EPSPs,int_window,input_plotting):

    # Header: Call the necessary variables
    sim_time = params['sim_time']
    dt = params['dt']
    t_peak = params['t_peak']


    # Header: import data: spike times, PPRs, EPSPs, Theta values to compute PPRs for different inter-spike intervals ==
    input_spike_times_runs = pd.read_pickle(global_files + r'/input_st_runs.p')


    # Header: Load the permuted EPSP data and synaptic scaling:
    Synaptic_conductance_scaling = np.load(experiment_path + 'Synaptic_conductance_scaling_SHUFFLED_EPSP.npy', allow_pickle=True)


    # Specify whether a shuffled EPSP file should be loaded that is available to all experiments in the global folder,
    # or an EPSP file that is re-shuffled for each experiment:
    if load_global_EPSPs == True:
        epsp_1 = np.load(global_files + 'SHUFFLED_EPSPs.npy') # load the permuted EPSPs
        Synaptic_conductance_scaling = np.load(global_files + 'Synaptic_conductance_scaling_SHUFFLED_EPSP.npy', allow_pickle=True)
        print('globally stored shuffled EPSPs were loaded')
    else:
        epsp_1 = np.load(experiment_path + 'SHUFFLED_EPSPs.npy')
        Synaptic_conductance_scaling = np.load(experiment_path + 'Synaptic_conductance_scaling_SHUFFLED_EPSP.npy', allow_pickle=True)
        print('newly generated shuffled EPSPs were loaded')



    # Header: Compute the overall conductance trace for all synaptic inputs onto the cell ==============================
    Conductance_array = []                                                                                      # Array that saves all conductance traces for all 100 runs
    for run in range(100):                                                                                      # iterate through the 100 runs
        G_input = np.zeros([270,100000])                                                                        # initialize an empty conductance array
        gmax = m_fit * epsp_1 + c_fit                                                                           # convert the EPSP of the connection into a conductance
        start = time.time()                                                                                     # measure loop time
        g_at_terminate = []
        for inputs in range(270):                                                                               # loop through all the inputs
            g = np.zeros(100000)                                                                                # initialize an empty conductance array for each input
            iter = 0
            for spike_t in input_spike_times_runs[run][inputs]:                                                 # iterate through all the spike times

                for t in np.arange(spike_t,sim_time,dt):                                                        # simulate only the time after spike for efficiency
                    T_idx = int(t/dt)                                                                           # convert time to the respective index
                    g_STP = gmax[inputs] * Synaptic_conductance_scaling[run][inputs][iter]                      # scale gmax by STP scaling factor
                    g[T_idx] = g_STP * (t - spike_t) / t_peak * np.exp(- (t - spike_t - t_peak) / t_peak)

                    if t - spike_t >= int_window:                                                               # terminate g computation at 20ms, where g == 0 to save time
                        # g_at_terminate.append(g[T_idx])                                                       # save current g value to compute the loss of data
                        break
                iter += 1

            G_input[inputs][:] = g                                                                              # write the g values for all inputs into a 2D array
        print(run)
        Conductance_array.append(G_input)                                                                       # append these arrays into a list object
    Conductance_array = np.array(Conductance_array)
    end = time.time()
    print(end - start)                                                # report the duration of the entire loop


    # CONTROL: Plot some EPSP traces to verify the synaptic scaling looks good -----------------------------------------
    plt.figure(figsize = (12,8))
    color = cm.rainbow(np.linspace(0, 1, np.shape(input_plotting)[0]))
    iter = 0
    for i in input_plotting:
        plt.plot(Conductance_array[0][i] - iter, alpha = 0.7, linewidth = 3, c=color[iter])
        iter += 1
    plt.ylabel('Synaptic conductance')
    plt.xlabel('simulated samples at 0.1 Hz')
    plt.legend(input_plotting)
    plt.savefig(experiment_path + 'Gsyn_inputs_ShUFFLED_EPSPs_examples.eps',format='eps')
    plt.show()


    # Header: Save the raw conductance traces of all experiments =======================================================
    np.save(experiment_path + '3_shuffled_EPSPs/Conductance_trace_SUMMED.npy',np.sum(Conductance_array,axis = 1))
