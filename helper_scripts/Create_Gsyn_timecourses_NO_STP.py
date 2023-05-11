# Header Convert the spike times and the synaptic conductance scalings into the final Gsyn traces for the model
# Header Save all these traces as a npy array for use in the simulation

from matplotlib.pyplot import cm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def Create_Gsyn_timecourses_NO_STP(params,global_files,experiment_path,m_fit,c_fit,int_window,input_plotting):
    # Header: Call the necessary variables
    sim_time = params['sim_time']
    dt = params['dt']
    t_peak = params['t_peak']


    # Header: import data: spike times, PPRs, EPSPs, Theta values to compute PPRs for different inter-spike intervals ==
    input_spike_times_runs = pd.read_pickle(global_files + r'/input_st_runs.p')
    synapse_parameters = pd.read_pickle(global_files + 'distributions.p')

    epsp_1 = np.flip(synapse_parameters['epsp_1'])          # Flip the EPSPs, so the strongest EPSPs align with the most correlated spikes


    # Header: Compute the overall conductance trace for all synaptic inputs onto the cell ==============================
    gmax = m_fit * epsp_1 + c_fit  # convert the EPSP of the connection into a conductance
    Conductance_array = []                                                                                      # Array that saves all conductance traces for all 100 runs

    for run in range(100):                                                                                      # iterate through the 100 runs
        G_input = np.zeros([270,100000])                                                                        # initialize an empty conductance array

        for inputs in range(270):                                                                               # loop through all the inputs
            g = np.zeros(100000)                                                                                # initialize an empty conductance array for each input
            for spike_t in input_spike_times_runs[run][inputs]:                                                 # iterate through all the spike times
                for t in np.arange(spike_t,sim_time,dt):                                                        # simulate only the time after spike for efficiency
                    T_idx = int(t/dt)                                                                           # convert time to the respective index
                    # DON'T SCALE THE EPSP BY A PPR VALUE:
                    g[T_idx] = gmax[inputs] * (t - spike_t) / t_peak * np.exp(- (t - spike_t - t_peak) / t_peak)

                    if t - spike_t >= int_window:                                                               # terminate g computation at 20ms, where g == 0 to save time
                        break

            G_input[inputs][:] = g                                                                              # write the g values for all inputs into a 2D array
        print(run)
        Conductance_array.append(G_input)                                                                       # append these arrays into a list object
    Conductance_array = np.array(Conductance_array)


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
    plt.savefig(experiment_path + 'Gsyn_inputs_noSTP_examples.eps',format='eps')
    plt.show()


    # Header: Save the raw conductance tracs of all experiments ========================================================
    np.save(experiment_path+'4_noSTP_all/Conductance_trace_SUMMED.npy',np.sum(Conductance_array,axis = 1))
    np.save(experiment_path+'5_noSTP_only_strong/Conductance_trace_SUMMED.npy',np.sum(Conductance_array[:, range(35), :], axis=1))