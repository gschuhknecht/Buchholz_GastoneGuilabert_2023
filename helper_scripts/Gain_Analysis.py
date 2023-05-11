# HEADER: this script runs the gain analyzes shwon in figures 5-7


# Header: Import dependencies =============================================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def Gain_Analysis(experiment_path,global_files,experiment,params,inputs_range,n_spikes_tested,scatter_size):
    t_refr = params['t_refr']

    # Header: Get data ========================================================================================================
    input_spike_times_runs = pd.read_pickle(global_files + r'/input_st_runs.p')
    Spikes_Input = np.multiply(input_spike_times_runs, 10)                  # convert ms to samples
    Spikes_Output = np.load(experiment_path + experiment +'/Spike_Times_Output.npy', allow_pickle= True)            # samples


    # Header: Collect all spikes fired from all 270 inputs and sort them according to spike time ==============================
    Sorted_input_spike_list = []
    for run in range(100):
        A = []
        for inputs in inputs_range:
            A = np.append(A,Spikes_Input[run][inputs])
        Sorted_input_spike_list.append(np.sort(A))


    # Header: Compute whether coincident spikes in the inputs resulted in a spike in the output ===============================
    # IDEA  1) Iterate through every single spike time in the inputs
    # IDEA  2) Include all input spikes that happen in the next 20 ms after that initial input spike
    # IDEA  3) Check if the output cell also spiked in this 20 ms window
    # IDEA  4) If output spiked, start this iteration process AFTER the output spike
    # IDEA  5) If output did not spike, continue the iteration with the next spike in the input spike train

    # Initialize the final lists that save the number of input spikes
    N_Coincident_Inputs_SPIKE = []
    N_Coincident_Inputs_noSPIKE = []

    for run in range(100):                          # iterate through all runs
        List = []                                   # helper list
        spike_fired_List = []                       # list for that run, saving the number of inputs leading to spike
        no_spike_fired_List = []                    # list for that run, saving the number of inputs leading to no spike
        last_output_spike_time = 0                  # running variable that saves the last time the output spiked
        last_t = []                                 # running variable that saves the time stamp of input_spike from the last iteration

        # 1) iterate through all input spike times:
        for input_spike in Sorted_input_spike_list[run]:                    # input_spike corresponds to the time stampt of the current input spike

            # 4) if an ouput spike has been fired in last iteration, continue only after this output spike time and its refractory period is over:
            if input_spike < last_output_spike_time + (t_refr * 10):        # if more inputs happened before the output spike ...
                continue                                                    # ... continue to skip over them and move to next input spike

            # 5) If output did not spike, count the number of spikes that happen in the next 20ms window
            else:

                # If several input spikes happened at the same time stamp, move to the next time stamp in the for loop
                if input_spike == last_t:        # if input_spike of current and previous iteration happened at same time
                    continue                     # exit from current loop iteration and go to the next time stamp

                # 5) register all spikes that happen in the 20ms window after the current input_spike:
                else:
                    spikes_after = np.where((Sorted_input_spike_list[run] >= input_spike) & (Sorted_input_spike_list[run] < input_spike + 200.0))
                    List.append(spikes_after)
                    last_t = input_spike                                # update the last_t variable

                    # check if a spike was fired in this 20ms interval:
                    spike_fired = np.any((Spikes_Output[run] >= input_spike) & (Spikes_Output[run] < input_spike + 200))
                    if spike_fired == False:                                        # if no spike was fired ...
                        no_spike_fired_List.append(np.shape(spikes_after)[1])       # ... append the number of spikes in the 20ms window to the no_spike list

                    else:                                                           # if no spike was fired ...
                        spike_fired_List.append(np.shape(spikes_after)[1])          # ... append the number of spikes in the 20ms window to the spike list
                        idx = np.where((Spikes_Output[run] >= input_spike) & (Spikes_Output[run] < input_spike + 200))  # get idx of the output spike time ...
                        last_output_spike_time = int(Spikes_Output[run][idx][0])    # ... and update the last-output-spike variable

        print(run)

        N_Coincident_Inputs_SPIKE.append(np.array(spike_fired_List))            # after each run, append the number of spikes to the respective list
        N_Coincident_Inputs_noSPIKE.append( np.array(no_spike_fired_List))      # after each run, append the number of spikes to the respective list


    # Header: Compute spike probability given the input spikes ===============================================================
    spike_probability = np.zeros([100,n_spikes_tested])         # initialize array to save the spike probability given n inputs for each run

    for run in range(100):
        coincident_spikes = np.zeros(n_spikes_tested)
        coincident_failures = np.zeros(n_spikes_tested)
        spike_fired_List = np.array(n_spikes_tested)

        # iterate through the number of coincident spikes for which the spike probability should be tested:
        for i in range(n_spikes_tested):
            coincident_spikes[i] = np.shape(np.where(N_Coincident_Inputs_SPIKE[run] == i))[1]       # How many times was a spike elicited given these coincident inputs?
            coincident_failures[i] = np.shape(np.where(N_Coincident_Inputs_noSPIKE[run] == i))[1]   # How many times was NO spike elicited given these coincident inputs?

        spike_probability[run,:] = coincident_spikes/(coincident_failures + coincident_spikes)      # compute P(spike) for each number of coincident inputs


    # Header: Plotting =======================================================================================================
    x_values = np.linspace(1,n_spikes_tested,n_spikes_tested)

    # Plot median and percentiles
    plt.plot(x_values,np.nanmedian(spike_probability,axis = 0), c = 'black')
    plt.scatter(x_values,np.nanmedian(spike_probability,axis = 0), c = 'black', s = scatter_size)
    plt.plot(x_values, np.nanpercentile(spike_probability, 75,axis = 0),c = 'lightgrey')
    plt.plot(x_values, np.nanpercentile(spike_probability, 25,axis = 0),c = 'lightgrey')
    plt.ylim([0,1])
    plt.xlim([0,35])
    plt.title('median and 25-75 percentile')
    plt.legend([experiment,'75th percentile','25th percentile'])
    plt.savefig(experiment_path + experiment + '/gain_analysis_median.eps', format='eps')
    plt.show()

    # Plot mean and S.D.
    plt.plot(x_values,np.nanmean(spike_probability,axis = 0), c = 'black')
    plt.scatter(x_values,np.nanmean(spike_probability,axis = 0), c = 'black', s = scatter_size)
    plt.plot(x_values,np.nanmean(spike_probability,axis = 0) + np.nanstd(spike_probability,axis = 0), c = 'lightgrey')
    plt.plot(x_values,np.nanmean(spike_probability,axis = 0) - np.nanstd(spike_probability,axis = 0), c = 'lightgrey')
    plt.ylim([0,1])
    plt.xlim([0,35])
    plt.title('mean and standard deviation')
    plt.legend([experiment,'+ sd','- sd'])
    plt.savefig(experiment_path + experiment + '/gain_analysis_mean.eps', format='eps')
    plt.show()