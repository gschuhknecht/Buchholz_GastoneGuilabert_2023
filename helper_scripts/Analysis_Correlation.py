# HEADER Compute Pearson Correlation Coefficients for spike trains convolved wih exponential decay


# Header: Import dependencies ==========================================================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def Analysis_Correlation(experiment_path,global_files,experiment,inputs_range,plot_run,plot_window,scatter_size,upper_lim):

    # Header: Set the path where data should be saved and which experiment should be analyzed ==========================
    # When running for the first time, the convolved input spikes needs to be computed first (see below):
    Input_Convolved_Spikes = np.load(global_files + 'Input_Convolved_Spikes.npy')


    # Header: Load spike times of inputs and output ====================================================================
    inputs = np.array(pd.read_pickle(global_files + r'/input_st_runs.p'))
    Spike_Times_Inputs = inputs * 10                    # the input spike train is saved in ms, convert ms into time points
    Spike_Times_Output = np.load(experiment_path + experiment +'/Spike_Times_Output.npy', allow_pickle= True)


    # Header: Convolve spike times with exponential decay ==============================================================
    tau = 10                        # 10 ms decay time constant
    tau_vR = int(tau / 0.1)         # convert to samples


    # Header: (1) Convolve OUTPUT times of spike trains for all 100 runs ===============================================
    Output_Convolved_Spikes = np.zeros([100, 100000])           # initialize

    for run in range(100):                                      # iterate through 100 runs
        convolved_spikes = np.zeros(100000)                               # initialize output array
        for i in Spike_Times_Output[run]:                       # go through spikes
            t = 0                                               # initialize time stamp

            for iter in np.arange(i, 100000 - 1):
                convolved_spikes[iter] += np.exp(-t / tau_vR)   # convolve spike train with a 10 ms exponential decay and add to previous value
                t += 1
                if t > 1000:                                    # only compute for 100 ms after spike
                    break
        print(run)
        Output_Convolved_Spikes[run] = convolved_spikes  # save the convolved spikes


    # CONTROL: Plot a small example trace of original spike times and their convolved exponential decays -------------------
    r = 0                                                      # which run to plot
    plt.figure(figsize= (16,8))
    plt.plot(Output_Convolved_Spikes[r],c='grey',linewidth = 2)
    for q in range(np.shape(Spike_Times_Output[r])[0]):
        plt.plot([int(Spike_Times_Output[r][q]),int(Spike_Times_Output[r][q])],[-0.05,-0.15],c='red',linewidth = 3)
    plt.title('LIF Model output spikes')
    plt.xlim([0,20000]), plt.show()


    # CONTROL: Plot a small example trace of original spike times and their convolved exponential decays -------------------
    r = plot_run                                                            # which run to plot
    c = 0                                                            # which cell to plot
    plt.figure(figsize=(16, 8))
    plt.plot(Input_Convolved_Spikes[r][c], c='grey', linewidth=2)
    for q in range(np.shape(Spike_Times_Inputs[r][c])[0]):
        plt.plot([int(Spike_Times_Inputs[r][c][q]), int(Spike_Times_Inputs[r][c][q])], [-0.05, -0.15], c='lightgreen',
                 linewidth=3)
    plt.title('Spike train of input 0')
    plt.xlim([0, 20000]), plt.show()


    # Header: Compute Pearson Correlation between input and output for each run ========================================
    n_runs = 100                                                    # how many runs should be computed?

    Correlation_all = np.zeros([n_runs,270])
    for run in range(n_runs):
        for cell in range(270):
            a = np.corrcoef(Input_Convolved_Spikes[run][cell],Output_Convolved_Spikes[run])
            Correlation_all[run,cell] = a[0,1]
    Correlation_mean = np.nanmean(Correlation_all,axis = 0)
    Correlation_median = np.nanmedian(Correlation_all,axis = 0)


    # Header: Compute Confidence intervals around the mean values ======================================================
    Confidene_intervals = np.zeros([2,270])
    Confidene_intervals[0] = np.nanpercentile(Correlation_all, 25, axis=0)
    Confidene_intervals[1] = np.nanpercentile(Correlation_all, 75, axis=0)


    # Header Save figures and data in excel for analysis in other software =============================================
    plt.figure(figsize=(12,5))
    plt.scatter(inputs_range,Correlation_median[inputs_range],c='black',s = scatter_size)
    plt.plot(inputs_range,Confidene_intervals[0][inputs_range], c ='grey'),plt.plot(inputs_range,Confidene_intervals[1][inputs_range], c ='grey')
    plt.plot([0,270],[0,0], c='black', linestyle = 'dashed'), plt.plot([34,34],[-0.05,.45], c='black', linestyle = 'dashed')
    plt.xlabel('input index'), plt.ylabel('correlation')
    plt.title(experiment + ' median + 25-75%')
    plt.ylim([-0.05,upper_lim])
    plt.xlim([0,270])
    plt.savefig(experiment_path + experiment +'/input-output_correlation_median.eps',format='eps')
    plt.show()

    sd = np.nanstd(Correlation_all,axis = 0)
    plt.figure(figsize=(12, 5))
    plt.scatter(inputs_range,Correlation_mean[inputs_range],c='black', s = scatter_size)
    plt.plot(inputs_range, Correlation_mean[inputs_range] + sd[inputs_range], c='grey')
    plt.plot(inputs_range, Correlation_mean[inputs_range] - sd[inputs_range], c='grey')
    plt.plot([0, 270], [0, 0], c='black', linestyle='dashed'), plt.plot([34, 34], [-0.05, .45], c='black',linestyle='dashed')
    plt.xlabel('input index'), plt.ylabel('correlation')
    plt.title(experiment + ' mean + sd')
    plt.ylim([-0.05, upper_lim])
    plt.xlim([0, 270])
    plt.savefig(experiment_path + experiment + '/input-output_correlation_mean.eps', format='eps')
    plt.show()

    excel_file = np.zeros([len(inputs_range),4])
    excel_file[:,0] = Correlation_mean[inputs_range]
    excel_file[:,1] = Correlation_median[inputs_range]
    excel_file[:,2] = Confidene_intervals[0][inputs_range]
    excel_file[:,3] = Confidene_intervals[1][inputs_range]
    np.savetxt(experiment_path + experiment +'/input-output_correlation.csv', excel_file, header = "mean, median, CIl_ow, CI_high ",
               delimiter=',', fmt='%s')


    # Header Save a rasterplot =========================================================================================
    # if experiment == '0_base_simulation':
    plt.eventplot(Spike_Times_Output[plot_run], lineoffsets=12, linelengths=30, color='red')
    plt.eventplot(np.flip(Spike_Times_Inputs[plot_run][0:27]), linelengths=1)
    plt.title(experiment)
    plt.xlim(plot_window)
    plt.ylim([-1, 28])
    plt.savefig(experiment_path + experiment +'/spike_raster_plot.eps',format='eps')
    plt.show()

    plt.figure(figsize = [12,20])
    plt.eventplot(Spike_Times_Output[plot_run], lineoffsets=50, linelengths=100, color='red')
    plt.eventplot(np.flip(Spike_Times_Inputs[plot_run][0:100]), linelengths=1)
    plt.title(experiment)
    plt.xlim(plot_window)
    plt.ylim([-1, 101])
    plt.savefig(experiment_path + experiment +'/spike_raster_plot_100inputs.eps',format='eps')
    plt.show()

    plt.figure(figsize = [12,20])
    plt.eventplot(Spike_Times_Output[plot_run], lineoffsets=50, linelengths=100, color='red')
    plt.eventplot(np.flip(Spike_Times_Inputs[plot_run][150:249]), linelengths=1)
    plt.title(experiment)
    plt.xlim(plot_window)
    plt.ylim([-1, 101])
    plt.savefig(experiment_path + experiment +'/spike_raster_plot_last_100inputs.eps',format='eps')
    plt.show()