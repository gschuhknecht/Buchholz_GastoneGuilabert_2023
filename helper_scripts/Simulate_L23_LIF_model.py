# HEADER Run the actual simulation with selected runs and for all 100 runs
# HEADER Compute overall spike rate per run
# HEADER Save the spike times for each run into a numpy file

# Header: Import dependencies ==========================================================================================
import numpy as np
import sys
import matplotlib.pyplot as plt

    # TODO: enter the correct path:
sys.path.append('.../Buchholz_GastoneGuilabert_2023/helper_scripts')
from modules_LIF_simulation import LIF_model


def Simulate_L23_model(Vm,params,experiment_path,experiment,plot_run,plot_window):

    path = experiment_path
    G = np.load(path+experiment+'/Conductance_trace_SUMMED.npy')


    # Header: Set model parameter ======================================================================================
    sim_time = params['sim_time']
    dt = params['dt']
    E_syn = params['E_syn']
    R_input = params['R_input']
    tau_m = params['tau_m']
    v_rest = Vm
    v_thres = params['v_thres']
    v_peak = params['v_peak']
    t_refr = params['t_refr']


    # Header: Run the model through all 100 runs =======================================================================
    firing_rate = np.zeros(100)
    Vm_cleaned_mean = np.zeros(100)
    Vm_cleaned_median = np.zeros(100)
    Vm_cleaned_sd = np.zeros(100)
    Vm_raw_runs_mean = np.zeros(100)
    Vm_raw_runs_sd = np.zeros(100)

    Vm_all_runs = np.zeros([100,100000])            # save all v values

    Spike_Times_Output = []
    for run in range(100):
        v = LIF_model(v_rest, v_thres, v_peak, R_input, tau_m, G[run], E_syn, sim_time, dt, t_refr)     # run model
        firing_rate[run] = np.shape(np.where(v == v_peak))[1] / 10                                      # compute firing rate
        print(run)
        spikes = np.where(v == v_peak)[0]
        Spike_Times_Output.append(spikes)

        Vm_all_runs[run] = v                        # save all v values

        Vm_cleaned_mean[run] = np.nanmean(v[(v <= v_thres) & (v > v_rest)])               # only include values below v_thres and above v_rest
        Vm_cleaned_median[run] = np.nanmedian(v[(v <= v_thres) & (v > v_rest)])
        Vm_cleaned_sd[run] = np.nanstd(v[(v <= v_thres) & (v > v_rest)])
        Vm_raw_runs_mean[run] = np.nanmean(v)
        Vm_raw_runs_sd[run] = np.nanstd(v)


    # Header: Plot histogams of subthreshold Vm ========================================================================
    Vm_all_cleaned = Vm_all_runs[ (Vm_all_runs < v_thres) & (Vm_all_runs > v_rest)]
    binedges = np.linspace(-72,-48,25)
    plt.hist(Vm_all_cleaned,binedges,density = True,histtype='step',color = 'black')
    plt.ylim([0,0.2])
    plt.savefig(experiment_path + experiment +'/Vm_histogram.eps',format='eps')
    plt.show()


    # Header: Convert to excel file for simplicity ====================================================================
    excel_file = np.zeros([100, 5])
    excel_file[:, 0] = firing_rate
    excel_file[:, 1] = Vm_cleaned_mean
    excel_file[:, 2] = Vm_cleaned_sd
    excel_file[:, 3] = Vm_raw_runs_mean
    excel_file[:, 4] = Vm_raw_runs_sd


    # Header:: Plot Vm during the input spike window shown in the paper ================================================
    v = LIF_model(v_rest, v_thres, v_peak, R_input, tau_m, G[plot_run], E_syn, sim_time, dt, t_refr)
    plt.figure(figsize = (20,10))
    plt.plot([0,100000],[v_thres,v_thres],c = 'grey')
    plt.plot(v, linewidth = 2, c = 'black')
    plt.xlim(plot_window)
    plt.ylim([v_rest - 5, v_peak + 5])
    plt.title('LIF Model output spikes')
    plt.savefig(experiment_path + experiment +'/LIF_model_Vm_for_figure.eps',format='eps')
    plt.show()


    # Header:: Plot Vm during the entire simulation time ===============================================================
    v = LIF_model(v_rest, v_thres, v_peak, R_input, tau_m, G[plot_run], E_syn, sim_time, dt, t_refr)
    plt.figure(figsize = (20,10))
    plt.plot([0,100000],[v_thres,v_thres],c = 'grey')
    plt.plot(v, linewidth = 1, c = 'black')
    plt.title('LIF Model output spikes')
    plt.ylim([v_rest-5,v_peak+5])
    plt.savefig(experiment_path + experiment +'/LIF_model_Vm_all.eps',format='eps')
    plt.show()


    # Header: Print and plot spike rate ================================================================================
    print(np.mean(firing_rate))
    np.std(firing_rate)

    plt.figure(figsize= (3,5))
    plt.scatter((np.random.randn(100)*0.2)+1,firing_rate, c = 'black', alpha = 0.5)
    plt.scatter(1,np.mean(firing_rate),c = 'red')
    plt.xlim([0,2]), plt.ylim([0,np.max(firing_rate+2)])
    plt.title('Spike rate')
    plt.savefig(experiment_path + experiment +'/Spike_rate.eps',format='eps')
    plt.show()


    # Header: Save the spike times of the model, the firing rate, and statistics on Vm without spikes ==================
    np.save(path + experiment + '/Spike_Times_Output.npy',Spike_Times_Output)
    np.savetxt(path + experiment + '/firing_rates.csv', excel_file,
               header = "firing rate, Vm_cleaned_mean, Vm_cleaned_sd, Vm_raw_mean, Vm_raw_sd", delimiter=',', fmt='%s')


    return (np.nanmean(Vm_cleaned_mean),np.nanmean(Vm_cleaned_median))






