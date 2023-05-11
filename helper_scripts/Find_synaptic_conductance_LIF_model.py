
def Find_synaptic_conductance_LIF_model(params,experiment_path):

    # Header: Import dependencies ======================================================================================
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm
    import sys

    # TODO: enter the correct path where the helper scripts are stored:
    sys.path.append('.../Buchholz_GastoneGuilabert_2023/helper_scripts')
    from modules_LIF_simulation import LIF_model


    # Header: Load params from the dict ================================================================================
    sim_time = 100
    spike_time = 20

    dt = params['dt']
    gmax = np.linspace(0,params['gmax_peak'],20)    # simulate 20 alpha synapses
    t_peak = params['t_peak']
    v_rest = params['v_rest']
    v_thres = params['v_thres']
    v_peak = params['v_peak']
    R_input = params['R_input']
    tau_m = params['tau_m']
    E_syn  = params['E_syn']
    t_refr = params['t_refr']


    # Header: Simulate a single alpha synapse for a range of different conductances ====================================
    G = np.zeros(np.int(sim_time/dt))                               # initialize an empty conductance array
    EPSP_amplitudes = np.zeros(np.shape(gmax)[0])                   # initialize and empty EPSP array

    fig, ax = plt.subplots(2, 1,figsize = (16,16))
    color = cm.rainbow(np.linspace(0, 1, np.shape(gmax)[0]))
    for run in range(np.shape(gmax)[0]):
        for T in range(np.shape(G)[0]):
            t = T * dt               # convert time steps back into ms
            if t <= spike_time:
                G[T] = 0
            else:
                G[T] = gmax[run] * (t - spike_time)/t_peak * np.exp(- (t- spike_time - t_peak)/t_peak)

        v = LIF_model(v_rest,v_thres,v_peak,R_input,tau_m,G,E_syn,sim_time,dt,t_refr)
        EPSP_amplitudes[run] = np.max(v) - v_rest
        ax[0].plot(v, c=color[run])
        ax[1].plot(G, c=color[run])

        ax[0].set_ylabel('EPSP amplitude', fontsize = 20)
        ax[1].set_ylabel('peak synaptic conductance', fontsize = 20)

    plt.savefig(experiment_path + 'Plot_Gsyn_and_EPSP.eps',format='eps')
    plt.show()

    # Header: fit a line to the scatter plot of EPSP - synaptic conductance ============================================
    m,c = np.polyfit(EPSP_amplitudes,gmax,1)

    plt.scatter(EPSP_amplitudes,gmax,c=color)
    plt.plot(EPSP_amplitudes, m * EPSP_amplitudes + c)
    plt.xlabel('EPSP amplitude')
    plt.ylabel('peak synaptic conductance')
    plt.savefig(experiment_path + 'Correlation_Gsyn_EPSP.eps', format='eps')
    plt.show()

    print('best-fit m is:'),print(m)
    print('best-fit c is:'),print(c)

    return(m,c)