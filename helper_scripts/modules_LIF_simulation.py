# HEADER: Defines all functions for the modeling

import numpy as np

# Header: LIF Model ====================================================================================================
def LIF_model(v_rest,v_thres,v_peak,R_input,tau_m,G,Esyn,sim_time,dt,t_refr):
    g_leak = 1 / (R_input * 0.001)                  # Leak conductance in 1/GOhm
    n = np.int(np.round(sim_time / dt))                    # total number of simulated time steps
    v = np.ones(n) * v_rest                         # initialize voltage vector
    refractory = 0                                  # variable that saves duration of refractory period

    for i in range(n - 1):                          # start at 1 and always computed V(t+1)
        if refractory > 0:
            v[i+1] = v_rest
            refractory -= 1
        else:
            I_syn = - G[i] * (v[i] - Esyn)            # syn current depends on reversal potential
            dv = (-(v[i] - v_rest) + I_syn / g_leak) * (dt / tau_m)
            v[i + 1] = v[i] + dv

            if v[i + 1] >= v_thres:
                v[i] = v_peak
                v[i + 1] = v_rest
                refractory = t_refr / dt                # the number of time steps in refracrory period
    return np.array(v)

# Header: PPR Function - Equation (8) ==================================================================================
def PPR_function(tau_rec,tau_facil,u,f,delta_tn):
    equ1 = 1 - (u * np.exp(-delta_tn / tau_rec))
    equ2 = u + (f * (1-u) * np.exp(-delta_tn / tau_facil))
    PPR = (equ1*equ2)/u
    return(PPR)

# Header: Costa's Functions to compute PPRs during arbitrary spike train - Equations (3) & (4) =========================
def Costa_Functions(R_n, u_n, tau_rec, U, f, tau_facil, ISI):
    R = 1 - (1 - R_n * (1 - u_n)) * np.exp(- ISI/tau_rec)
    u = U + (u_n + f*(1-u_n) - U) * np.exp(- ISI/tau_facil)
    return(R,u)













