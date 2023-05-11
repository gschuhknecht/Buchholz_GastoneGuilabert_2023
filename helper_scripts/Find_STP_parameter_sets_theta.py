# HEADER Find_STP_parameter_sets_theta
# HEADER Find the parameters sets Theta (tau_rec, tau_facil, U, f) to model short-term plasticity

# Header: Import dependencies ==========================================================================================
import numpy as np
import matplotlib.pyplot as plt
import sys


# TODO: enter the correct path where the helper scripts are stored:
sys.path.append('.../Buchholz_GastoneGuilabert_2023/helper_scripts')
from modules_LIF_simulation import PPR_function

# Header: Set the range of Theta parameter values ======================================================================
delta_tn = 20
data_points = 1000

tau_rec = np.linspace(1700,20,data_points)
tau_facil = np.linspace(20,1700,data_points)
u = np.linspace(0.7,0.1,data_points)
f = np.linspace(0.05,0.11,data_points)

PPR_values = np.zeros(data_points)
for i in range(data_points):
    PPR_values[i] = PPR_function(tau_rec[i],tau_facil[i],u[i],f[i],delta_tn)

Theta_parameters = np.array([tau_rec,tau_facil,u,f,PPR_values])

# TODO: enter the correct path for the global files:
np.save('.../Buchholz_GastoneGuilabert_2023/global_files/Theta_parameters.npy',Theta_parameters)


#Header: Plotting ======================================================================================================
fig = plt.figure()

ax1 = plt.subplot()
ax1.plot(PPR_values,tau_rec, c = 'blue',linewidth = 2)
ax1.plot(PPR_values,tau_facil, c = 'blue',linestyle = 'dashed',linewidth = 2)

ax2 = ax1.twinx()
ax2.plot(PPR_values,u, c = 'red',linewidth = 2)
ax2.plot(PPR_values,f, c = 'red',linestyle = 'dashed',linewidth = 2)
ax2.spines['left'].set_color('blue')
ax2.spines['left'].set_linewidth(2)

ax2.spines['right'].set_color('red')
ax2.spines['right'].set_linewidth(2)

plt.show()