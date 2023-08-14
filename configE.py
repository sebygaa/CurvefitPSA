# %%
# Importing packages
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
# %%
# DATA from Subraveti et al. 2019
CO2_purity_E = [-0.001,
                    85.66954644,
                    89.77321814,
                    93.66090713,
                    95.21598272,
                    96.07991361,
                    96.72786177,
                    96.90064795,
                    97.03023758,
                    97.11663067,
                    97.15982721,
                    97.15982721+0.0001]
CO2_recovery_E = [99.82758621+0.0001,
                    99.82758621,
                    99.82758621,
                    99.31034483,
                    98.10344828,
                    96.37931034,
                    91.03448276,
                    85.17241379,
                    69.65517241,
                    52.5862069,
                    40,
                    -0.0001]
Rmax_E = 97.15982721
x2R_E = PchipInterpolator(CO2_purity_E, CO2_recovery_E, True)
# %%
# intermediate parameter (t) to x (purity) and R (recovery)
def t2R_D(t):
    R_return = Rmax_E/2*np.tanh(3*t)+Rmax_E/2
    return R_return
t_dom_test = np.linspace(-5,5,501)
x_E_test = t2R_D(t_dom_test)
R_E_test = x2R_E(x_E_test)

x_dom_test = np.linspace(10,95,101)
R_E_test2 = x2R_E(x_dom_test)

plt.figure(dpi = 80)
plt.plot(CO2_purity_E, CO2_recovery_E, 'ro')
#plt.plot(x_dom_test, R_D_test2,)
plt.plot(x_E_test, R_E_test)
plt.xlabel('CO$_2$ purity (%)')
plt.ylabel('CO$_2$ recovery (%)')
plt.grid(linestyle = ':')
plt.xlim([40, 105])
plt.ylim([40, 105])

# %%
# Example of Optimization
