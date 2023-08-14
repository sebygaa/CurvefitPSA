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
xmax_E = 97.15982721
x2R_E = PchipInterpolator(CO2_purity_E, CO2_recovery_E, True)
# %%
# intermediate parameter (t) to x (purity) and R (recovery)
def t2x_E(t):
    x_return = xmax_E/2*np.tanh(3*t)+xmax_E/2
    return x_return
t_dom_test = np.linspace(-5,5,501)
x_E_test = t2x_E(t_dom_test)
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
# Example of Optimizationd
# You have to make the following functin
# Using cost estimation results
def CO2cost(t):
    x_CO2 = t2x_E(t)
    R_CO2 = x2R_E(x_CO2)
    CO2cost_per_CO2 = 10*np.exp(1/(100-x_CO2))
    CO2cost_total = CO2cost_per_CO2*R_CO2*100
    return -CO2cost_total

CO2_test = CO2cost(t_dom_test)
plt.figure(dpi=80)
plt.plot(t_dom_test, -CO2_test, 'o')

# %%
# Optimization
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
t0 = 0.8
t_ran = [[0,2]]
opt_res = minimize(CO2cost, t0, method = 'Nelder-mead')
opt_res2 = differential_evolution(CO2cost, t_ran,)
t_sol =opt_res.x
x_sol = t2x_E(t_sol)
R_sol = x2R_E(x_sol)

t_sol2 = opt_res2.x
x_sol2 = t2x_E(t_sol2)
R_sol2 = x2R_E(x_sol2)
print('=== FIRST SOLVER ===')
print(opt_res)
print('t= ', t_sol)
print('Optimal CO2 purity = ', x_sol)
print('Optimal CO2 recovery=', R_sol)
print()
print('=== SECOND SOLVER ===')
print(opt_res2)
print('t= ', t_sol2)
print('Optimal CO2 purity = ', x_sol2)
print('Optimal CO2 recovery=', R_sol2)

# %%
