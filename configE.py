# %%
# Importing packages
import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt

# %%
# Configuration E
# %%
# DATA from Subraveti et al. 2019
# Configuration E
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
def CO2cost_E(t):
    x_CO2 = t2x_E(t)
    R_CO2 = x2R_E(x_CO2)
    CO2cost_total = 10*np.exp(1/(100-x_CO2))
    CO2cost_per_CO2 = CO2cost_total/R_CO2*100
    return CO2cost_per_CO2

CO2_test = CO2cost_E(t_dom_test)
xCO2_test = t2x_E(t_dom_test)
plt.figure(dpi=80)
#plt.plot(t_dom_test, -CO2_test, 'o')
plt.plot(xCO2_test, CO2_test, 'o')
plt.xlabel('CO2 purity (%)')
plt.ylabel('Cost for CO2 capture ($)')

def CO2cost_E_w_constraint(t):
    x_CO2 = t2x_E(t)
    R_CO2 = x2R_E(x_CO2)
    constraint = 0
    if x_CO2<95:
        constraint = 400*(x_CO2-95)**2
    CO2cost_total = 10*np.exp(1/(100-x_CO2))
    CO2cost_per_CO2 = CO2cost_total/R_CO2*100 + constraint
    return CO2cost_per_CO2

# %%
# Optimization
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
t0 = 1.0
t_ran = [[0,2]]
opt_res = minimize(CO2cost_E_w_constraint, t0, method = 'Nelder-mead')
opt_res2 = differential_evolution(CO2cost_E_w_constraint, t_ran,)
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
# Configuration of F
# %%
# DATA from Subraveti et al. 2019
# COnfiguration F
CO2_purity_F = [-0.001,
                    84.71922246,
                    88.08855292,
                    90.24838013,
                    92.32181425,
                    95.45356371,
                    96.20950324,
                    97.20302376,
                    97.76457883,
                    98.28293737,
                    98.52051836,
                    98.58531317,
                    98.67170626,
                    98.71490281,
                    98.8012959,
                    98.8012959+0.0001]
CO2_recovery_F = [99.9137931+0.0001,
                    99.9137931,
                    99.74137931,
                    99.82758621,
                    99.74137931,
                    99.56896552,
                    99.48275862,
                    98.79310345,
                    97.84482759,
                    96.20689655,
                    90.34482759,
                    82.24137931,
                    66.03448276,
                    61.37931034,
                    41.37931034,
                    -0.0001]
xmax_F = 98.8012959
x2R_F = PchipInterpolator(CO2_purity_F, CO2_recovery_F, True)
# %%
def t2x_F(t):
    x_return = xmax_F/2*np.tanh(3*t)+xmax_F/2
    return x_return
t_dom_test = np.linspace(-5,5,501)
x_F_test = t2x_F(t_dom_test)
R_F_test = x2R_F(x_F_test)

x_dom_test = np.linspace(10,95,101)
R_F_test2 = x2R_F(x_dom_test)

plt.figure(dpi = 80)
plt.plot(CO2_purity_F, CO2_recovery_F, 'ro')
#plt.plot(x_dom_test, R_D_test2,)
plt.plot(x_F_test, R_F_test)
plt.xlabel('CO$_2$ purity (%)')
plt.ylabel('CO$_2$ recovery (%)')
plt.grid(linestyle = ':')
plt.xlim([40, 105])
plt.ylim([40, 105])

# %%
# Example of Optimizationd
# You have to make the following functin
# Using cost estimation results
def CO2cost_F(t):
    x_CO2 = t2x_F(t)
    R_CO2 = x2R_F(x_CO2)
    CO2cost_total = 10*np.exp(1/(100-x_CO2))
    CO2cost_per_CO2 = CO2cost_total/R_CO2*100
    return CO2cost_per_CO2

CO2_test = CO2cost_F(t_dom_test)
xCO2_test = t2x_F(t_dom_test)
plt.figure(dpi=80)
#plt.plot(t_dom_test, -CO2_test, 'o')
plt.plot(xCO2_test, CO2_test, 'o')
plt.xlabel('CO2 purity (%)')
plt.ylabel('Cost for CO2 capture ($)')

def CO2cost_F_w_constraint(t):
    x_CO2 = t2x_F(t)
    R_CO2 = x2R_F(x_CO2)
    constraint = 0
    if x_CO2<95:
        constraint = 400*(x_CO2-95)**2
    CO2cost_total = 10*np.exp(1/(100-x_CO2))
    CO2cost_per_CO2 = CO2cost_total/R_CO2*100 + constraint
    return CO2cost_per_CO2

# %%
# Optimization
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
t0 = 1.0
t_ran = [[0,2]]
opt_res = minimize(CO2cost_F_w_constraint, t0, method = 'Nelder-mead')
opt_res2 = differential_evolution(CO2cost_F_w_constraint, t_ran,)
t_sol = opt_res.x
x_sol = t2x_F(t_sol)
R_sol = x2R_F(x_sol)

t_sol2 = opt_res2.x
x_sol2 = t2x_F(t_sol2)
R_sol2 = x2R_F(x_sol2)
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
# %%
# Configuration of D
# %%
# DATA from Subraveti et al. 2019
# COnfiguration D
CO2_purity_D = [-0.001,
                    81.96428571,
                    97.33258929,
                    99.00669643,
                    99.47544643,
                    99.81026786,
                    99.91071429,
                    99.97767857,
                    99.97767858,
                    99.97767857+0.0001]
CO2_recovery_D = [99.86666667+0.0001,
                    99.86666667,
                    99.6,
                    99.6,
                    99.46666667,
                    98.93333333,
                    98.13333333,
                    95.6,
                    43.2,
                    -0.0001]
xmax_D = 99.975
x2R_D = PchipInterpolator(CO2_purity_D, CO2_recovery_D, True)
# %%
def t2x_D(t):
    x_return = xmax_D/2*np.tanh(3*t)+xmax_D/2
    return x_return
t_dom_test = np.linspace(-5,5,501)
x_D_test = t2x_D(t_dom_test)
R_D_test = x2R_D(x_D_test)

x_dom_test = np.linspace(10,95,101)
R_D_test2 = x2R_D(x_dom_test)

plt.figure(dpi = 80)
plt.plot(CO2_purity_D, CO2_recovery_D, 'ro')
#plt.plot(x_dom_test, R_D_test2,)
plt.plot(x_D_test, R_D_test)
plt.xlabel('CO$_2$ purity (%)')
plt.ylabel('CO$_2$ recovery (%)')
plt.grid(linestyle = ':')
plt.xlim([40, 105])
plt.ylim([40, 105])

# %%
# Example of Optimizationd
# You have to make the following functin
# Using cost estimation results
def CO2cost_D(t):
    x_CO2 = t2x_D(t)
    R_CO2 = x2R_D(x_CO2)
    CO2cost_total = 10*np.exp(1/(100-x_CO2))
    CO2cost_per_CO2 = CO2cost_total/R_CO2*100
    return CO2cost_per_CO2

CO2_test = CO2cost_D(t_dom_test)
xCO2_test = t2x_D(t_dom_test)
plt.figure(dpi=80)
#plt.plot(t_dom_test, -CO2_test, 'o')
plt.plot(xCO2_test, CO2_test, 'o')
plt.xlabel('CO2 purity (%)')
plt.ylabel('Cost for CO2 capture ($)')

def CO2cost_D_w_constraint(t):
    x_CO2 = t2x_D(t)
    R_CO2 = x2R_D(x_CO2)
    constraint = 0
    if x_CO2<95:
        constraint = 1E8*(x_CO2-95)**2
    CO2cost_total = 10*np.exp(1/(100-x_CO2))
    CO2cost_per_CO2 = CO2cost_total/R_CO2*100 + constraint
    return CO2cost_per_CO2

# %%
# Optimization
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
t0 = 1.0
t_ran = [[0,2]]
opt_res = minimize(CO2cost_D_w_constraint, t0, method = 'Nelder-mead')
opt_res2 = differential_evolution(CO2cost_D_w_constraint, t_ran,)
t_sol = opt_res.x
x_sol = t2x_F(t_sol)
R_sol = x2R_F(x_sol)

t_sol2 = opt_res2.x
x_sol2 = t2x_F(t_sol2)
R_sol2 = x2R_F(x_sol2)
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
