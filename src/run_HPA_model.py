import numpy as np
from matplotlib import pyplot as plt
import math
import json
from ddeint import ddeint

# Define parameter config file path
config_file = 'configs/mean_parameters.json'

# Load config
with open(config_file, 'r') as f:
    config = json.load(f)

# Get parameters from config
suggested_params = config.get('parameters', {})
fixed_params = config.get('fixed_params', {})
num_days = config.get('num_days', 4)
parameters = {**fixed_params, **suggested_params}

h = parameters['h']
k_a = parameters['k_a']
k_c = parameters['k_c']
alpha = parameters['alpha']
delay = parameters['delay']
t_s = parameters['t_s']
gamma_a = parameters['gamma_a']
gamma_c = parameters['gamma_c']
sigma = parameters['sigma']
m_a = parameters['m_a']
m_c = parameters['m_c']
lambda_a = parameters['lambda_a']
lambda_s = parameters['lambda_s']

# Create time array
timesteps = 1440 * num_days
step = 0.1
times = np.arange(0, timesteps, step)

# Define CRH function
def crh(t, t_s, lambda_a, lambda_s, sigma, T_c=1440):
    crh  =  lambda_a * math.e**(lambda_s*math.cos(2*math.pi*((t-t_s)/T_c)+sigma*math.cos(2*math.pi*((t-t_s)/T_c))))
    #crh = 70*math.cos(2*math.pi*(t/T_c))+75 
    return crh

# Define the DDE model
def model(Y, t):
    A, C = Y(t)
    C_delay = Y(t - delay)[1]

    dAdt = -gamma_a*A + h*((k_c**m_a)*crh(t, t_s, lambda_a, lambda_s, sigma))/(k_c**m_a+C_delay**m_a)
    dCdt = -gamma_c*C + alpha*((A**m_c)/(k_a**m_c + A**m_c))

    return [dAdt, dCdt]

# Define initial conditions
def initial_conditions(t):
    return [5, 400]  

# Run the simulation
print("Running simulation...")       
result = ddeint(model, initial_conditions, times)
print("Simulation complete.")

# Plot output
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 5))
ax1.plot(times, result.T[0])
ax2.plot(times, result.T[1])
ax2.set_xlabel('Time (minutes)')
ax1.set_ylabel('ACTH concentration')
ax2.set_ylabel('Cortisol concentration')

for ax in [ax1, ax2]:
    ax.set_xlim(0, times[-1])
    for i in range(num_days):
        ax.axvline(x=1440*i, color='gray', linestyle='--') 

crh_drive = []
for t in times:
    crh_drive.append(crh(t, 0, lambda_a, lambda_s, sigma))

ax3 = ax1.twinx()
ax3.plot(times, crh_drive, color = 'red', alpha = 0.4)
ax3.set_ylabel('CRH drive', color = 'red')

ax4 = ax2.twinx()
ax4.plot(times, crh_drive, color = 'red', alpha = 0.4)
ax4.set_ylabel('CRH drive', color = 'red')

plt.savefig(f'figures/model_output/test_sim.png')

