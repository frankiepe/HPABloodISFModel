import os
import sys
top_level_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, top_level_dir)
import numpy as np
from matplotlib import pyplot as plt
import json
from methods.models import HPAModelFEInterCBGAlb
from methods import day_len

# Define parameter config file path
config_file = 'configs/HPAModelFEInterCBGAlb/test_parameters.json'

# Load config
with open(config_file, 'r') as f:
    config = json.load(f)

# Get parameters from config
parameters = config.get('parameters', {})
num_days = 6
days_to_keep = 1

# Create time array
timesteps = day_len * num_days
step = 0.1 # stepsize must be sufficiently small for convergence of dde solver
if day_len/step != int(day_len/step):
    print(f"Warning: day_len ({day_len}) is not divisible by step ({step}). This may cause issues when plotting.")
times = np.arange(0, timesteps, step)

# Initialise model
dde_model = HPAModelFEInterCBGAlb(parameters=parameters, num_days=num_days, days_to_keep=days_to_keep, step=step)

# Run the simulation
print("Running simulation...")       
result = dde_model.simulate(list(parameters.values()), times)
print("Simulation complete.")

# Get plotting times (some cropping may occur if day_len/step != whole number)
plot_times = times[int((day_len/step)*(num_days-days_to_keep)):]

# Plot output
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

F_tot = result.T[1]+result.T[3]+result.T[4]
E_tot = result.T[2]+result.T[5]+result.T[6]

ax1.plot(plot_times, F_tot, label='Total Cortisol', color='blue')
ax1.plot(plot_times, result.T[1], label='Free Cortisol', color='green')
ax1.plot(plot_times, E_tot, label='Total Cortisone', color='red')
ax1.plot(plot_times, result.T[2], label='Free Cortisone', color='yellow')
ax2.plot(plot_times, result.T[0], label='ACTH', color='orange')
ax1.set_ylabel('nmol/L')
ax2.set_ylabel('pmol/L')
ax2.set_xlabel('Time (minutes)')

for ax in [ax1, ax2]:
    ax.set_xlim(plot_times[0], plot_times[-1])
    for i in range(days_to_keep):
        ax.axvline(x=day_len*(i+num_days-days_to_keep), color='gray', linestyle='--') 

crh_drive = [dde_model.crh(t) for t in plot_times]

ax3 = ax1.twinx()
ax3.plot(plot_times, crh_drive, color = 'grey', alpha = 0.4)
ax3.set_ylabel('CRH drive', color = 'grey')

ax4 = ax2.twinx()
ax4.plot(plot_times, crh_drive, color = 'grey', alpha = 0.4)
ax4.set_ylabel('CRH drive', color = 'grey')

ax1.set_title('Cortisol and Cortisone in Blood Plasma')
ax2.set_title('ACTH in Blood Plasma')

ax1.legend()
ax2.legend()

plt.suptitle('Cortisol, Cortisone and ACTH Levels Over Time')
plt.savefig(f'figures/model_output/HPAModelFEInterCBGAlb/test_sim.png')
