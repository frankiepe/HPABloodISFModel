import os
import sys
top_level_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, top_level_dir)
import numpy as np
from matplotlib import pyplot as plt
import json
from methods.models import BaseHPAModel
from methods import day_len

# Define parameter config file path
config_file = 'configs/test_parameters.json'

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
dde_model = BaseHPAModel(parameters=parameters, num_days=num_days, days_to_keep=days_to_keep, step=step)

# Run the simulation
print("Running simulation...")       
result = dde_model.simulate(list(parameters.values()), times)
print("Simulation complete.")

# Get plotting times (some cropping may occur if day_len/step != whole number)
plot_times = times[int((day_len/step)*(num_days-days_to_keep)):]

# Plot output
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
ax1.plot(plot_times, result.T[0])
ax2.plot(plot_times, result.T[1])
ax2.set_xlabel('Time (minutes)')
ax1.set_ylabel('ACTH concentration')
ax2.set_ylabel('Cortisol concentration')

for ax in [ax1, ax2]:
    ax.set_xlim(plot_times[0], plot_times[-1])
    for i in range(days_to_keep):
        ax.axvline(x=day_len*(i+num_days-days_to_keep), color='gray', linestyle='--') 

crh_drive = [dde_model.crh(t) for t in plot_times]

ax3 = ax1.twinx()
ax3.plot(plot_times, crh_drive, color = 'red', alpha = 0.4)
ax3.set_ylabel('CRH drive', color = 'red')

ax4 = ax2.twinx()
ax4.plot(plot_times, crh_drive, color = 'red', alpha = 0.4)
ax4.set_ylabel('CRH drive', color = 'red')

plt.savefig(f'figures/model_output/test_sim.png')
