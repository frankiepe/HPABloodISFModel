import os
import sys
top_level_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, top_level_dir)
import pandas as pd
import numpy as np
import json
import argparse
import methods.plotting as plotting
from methods import model_dict
import seaborn as sns
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Fit models to data.')
parser.add_argument('-m', '--model', type = str, help='Select model for optimisation: ' \
                    '1: BaseHPAModel, 2: HPAModelFEInter, 3a: HPAModelFEInterCBGAlbSimple, ' \
                    '3b: HPAModelFEInterCBGAlb, 4: HPAModelFEInterCBGAlbBloodISF, 5: HPAModelFEInterBothCBGAlbBloodISF, ' \
                    '6: HPAModelCRHSupp')
parser.add_argument('-n', '--ninds', type=int, help='Number of datasets fitted')
parser.add_argument('-o', '--outdir', type=str, help='Directory for plotting output')
parser.add_argument('-f', '--fixed', type=str, nargs='*', help='Parameters to fix (i.e. not fit)')
parser.add_argument('-t', '--thresh', type = int, default=2000, help='Objective threshold for ABC filter')
args = parser.parse_args()

def process_ABC_pop(m_n, num_inds, outdir, fixed, thresh):
    # Get config file
    init_pars_file = f'configs/{model_dict[m_n]}/test_parameters.json'

    # Load config
    with open(init_pars_file, 'r') as f:
        config = json.load(f)

    # Get parameters from config
    init_pars = config.get('parameters', {})
    fixed_pars = init_pars.copy()
    
    # Filter fixed vs. non-fixed parameters
    all_par_keys = list(init_pars.keys())
    for par_key in all_par_keys:
        if par_key in fixed:
            init_pars.pop(par_key)
        else:
            fixed_pars.pop(par_key)

    pars_accept = []
    for d_n in range(1,num_inds+1):
        df = pd.read_csv("output/" + outdir + f"/{model_dict[m_n]}/pars" + f"/all_pars_model{m_n}_dataID{d_n}.csv")
        df_thresh = df[df.Obj<=thresh] 
        for _, row in df_thresh.iterrows():
            pars_accept.append([float(row[i]) for i in init_pars.keys()])

    print(f"Accepted sets: {len(pars_accept)}")

    # Plot histograms of accepted parameters and correlation plot
    if len(pars_accept) > 0:
        for i in np.arange(0, len(init_pars.keys())):
            param_name = list(init_pars.keys())[i]
            param_values = [p[i] for p in pars_accept]

            if outdir is not None:
                hist_file = os.path.join("output/" + outdir + f"/{model_dict[m_n]}/plots", f"hist_{param_name}_model{m_n}_pop.png")
            else:
                hist_file = f"hist_{param_name}_model{m_n}_pop.png"

            plotting.plot_parameter_histograms(param_values, param_name, hist_file, thresh)
            print(f"Saved histogram for '{param_name}' to: {hist_file}")

        df = pd.DataFrame(pars_accept, columns=init_pars.keys())
        sns.pairplot(df, kind="kde", corner=True)
        plt.savefig("output/" + outdir + f"/{model_dict[m_n]}/plots" + f"/pairplot_model{m_n}_pop.png")
    else:
        print("No accepted parameters were found; histogram was not created.")

if __name__ == "__main__":
    m_n = args.model
    num_inds = args.ninds
    outdir = args.outdir
    fixed = args.fixed
    thresh = args.thresh
    if fixed is None:
        fixed = {}

    # Make directories if necessary
    if not os.path.exists(f'output/{outdir}'):
        os.makedirs(f'output/{outdir}', exist_ok=True)
    if not os.path.exists(f'output/{outdir}/{model_dict[m_n]}/pars'):
        os.makedirs(f'output/{outdir}/{model_dict[m_n]}/pars', exist_ok=True)
    if not os.path.exists(f'output/{outdir}/{model_dict[m_n]}/plots'):
        os.makedirs(f'output/{outdir}/{model_dict[m_n]}/plots', exist_ok=True)

    # Process ABC output
    process_ABC_pop(m_n, num_inds, outdir, fixed, thresh)
    
