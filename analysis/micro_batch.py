import numpy as np
import pandas as pd
from pathlib import Path
import scipy.stats as stats
import matplotlib.pyplot as plt
from .utils import load_yaml, load_macro_data, box_plot_scenarios

### Path to database ###

DATABASE_PATH = "F:\\Documents 202507\\University\\University of Sussex\\BSc Dissertation\\Publishing\\MacroABM\\data"

### Plot Parameters ###

# change matplotlib font to serif
plt.rcParams['font.family'] = ['serif']
# figure size
x_figsize = 10
y_figsize = x_figsize/(4/3)
# fontsize
fontsize = 40

### Paths ###

# current working directory path
cwd_path = Path.cwd()
# analysis path
analysis_path = cwd_path / "analysis"
# figure path
figure_path = analysis_path / "figures" / "micro_batch"
# create figure path if it doesn't exist
figure_path.mkdir(parents=True, exist_ok=True)
# parameters path 
params_path = cwd_path / "src" / "macroabm" / "config" / "parameters.yaml"

### load model parameters ###

# parameters
params = load_yaml(params_path)
# analysis parameters
steps = params['simulation']['steps']
num_years = params['simulation']['years']
start = params['simulation']['start']*steps
end = (start + params['simulation']['years'])*steps
middle = int((end + start)/2)
years = np.linspace(0, num_years, num_years*steps)

### paths to data ###
# get database_path from parameters
data_path = cwd_path / "data"
# check if dynamically set database path exists
if data_path.exists() and data_path.is_dir():
    # get database names
    databases_paths = [f for f in data_path.iterdir() if f.is_file()]
else:
    # other wise use manual database path
    data_path = Path(DATABASE_PATH)
    try:
        databases_paths = [f for f in data_path.iterdir() if f.is_file()]
    except FileNotFoundError as e:
        # raise an error if neither exist
        print(f"Error: no data folder with databases and no folder found at manual location {DATABASE_PATH}")
        print(e)  # prints the original FileNotFoundError message
        databases_paths = []

### Plot Box Plots Scenarios ###

print("Creating box plots...")

scenarios = ["G1", "G2", "ZG1", "ZG2"]

macro_scenario_data = dict()

for scenario, database_path in zip(scenarios, databases_paths):
    macro_scenario_data[scenario] = load_macro_data(database_path, params, steps, start)


yticks = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
xticks = [1, 2, 3, 4]
colours = ['tab:blue', 'tab:blue', 'tab:green', 'tab:green']

print("- creating HPI plots")

### Consumption Firm HPI ###

box_plot_scenarios(
    macro_scenario_data,
    variable = "cfirm_hpi", 
    figsize = (x_figsize, y_figsize),
    fontsize = fontsize, 
    xlabels = scenarios,
    xticks = xticks,
    colours = colours,
    figure_path = figure_path
)

### Capital Firm HPI ###

box_plot_scenarios(
    macro_scenario_data,
    variable = "kfirm_hpi", 
    figsize = (x_figsize, y_figsize),
    fontsize = fontsize, 
    xlabels = scenarios,
    xticks = xticks,
    colours = colours,
    figure_path = figure_path
)

### Bank HPI ###

box_plot_scenarios(
    macro_scenario_data,
    variable = "bank_hpi", 
    figsize = (x_figsize, y_figsize),
    fontsize = fontsize, 
    xlabels = scenarios,
    xticks = xticks,
    colours = colours,
    figure_path = figure_path
)

print("- creating normalised HHI plots")

### Consumption Firm Normalised HHI ###

box_plot_scenarios(
    macro_scenario_data,
    variable = "cfirm_nhhi", 
    figsize = (x_figsize, y_figsize),
    fontsize = fontsize, 
    xlabels = scenarios,
    xticks = xticks,
    colours = colours,
    figure_path = figure_path
)

### Capital Firm Normalised HHI ###

box_plot_scenarios(
    macro_scenario_data,
    variable = "kfirm_nhhi", 
    figsize = (x_figsize, y_figsize),
    fontsize = fontsize, 
    xlabels = scenarios,
    xticks = xticks,
    colours = colours,
    figure_path = figure_path
)

### Bank Normalised HHI ###

box_plot_scenarios(
    macro_scenario_data,
    variable = "bank_nhhi", 
    figsize = (x_figsize, y_figsize),
    fontsize = fontsize, 
    xlabels = scenarios,
    xticks = xticks,
    colours = colours,
    figure_path = figure_path
)

print("- creating normal default probabilities plots")

### Consumption Firm Normal Prob Default ###

box_plot_scenarios(
    macro_scenario_data,
    variable = "cfirm_prob_crises0", 
    figsize = (x_figsize, y_figsize),
    fontsize = fontsize, 
    xlabels = scenarios,
    xticks = xticks,
    colours = colours,
    figure_path = figure_path
)

### Capital Firm Normal Prob Default ###

box_plot_scenarios(
    macro_scenario_data,
    variable = "kfirm_prob_crises0", 
    figsize = (x_figsize, y_figsize),
    fontsize = fontsize, 
    xlabels = scenarios,
    xticks = xticks,
    colours = colours,
    figure_path = figure_path
)

### Bank Normal Prob Default ###

box_plot_scenarios(
    macro_scenario_data,
    variable = "bank_prob_crises0", 
    figsize = (x_figsize, y_figsize),
    fontsize = fontsize, 
    xlabels = scenarios,
    xticks = xticks,
    colours = colours,
    figure_path = figure_path
)

print("- creating crisis default probabilities plots")

### Consumption Firm Crises Prob Default ###

box_plot_scenarios(
    macro_scenario_data,
    variable = "cfirm_prob_crises1", 
    figsize = (x_figsize, y_figsize),
    fontsize = fontsize, 
    xlabels = scenarios,
    xticks = xticks,
    colours = colours,
    figure_path = figure_path
)

### Capital Firm Crises Prob Default ###

box_plot_scenarios(
    macro_scenario_data,
    variable = "kfirm_prob_crises1", 
    figsize = (x_figsize, y_figsize),
    fontsize = fontsize, 
    xlabels = scenarios,
    xticks = xticks,
    colours = colours,
    figure_path = figure_path
)

### Bank Crises Prob Default ###

box_plot_scenarios(
    macro_scenario_data,
    variable = "bank_prob_crises1", 
    figsize = (x_figsize, y_figsize),
    fontsize = fontsize, 
    xlabels = scenarios,
    xticks = xticks,
    colours = colours,
    figure_path = figure_path
)

print(f"FINISHED MICRO BATCH ANALYSIS! Check your micro_batch figures folder\n=> {figure_path}\n")