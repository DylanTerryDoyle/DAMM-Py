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
y_figsize = x_figsize/2
# fontsize
fontsize = 25
# upper decile 
upper = 0.9
# lower decile
lower = 0.1

### Paths ###

# current working directory path
cwd_path = Path.cwd()
# analysis path
analysis_path = cwd_path / "analysis"
# figure path
figure_path = analysis_path / "figures" / "macro_batch"
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

### start loop over databases ###

for database_path in databases_paths:
    # database suffix 
    suffix = str(database_path.name)[:-3]
    print(f"Analysing results for database {suffix}...")
    # get macro data 
    macro_data = load_macro_data(database_path, params, steps, start)

    ### Plot GDP Ratios ###

    print(f"- creating GDP ratio figures")

    # average over simulations
    macro_group = macro_data.groupby(by="time")
    macro_median = macro_group.quantile(0.5)
    macro_upper = macro_group.quantile(upper)
    macro_lower = macro_group.quantile(lower)
    # create figure
    plt.figure(figsize=(x_figsize,y_figsize))
    # debt ratio
    plt.plot(years, macro_median["debt_ratio"], color="tab:red", linewidth=1)
    plt.fill_between(years, macro_median['debt_ratio'], macro_upper['debt_ratio'], color='tab:red', alpha=0.2)
    plt.fill_between(years, macro_median['debt_ratio'], macro_lower['debt_ratio'], color='tab:red', alpha=0.2)
    # wage share
    plt.plot(years, macro_median["wage_share"], color="tab:green", linewidth=1)
    plt.fill_between(years, macro_median['wage_share'], macro_upper['wage_share'], color='tab:green', alpha=0.2)
    plt.fill_between(years, macro_median['wage_share'], macro_lower['wage_share'], color='tab:green', alpha=0.2)
    # profit share
    plt.plot(years, macro_median["profit_share"], color="tab:blue", linewidth=1)
    plt.fill_between(years, macro_median['profit_share'], macro_upper['profit_share'], color='tab:blue', alpha=0.2)
    plt.fill_between(years, macro_median['profit_share'], macro_lower['profit_share'], color='tab:blue', alpha=0.2)
    # horizontal line at 0
    plt.axhline(0, color='k', linewidth=0.5, alpha=0.75)
    plt.ylim((-0.1,2.1))
    plt.yticks([0, 0.5, 1, 1.5, 2], fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(figure_path / f"gdp_shares_{suffix}", bbox_inches='tight')

    ### Plot Real GDP Growth Distribution ###

    print("- creating real GDP growth distribution figures")

    # simulated real GDP distribution - generalised normal
    num_bins = 100
    std_gdp_growth = (macro_data['rgdp_growth'] - macro_data['rgdp_growth'].mean())/macro_data['rgdp_growth'].std()
    res = np.histogram(std_gdp_growth, bins=num_bins, density=True)
    density = res[0]
    bins = res[1]
    x = np.linspace(-15, 15, 400)

    # gennorm (exponential power/subbotin) fit 
    gennorm_params = stats.gennorm.fit(std_gdp_growth)
    pdf = stats.gennorm.pdf(x, gennorm_params[0], gennorm_params[1], gennorm_params[2])

    plt.figure(figsize=(x_figsize,x_figsize/1.3))
    plt.scatter((bins[1:] + bins[:-1])/2, density, facecolors='none', edgecolors='k', marker='o', label='Simulated')
    plt.plot(x, pdf, color='k', linewidth=1, label='Subbotin')
    plt.ylim([0.00005,1])
    plt.xlim([-22, 22])
    plt.yscale('log')
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize, loc='upper right')
    plt.savefig(figure_path / f"rgdp_growth_dist_{suffix}", bbox_inches='tight')
    print(f'    Subbotin:\n    - beta = {gennorm_params[0]}\n    - mu = {gennorm_params[1]}\n    - alpha = {gennorm_params[2]}')

### Plot Box Plots Scenarios ###

print("Creating box plots...")

scenarios = ["G1", "G2", "ZG1", "ZG2"]

macro_scenario_data = dict()

for scenario, database_path in zip(scenarios, databases_paths):
    macro_scenario_data[scenario] = load_macro_data(database_path, params, steps, start)


yticks = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
xticks = [1, 2, 3, 4]
colours = ['tab:blue', 'tab:blue', 'tab:green', 'tab:green']

### Real GDP Growth ###

box_plot_scenarios(
    macro_scenario_data,
    variable = "rgdp_growth", 
    figsize = (x_figsize, y_figsize),
    fontsize = fontsize, 
    xlabels = scenarios,
    xticks = xticks,
    colours = colours,
    figure_path = figure_path
)

### Productivity Growth ###

box_plot_scenarios(
    macro_scenario_data,
    variable = "productivity_growth", 
    figsize = (x_figsize, y_figsize),
    fontsize = fontsize, 
    xlabels = scenarios,
    xticks = xticks,
    colours = colours,
    figure_path = figure_path
)

### Inflation Rate ###

box_plot_scenarios(
    macro_scenario_data,
    variable = "inflation", 
    figsize = (x_figsize, y_figsize),
    fontsize = fontsize, 
    xlabels = scenarios,
    xticks = xticks,
    colours = colours,
    figure_path = figure_path
)

### Wage Inflation Rate ###

box_plot_scenarios(
    macro_scenario_data,
    variable = "wage_inflation", 
    figsize = (x_figsize, y_figsize),
    fontsize = fontsize, 
    xlabels = scenarios,
    xticks = xticks,
    colours = colours,
    figure_path = figure_path
)

### Unemployment Rate ###

box_plot_scenarios(
    macro_scenario_data,
    variable = "unemployment_rate", 
    figsize = (x_figsize, y_figsize),
    fontsize = fontsize, 
    xlabels = scenarios,
    xticks = xticks,
    colours = colours,
    figure_path = figure_path
)

### Gini Coefficient ###

box_plot_scenarios(
    macro_scenario_data,
    variable = "gini", 
    figsize = (x_figsize, y_figsize),
    fontsize = fontsize, 
    xlabels = scenarios,
    xticks = xticks,
    colours = colours,
    figure_path = figure_path
)

### Credit Rate ###

box_plot_scenarios(
    macro_scenario_data,
    variable = "credit", 
    figsize = (x_figsize, y_figsize),
    fontsize = fontsize, 
    xlabels = scenarios,
    xticks = xticks,
    colours = colours,
    figure_path = figure_path
)

### Probability of a Crises ###

box_plot_scenarios(
    macro_scenario_data,
    variable = "crises_prob", 
    figsize = (x_figsize, y_figsize),
    fontsize = fontsize, 
    xlabels = scenarios,
    xticks = xticks,
    colours = colours,
    figure_path = figure_path
)

print(f"FINISHED MACRO BATCH ANALYSIS! Check your macro_batch figures folder\n=> {figure_path}\n")