import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from analysis.utils import load_yaml, load_macro_data, load_micro_data, box_plot_scenarios, large_ages, small_ages, calculate_bank_age

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

### paths to data ###
# get database_path from parameters
data_path = cwd_path / "data"
# check if dynamically set database path exists
if data_path.exists() and data_path.is_dir():
    # get database names
    database_paths = [f for f in data_path.iterdir() if f.is_file()]
else:
    # other wise use manual database path
    data_path = Path(DATABASE_PATH)
    try:
        database_paths = [f for f in data_path.iterdir() if f.is_file()]
    except FileNotFoundError as e:
        # raise an error if neither exist
        print(f"Error: no data folder with databases and no folder found at manual location {DATABASE_PATH}")
        print(e)  # prints the original FileNotFoundError message
        database_paths = []

### Plot Box Plots Scenarios ###

print("Creating box plots...")

# scenario names
scenarios = ["G1", "G2", "ZG1", "ZG2"]

# macro data for all scenarios
macro_scenario_data = {scenario: load_macro_data(database_path, params, steps, start) for scenario, database_path in zip(scenarios, database_paths)}

# figure settings
xticks = [1, 2, 3, 4]
colours = ['tab:blue', 'tab:blue', 'tab:green', 'tab:green']

### HPI Plots ###

print("- creating HPI plots")

# Consumption Firm HPI

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

# Capital Firm HPI 

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

# Bank HPI 

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

### Normalised HHI ###

print("- creating normalised HHI plots")

# Consumption Firm Normalised HHI 

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

# Capital Firm Normalised HHI 

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

# Bank Normalised HHI 

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

### Normal Prob Default ###

print("- creating normal default probabilities plots")

# Consumption Firm Normal Prob Default 

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

# Capital Firm Normal Prob Default 

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

# Bank Normal Prob Default 

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

### Crises Prob Default ###

print("- creating crisis default probabilities plots")

# Consumption Firm Crises Prob Default 

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

# Capital Firm Crises Prob Default

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

# Bank Crises Prob Default 

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

### Age box plots ###

# market share percentiles
# large agents
q_large = 0.99
# small agents
q_small = 0.5

### Consumption Firm Ages ###

print("- creating cfirm age plots...")

# cfirm age data
cfirm_query = f"""
    SELECT
        simulation,
        time,
        market_share,
        age
    FROM 
        firm_data
    WHERE 
        firm_type = 'ConsumptionFirm'
        AND time > {start};
"""

# cfirm data for all scenarios
cfirm_scenario_data = {scenario: load_micro_data(cfirm_query, database_path) for scenario, database_path in zip(scenarios, database_paths)}

# cfirm ages
cfirm_scenario_large_ages = dict()
cfirm_scenario_small_ages = dict()

for scenario in cfirm_scenario_data.keys():
    # rescale age
    cfirm_scenario_data[scenario]["age"] = num_years*((cfirm_scenario_data[scenario]["age"] - cfirm_scenario_data[scenario]["age"].min())/(cfirm_scenario_data[scenario]["age"].max() - cfirm_scenario_data[scenario]["age"].min()))
    # age of large cfirms 
    cfirm_scenario_large_ages[scenario], large_market_share = large_ages(cfirm_scenario_data[scenario], q_large)
    # age of small cfirms
    cfirm_scenario_small_ages[scenario], small_market_share = small_ages(cfirm_scenario_data[scenario], q_small)
    # print market share cutoff    
    print(f"  - Scenario {scenario} large cfirms market share: {large_market_share}")
    print(f"  - Scenario {scenario} small cfirms market share: {small_market_share}\n")

# cfirm large age box plots

box_plot_scenarios(
    cfirm_scenario_large_ages,
    variable = "age", 
    figsize = (x_figsize, y_figsize),
    fontsize = fontsize, 
    xlabels = scenarios,
    xticks = xticks,
    colours = colours,
    figure_path = figure_path
)

# cfirm small age box plots

box_plot_scenarios(
    cfirm_scenario_small_ages,
    variable = "age", 
    figsize = (x_figsize, y_figsize),
    fontsize = fontsize, 
    xlabels = scenarios,
    xticks = xticks,
    colours = colours,
    figure_path = figure_path
)

### Capital Firm Ages ###

print("- creating kfirm age plots...")

# kfirm age data
kfirm_query = f"""
    SELECT
        simulation,
        time,
        market_share,
        age
    FROM 
        firm_data
    WHERE 
        firm_type = 'CapitalFirm'
        AND time > {start};
"""

# kfirm data for all scenarios
kfirm_scenario_data = {scenario: load_micro_data(kfirm_query, database_path) for scenario, database_path in zip(scenarios, database_paths)}

# kfirm ages
kfirm_scenario_large_ages = dict()
kfirm_scenario_small_ages = dict()

for scenario in kfirm_scenario_data.keys():
    # rescale age
    kfirm_scenario_data[scenario]["age"] = num_years*((kfirm_scenario_data[scenario]["age"] - kfirm_scenario_data[scenario]["age"].min())/(kfirm_scenario_data[scenario]["age"].max() - kfirm_scenario_data[scenario]["age"].min()))
    # age of large kfirms 
    kfirm_scenario_large_ages[scenario], large_market_share = large_ages(kfirm_scenario_data[scenario], q_large)
    # age of small kfirms
    kfirm_scenario_small_ages[scenario], small_market_share = small_ages(kfirm_scenario_data[scenario], q_small)
    # print market share cutoff    
    print(f"  - Scenario {scenario} large kfirms market share: {large_market_share}")
    print(f"  - Scenario {scenario} small kfirms market share: {small_market_share}\n")

# kfirm large age box plots

box_plot_scenarios(
    kfirm_scenario_large_ages,
    variable = "age", 
    figsize = (x_figsize, y_figsize),
    fontsize = fontsize, 
    xlabels = scenarios,
    xticks = xticks,
    colours = colours,
    figure_path = figure_path
)

# kfirm small age box plots

box_plot_scenarios(
    kfirm_scenario_small_ages,
    variable = "age", 
    figsize = (x_figsize, y_figsize),
    fontsize = fontsize, 
    xlabels = scenarios,
    xticks = xticks,
    colours = colours,
    figure_path = figure_path
)

### Bank Ages ###

print("- creating bank age plots...")

# kfirm age data
bank_query = f"""
    SELECT 
        simulation, 
        time, 
        id,
        min_capital_ratio,
        loans, 
        reserves,
        equity,
        market_share
    FROM 
        bank_data
    WHERE time > {start};
"""

# bank data for all scenarios
bank_scenario_data = {scenario: load_micro_data(bank_query, database_path) for scenario, database_path in zip(scenarios, database_paths)}

# bank ages
bank_scenario_large_ages = dict()
bank_scenario_small_ages = dict()

for scenario in bank_scenario_data.keys():
    # calculate bank age (already rescaled)
    bank_scenario_data[scenario]["age"] = calculate_bank_age(bank_scenario_data[scenario])
    # age of large banks 
    bank_scenario_large_ages[scenario], large_market_share = large_ages(bank_scenario_data[scenario], q_large)
    # age of small banks
    bank_scenario_small_ages[scenario], small_market_share = small_ages(bank_scenario_data[scenario], q_small)
    # print market share cutoff    
    print(f"  - Scenario {scenario} large banks market share: {large_market_share}")
    print(f"  - Scenario {scenario} small banks market share: {small_market_share}\n")

# bank large age box plots

box_plot_scenarios(
    bank_scenario_large_ages,
    variable = "age", 
    figsize = (x_figsize, y_figsize),
    fontsize = fontsize, 
    xlabels = scenarios,
    xticks = xticks,
    colours = colours,
    figure_path = figure_path
)

# bank small age box plots

box_plot_scenarios(
    bank_scenario_small_ages,
    variable = "age", 
    figsize = (x_figsize, y_figsize),
    fontsize = fontsize, 
    xlabels = scenarios,
    xticks = xticks,
    colours = colours,
    figure_path = figure_path
)

print(f"FINISHED MICRO BATCH ANALYSIS! Check your micro_batch figures folder\n=> {figure_path}\n")