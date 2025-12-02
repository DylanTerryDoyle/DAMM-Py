import sqlite3
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
# add parent directory to system path
from analysis.utils import load_yaml, bank_debtrank, expected_systemic_loss

### Custom Database Location ###

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
figure_path = analysis_path / "figures" / "expected_systemic_loss"
# create figure path if it doesn't exist
figure_path.mkdir(parents=True, exist_ok=True)
# parameters path 
params_path = cwd_path / "src" / "macroabm" / "config" / "parameters.yaml"

### parameters ###

# parameters
params = load_yaml(params_path)
# analysis parameters
steps = params['simulation']['steps']
start = params['simulation']['start']*steps
years = np.linspace(0, params['simulation']['years'], params['simulation']['years']*steps)

### Paths to Data ###

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

### connect to sql database ###

# database index
for database_path in database_paths:
    # database suffix 
    suffix = str(database_path.name)[:-3]
    print(f"Analysing ESL for database {suffix}...")
    # create connection to database
    con = sqlite3.connect(database_path)

    ### Data for DebtRank ###
    
    # get edges data
    print("- Downloading credit network edge data...")
    edges = pd.read_sql_query(
        f"""
            SELECT *
            FROM 
                edge_data
            WHERE 
                time > {start}
        ;
        """,
        con
    )

    # get macro data
    print("- Downloading macro data...")
    macro = pd.read_sql_query(
        f"""
            SELECT *
            FROM 
                macro_data
            WHERE 
                time > {start}
        ;
        """,
        con
    )

    # get bank data
    print("- Downloading bank data...")
    banks = pd.read_sql_query(
        f"""
            SELECT *
            FROM 
                bank_data
            WHERE 
                time > {start}
        ;
        """,
        con
    )

    # bankruptcy flag
    banks['bankrupt'] = banks['min_capital_ratio']*(banks['loans'] + banks['reserves']) == banks['equity']

    # probability of default
    bank_group = banks.groupby(["simulation", "id"])
    prob_default = bank_group[["bankrupt"]].mean()
    prob_default.reset_index(inplace=True)
    prob_default.rename(columns={"bankrupt": "prob_default"}, inplace=True)

    # join prob default to edges data
    print("- Joining edge and bank probability of default...")
    edges = edges.merge(prob_default, left_on=["simulation", "source"], right_on=["simulation", "id"], how='left')
    edges.drop(columns=["id"], inplace=True)

    ### start debtrank calculation ###

    print(f"- Calculating DebtRank & ESL for scenario {suffix}...")
    # number of simulations 
    num_sims = len(edges['simulation'].unique())
    # number of periods per simulation
    num_periods = len(years)
    # array to hold expected systemic loss as a percentage of nominal GDP results
    esl_gdp = np.zeros(shape=(num_sims,num_periods))
    # simulation loop
    for s in tqdm(range(num_sims)):
        # time period loop for simulation s
        for t in range(num_periods):
            # edge data for current time period
            current_edges = edges.loc[(edges["simulation"] == s) & (edges["time"] == start + t+1)]
            # bank assets in period t
            bank_assets = current_edges.sort_values(by=["source"]).drop_duplicates(subset=["source"]).loc[:,["bank_assets"]].to_numpy().ravel()
            # firm assets in period t
            firm_assets = current_edges.sort_values(by=["target"]).drop_duplicates(subset=["target"]).loc[:,["firm_assets"]].to_numpy().ravel()
            # bank probability of default
            bank_prob_default = current_edges.sort_values(by=["target"]).drop_duplicates(subset=["source"]).loc[:,["prob_default"]].to_numpy().ravel()
            # credit network loan matrix
            C = pd.pivot_table(current_edges, values=["loan"], index=["source"], columns=["target"]).fillna(0).to_numpy()
            # number of banks 
            num_banks, num_firms = C.shape
            # Bank loan vector: shape => (num_banks, 1)
            C_banks = C.sum(axis=1).reshape(-1,1)
            # Bank propagation matrix
            W_banks = np.divide(C, C_banks, out = np.zeros_like(C, dtype=np.float64), where = C_banks != 0)
            # Firm loan vector: shape => (num_firms, 1)
            C_firms = C.sum(axis=0).reshape(-1,1) 
            # Firm propagation matrix
            W_firms = np.divide(C.T, C_firms, out=np.zeros_like(C.T, dtype=np.float64), where=C_firms != 0)
            # compute debtrank for both banks and firms if each bank fails 
            bank_dr, firm_dr = bank_debtrank(W_banks, W_firms, bank_assets, firm_assets, num_banks, num_firms)
            # expected systemic loss
            esl = expected_systemic_loss(bank_dr, firm_dr, bank_prob_default, bank_assets, firm_assets)
            # nominal GDP in period t
            nominal_gdp = macro.loc[(macro["simulation"] == s) & (macro["time"] == start + t+1)]["nominal_gdp"].iloc[0]
            # expected systemic loss as a percentage of nomnal GDP
            esl_gdp[s,t] = esl/nominal_gdp
    
    ### save results ###
    print("- Saving results to CSV file")
    df_esl_gdp = pd.DataFrame(esl_gdp).to_csv(f"esl_gdp_{suffix}.csv")
    
    ### plot results ###
    
    # median esl over simulations
    median = np.quantile(esl_gdp, q=0.5, axis=0)
    # Top 9th decile
    upper = np.quantile(esl_gdp, q=upper, axis=0)
    # Top 1st decile
    lower = np.quantile(esl_gdp, q=lower, axis=0)

    plt.figure(figsize=(x_figsize,y_figsize))
    plt.plot(years, median, color='k', linewidth=1, label='Mean')
    plt.fill_between(years, median, upper, color='grey', alpha=0.2, label='IDR')
    plt.fill_between(years, median, lower, color='grey', alpha=0.2)
    # legend
    plt.legend(loc='upper left', fontsize=fontsize)
    # ticks 
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    # y limit
    # plt.ylim((-0.05, 1.35))
    # save figure
    plt.savefig(f'{figure_path}\\esl_{suffix}.png', bbox_inches='tight')

    print(f"- Created plot of ESL/GDP for scenario {suffix}:")
    print(f"  => {figure_path}")

    ### Close Database Connection ###
    
    con.close()
