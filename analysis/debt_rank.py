import os
import sys
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# add parent directory to system path
sys.path.append('..')
from utils_analysis import load_parameters, bank_debtrank, expected_systemic_loss

### matplotlib settings ###

# change matplotlib font to serif
plt.rcParams['font.family'] = ['serif']
# figure size
x_figsize = 10
y_figsize = x_figsize/2
# fontsize
fontsize = 25

### paths to folders ###

analysis_path = os.getcwd()
parent_path = os.path.abspath(os.path.join(analysis_path, os.pardir))
figure_path = f"{analysis_path}\\figures\\batch"
table_path = f"{analysis_path}\\tables"

### load model parameters ###

# parameters
params = load_parameters(f'{parent_path}\\src\\parameters.yaml')
# analysis parameters
steps = params['simulation']['steps']
start = params['simulation']['start']*steps
end = (params['simulation']['start'] + params['simulation']['years'])*steps
middle = int(end/2)
years = np.linspace(0, params['simulation']['years'], params['simulation']['years']*steps)

### paths to data ###
data_path = params['database_path']
databases = os.listdir(data_path)

### connect to sql database ###

# database index
for index in range(len(databases)):
    # get database name
    database_name = databases[index]
    # suffix for saved file name
    suffix = database_name[:-3]
    # create connection to database
    con = sqlite3.connect(f"{data_path}\\{database_name}")
    # create database cursor
    cur = con.cursor()

    # get edges data
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

    # get edges data
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

    # get edges data
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
    edges = edges.merge(prob_default, left_on=["simulation", "source"], right_on=["simulation", "id"], how='left')
    edges.drop(columns=["id"], inplace=True)

    # delete banks df (save memory)
    del banks

    ### start debtrank calculation ###

    # number of simulations 
    num_sims = len(edges['simulation'].unique())
    # number of periods per simulation
    num_periods = len(years)
    # array to hold expected systemic loss as a percentage of nominal GDP results
    esl_gdp = np.zeros(shape=(num_sims,num_periods))
    # simulation loop
    for s in range(1):
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
    
    ### plot results ###
    
    # median esl over simulations
    median = np.quantile(esl_gdp, q=0.5, axis=0)
    # Top 9th decile
    upper = np.quantile(esl_gdp, q=0.9, axis=0)
    # Top 1st decile
    lower = np.quantile(esl_gdp, q=0.1, axis=0)

    plt.figure(figsize=(x_figsize,y_figsize))
    plt.plot(years, median, color='k', linewidth=1, label='Mean')
    plt.fill_between(years, median, upper, color='grey', alpha=0.3, label='IDR')
    plt.fill_between(years, median, lower, color='grey', alpha=0.3)
    # legend
    plt.legend(loc='upper left', fontsize=fontsize)
    # ticks 
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    # save figure
    plt.savefig(f'{figure_path}\\esl_{suffix}.png', bbox_inches='tight')
    plt.show()