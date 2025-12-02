import yaml
import sqlite3
import numpy as np
import pandas as pd
import powerlaw as pl
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from numpy.typing import NDArray
import statsmodels.api as sm

def load_yaml(file: Path | str) -> dict:
    """
    Load YAML file as dictionary.
    
    Parameters
    ----------
        filename : str
            name of YAML file to load
    
    Returns
    -------
        file_dict : dict
            YAML file loaded as dictionary 
    """
    with open(file, 'r') as f:
        file_dict = yaml.safe_load(f)
    return file_dict

def load_macro_data(path: Path | str, params: dict, steps: int, start: int):
    '''Load macro_data table from the SQL database into a pandas DataFrame and calculate variables.'''
    # create connection to database
    con = sqlite3.connect(path)
    # load table into pandas DataFrame
    data = pd.read_sql_query(f"SELECT * FROM macro_data", con)
    ### define new time series ###
    # year
    data['year'] = (data['time'] // steps).astype(int)
    # avg hpi
    data['avg_hpi'] = (data['cfirm_distresspi'] + data['kfirm_distresspi'] + data['bank_distresspi'])/3
    # avg nhhi
    data['avg_nhhi'] = (data['cfirm_nhhi'] + data['kfirm_nhhi'] + data['bank_nhhi'])/3
    # real gdp growth 
    data['rgdp_growth'] = np.log(data['real_gdp']) - np.log(data['real_gdp'].shift(steps))
    # inflation
    data['inflation'] = np.log(data['cfirm_price_index']) - np.log(data['cfirm_price_index'].shift(steps))
    # wage inflation
    data['wage_inflation'] = np.log(data['avg_wage']) - np.log(data['avg_wage'].shift(steps))
    # debt ratio
    data['debt_ratio'] = data['debt']/data['nominal_gdp']
    # wage share
    data['wage_share'] = data['wages']/data['nominal_gdp']
    # profit share
    data['profit_share'] = data['profits']/data['nominal_gdp']
    # normalised productivity to start date
    data['productivity'] = (data['real_gdp']/data['employment'])/(data['real_gdp'][start]/data['employment'][start])
    # productivity growth
    data['productivity_growth'] = np.log(data['productivity']) - np.log(data['productivity'].shift(steps))
    # credit: debt growth rate
    data['credit'] = np.log(data['debt']) - np.log(data['debt'].shift(steps))
    # change unemployment 
    data['change_unemployment'] = data['unemployment_rate'] - data['unemployment_rate'].shift(steps)
    # probability of at least one crisis in a given year
    data['crises'] = (data['rgdp_growth'] < -0.03).astype(int)
    data['yearly_crises'] = data.groupby(['simulation', 'year'])['crises'].transform('max')
    data['crises_prob'] = data.groupby('time')['yearly_crises'].transform('mean')
    
    # quarterly default probability
    data['cfirm_prob'] = data['cfirm_bankruptcy'] / params['market']['num_cfirms']
    data['kfirm_prob'] = data['kfirm_bankruptcy'] / params['market']['num_kfirms']
    data['bank_prob']  = data['bank_bankruptcy']  / params['market']['num_banks']
    
     # average per year default probability, per simulation, split by crisis 
    yearly_probs = (
        data.groupby(['simulation','year','crises'])[['cfirm_prob','kfirm_prob','bank_prob']]
        .mean()
        .reset_index()
    )
    
    # average across years within each simulation, conditional on there being a crisis
    cond_probs = (
        yearly_probs.groupby(['simulation','crises'])[['cfirm_prob','kfirm_prob','bank_prob']]
        .mean()
        .reset_index()
    )

    # Pivot to get separate columns for crisis=1 (true) and crisis=0 (false)
    cond_probs = cond_probs.pivot(index='simulation', columns='crises')
    cond_probs.columns = [f"{var}_crises{c}" for var,c in cond_probs.columns]
    cond_probs = cond_probs.reset_index()

    # merge back into main data so each row has the conditional averages
    data = data.merge(cond_probs, on='simulation', how='left')

    # drop transient data
    data = data.loc[data['time'] > start]
    # reset index
    data.reset_index(inplace=True, drop=True)
    # close connection to database
    con.close()
    return data

def load_micro_data(query: str, path: Path | str) -> pd.DataFrame:
    '''Load a table from the SQL database into a pandas DataFrame.'''
    # create connection to database
    con = sqlite3.connect(path)
    # load table into pandas DataFrame
    data = pd.read_sql_query(query, con)
    # close database connection
    con.close()
    return data

def box_plot_scenarios(
    plot_data: dict[str,pd.DataFrame], 
    variable: str,
    figsize: tuple[float,float] | None = None,
    fontsize: int | None = None,
    xlabels: list[str] | None = None, 
    xticks: list[int] | None = None, 
    yticks: list[float] | None = None,
    ylim: tuple[float, float] | None = None,
    colours: list[str] | None = None,
    figure_path: Path | str | None = None,
    dp: int = 3
    ):
    
    ### create plot data ###
    # copy plot data
    plot_data = plot_data.copy()
    # get all values across scenarios for variable
    for scenario in plot_data.keys():
        plot_data[scenario] = plot_data[scenario][variable].to_numpy().ravel()
    
    ### box plot ###
    plt.figure(figsize=figsize)
    bplot = plt.boxplot(plot_data.values(), patch_artist=True, showfliers=False, medianprops=dict(color='black', linewidth=2), whis=(2.5, 97.5))
    
    ### set box colour ###
    if colours:
        for patch, color in zip(bplot['boxes'], colours):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
            
    ### figure settings ###
    # decimal places 
    if yticks is not None:
        plt.yticks(yticks, [f"{y:.{dp}f}" for y in yticks], fontsize=fontsize)
    else:
        plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter(f'%.{dp}f'))
    # y axis ticks
    plt.yticks(yticks, fontsize=fontsize)
    # y axis limits
    plt.ylim(ylim)
    # x axis ticks
    plt.xticks(xticks, xlabels, fontsize=fontsize)
    # save figure
    plt.savefig(figure_path / f"box_plot_{variable}", bbox_inches='tight')
    
def large_ages(data: pd.DataFrame, q: float) -> pd.Series:
    # calculate market share quantile 
    ms_q = data['market_share'].quantile(q)
    
    # Get all market shares 
    ages = data['age'][data['market_share'] >= ms_q]
    # return age diffs
    return ages, ms_q

def small_ages(data: pd.DataFrame, q: float) -> pd.Series:
    # calculate market share quantile 
    ms_q = data['market_share'].quantile(q)
    # Get all market shares 
    ages = data['age'][data['market_share'] <= ms_q]
    # return age diffs
    return ages, ms_q

def calculate_bank_age(bank_data: pd.DataFrame):
    data = bank_data.copy()
    # bankruptcy conditon
    data['bankruptcy'] = data['min_capital_ratio']*(data['loans'] + data['reserves']) == data['equity']
    # reset 
    data["reset"] = (data.groupby(["simulation", "id"])["bankruptcy"].cumsum())
    # Within each (simulation, id, reset) segment, count observations since last bankruptcy
    data["age_quarters"] = (data.groupby(["simulation", "id", "reset"]).cumcount() + 1)
    # Convert quarters to years (each observation = 0.25 years)
    data["age"] = data["age_quarters"] * 0.25
    # return results 
    return data['age']

def logscale_ticks(low: float, high: float, num: int) -> np.ndarray:
    """
    Get ticks for either x or y axis from data, rounded to first digit spaced by num in log 10.
    
    Parameters
    ----------
        series : pd.Series
            time series
            
        num : int 
            number of ticks to return
    
    Returns
    -------
        ticks : numpy array
            axis ticks spaced by num in log 10
    """
    log_arr = np.logspace(np.log10(low),np.log10(high), num)
    lengths = np.vectorize(len)(np.char.mod('%d', log_arr))
    factor = 10 ** (lengths - 1)
    round_log_arr = np.int64(np.round(log_arr.astype(int) / factor) * factor)
    return round_log_arr

def plot_autocorrelation(simulated: np.typing.ArrayLike, empirical: np.typing.ArrayLike, feature: str, figsize: tuple[int,int], fontsize: int, lags: int, lamda: int, savefig: str):
    """
    Plots the autocorrelation function (ACF) of the simulated time-series with a 95% confidence interval (CI), calculated over all simulations,
    and also plots the empirical time-series ACF for a given feature vector of the two time-series for a given number of lags.
    
    Parameters
    ----------
        simulated : pandas.DataFrame
            model simulated time-series with s simulations
        
        empirical : pandas.DataFrame
            single empirical time-series
        
        feature : str
            name of column in both simulated and empirical DataFrame
        
        figsize : tuple (int, int) 
            size of figure (x-axis, y-axis)
            
        fontsize : int
            fontsize of legend and ticks
        
        lags: int
            number of autocorrelation lags
        
        lamda : int
            Hodrick-Prescott filter lambda parameter (quarterly => lambda=1600)
            
        savefig : str
            path and figure name 
    """
    # array to store autocorrelation for each simulation
    sim_autocorr = np.zeros(shape=(max(simulated['simulation']) + 1, lags + 1))
    # calculate autocorrelation for each simulation
    for s in simulated['simulation'].unique():
        temp_simulated = simulated.loc[simulated['simulation'] == s]
        # decompose trend and cycle component using Hodrick-Prescott filter
        sim_cycle, sim_trend = sm.tsa.filters.hpfilter(temp_simulated[feature], lamda)
        # autocorrelation 
        sim_autocorr[s,:] = sm.tsa.acf(sim_cycle, nlags=lags)
    # median autocorrelation
    sim_autocorr_median = np.quantile(sim_autocorr, 0.5, axis=0)
    # upper 5% quantile
    sim_autocorr_upper = np.quantile(sim_autocorr, 0.95, axis=0)
    # lower 5% quantile
    sim_autocorr_lower = np.quantile(sim_autocorr, 0.05, axis=0)
    # decompose trend and cycle component of empirical time-series using Hodrick-Prescott filter
    emp_cycle, emp_trend = sm.tsa.filters.hpfilter(empirical[feature], lamda)
    # empirical autocorrelation
    emp_autocorr = sm.tsa.acf(emp_cycle, nlags=lags)
    # plot results
    plt.figure(figsize=figsize)
    # x values (lags)
    x = np.arange(0, lags+1)
    # plot empirical autocorrelation 
    plt.plot(emp_autocorr, color='k', linestyle='--', linewidth=1, label='Empirical')
    # plot simulated autocorrelation median
    plt.plot(sim_autocorr_median, color='k', linewidth=1, label='Simulated')
    # plot confidence interval
    plt.fill_between(x, sim_autocorr_median, sim_autocorr_upper, color='grey', alpha=0.2, label='95% CI')
    plt.fill_between(x, sim_autocorr_median, sim_autocorr_lower, color='grey', alpha=0.2)
    # plot 0 line
    plt.axhline(0, color='k')
    # figure ticks
    plt.yticks([1.00,0.75,0.50,0.25,0.00,-0.25,-0.50,-0.75,-1.00], ['1.00','0.75','0.50','0.25','0.00','–0.25','–0.50','–0.75','–1.00'], fontsize=fontsize)
    plt.xticks([0,5,10,15,20], [0,5,10,15,20], fontsize=fontsize)
    # legend
    plt.legend(fontsize=fontsize, loc='upper right')
    # save figure
    plt.savefig(savefig, bbox_inches='tight')
    # show figure
    plt.show()

def plot_cross_correlation(simulated: np.typing.ArrayLike, empirical: np.typing.ArrayLike, xfeature: str, yfeature: str, figsize: tuple[int,int], fontsize: int, lags: int, lamda: int, savefig: str):
    """
    Plots the cross correlation (xcorr) of the simulated time-series for feature xfeature and yfeature with a 95% confidence interval (CI), calculated over all simulations,
    and also plots the empirical time-series xcorr for a given xfeature and yfeature vector, each for a given number of lags.
    
    Parameters
    ----------
        simulated : pandas.DataFrame
            model simulated time-series with s simulations
        
        empirical : pandas.DataFrame
            single empirical time-series
        
        xfeature : str
            name of the x feature column in both simulated and empirical DataFrames
        
        yfeature : str
            name of the y feature column in both simulated and empirical DataFrames
        
        figsize : tuple (int, int) 
            size of figure (x-axis, y-axis)
            
        fontsize : int
            fontsize of legend and ticks
        
        lags: int
            number of correlation lags, in range (-lags, lags)
        
        lamda : int
            Hodrick-Prescott filter lambda parameter (quarterly => lambda=1600)
            
        savefig : str
            path and figure name 
    """
    # array to store autocorrelation for each simulation
    sim_xcorr = np.zeros(shape=(max(simulated['simulation']) + 1, lags*2 + 1))
    # calculate autocorrelation for each simulation
    for s in simulated['simulation'].unique():
        temp_simulated = simulated.loc[simulated['simulation'] == s]
        # decompose trend and cycle component using Hodrick-Prescott filter
        sim_xcycle, sim_xtrend = sm.tsa.filters.hpfilter(temp_simulated[xfeature], lamda)
        sim_ycycle, sim_ytrend = sm.tsa.filters.hpfilter(temp_simulated[yfeature], lamda)
        # autocorrelation
        sim_xcorr[s,:] = plt.xcorr(sim_xcycle, sim_ycycle, maxlags=lags)[1]
    # median autocorrelation
    sim_xcorr_median = np.quantile(sim_xcorr, 0.5, axis=0)
    # upper 5% quantile
    sim_xcorr_upper = np.quantile(sim_xcorr, 0.95, axis=0)
    # lower 5% quantile
    sim_xcorr_lower = np.quantile(sim_xcorr, 0.05, axis=0)
    # decompose trend and cycle component of empirical time-series using Hodrick-Prescott filter
    emp_xcycle, emp_xtrend = sm.tsa.filters.hpfilter(empirical[xfeature], lamda)
    emp_ycycle, emp_ytrend = sm.tsa.filters.hpfilter(empirical[yfeature], lamda)
    # empirical autocorrelation
    emp_xcorr = plt.xcorr(emp_xcycle, emp_ycycle, maxlags=lags)[1]
    # clear figure
    plt.clf()
    # start figure
    plt.figure(figsize=figsize)
    # x values (lags)
    x = np.arange(0, lags*2 + 1)
    # plot empirical xcorr
    plt.plot(emp_xcorr, color='k', linestyle='--', linewidth=1, label='Empirical')
    # plot median simulated xcorr
    plt.plot(sim_xcorr_median, color='k', linewidth=1, label='Simulated')
    # plot confidence interval
    plt.fill_between(x, sim_xcorr_median, sim_xcorr_upper, color='grey', alpha=0.2, label='95% CI')
    plt.fill_between(x, sim_xcorr_median, sim_xcorr_lower, color='grey', alpha=0.2)
    # plot 0 line
    plt.axhline(0, color='k')
    # figure ticks
    plt.yticks([1.00,0.75,0.50,0.25,0.00,-0.25,-0.50,-0.75,-1.00], ['1.00','0.75','0.50','0.25','0.00','–0.25','–0.50','–0.75','–1.00'], fontsize=fontsize)
    plt.xticks([0,int(0.5*lags),int(lags),int(1.5*lags),int(2*lags)], [f'–{lags}',f'–{int(0.5*lags)}',0,int(0.5*lags),lags], fontsize=fontsize)
    # legend
    plt.legend(fontsize=fontsize, loc='upper right')
    # save figure
    plt.savefig(savefig, bbox_inches='tight')
    # show figure
    plt.show()

def plot_ccdf(data: np.typing.ArrayLike, figsize: tuple[int,int], fontsize: int, ylim: tuple[float,float], savefig: str, dp: int=0) -> None:
    """
    Plots the complementary cumulative distribution function (CCDF) for a given series 
    and prints the power law exponent, cut-off value, and compares the distribution to a lognormal.
    
    Parameters
    ----------
        data : pd.Series or numpy array
            series to plot CCDF
            
        figsize : tuple (int, int) 
            size of figure (x-axis, y-axis)
            
        fontsize : int
            fontsize of legend and ticks
            
        savefig : str
            path and figure name 
        dp : int 
            power law fit x minimum number decimal points
    """
    # power law fit results 
    results = pl.Fit(data)
    a, m = results.alpha, results.xmin
    # complementary cdf 
    plt.figure(figsize=figsize)
    # x values 
    x = np.sort(data)
    # complementary cdf (ccdf)
    cdf = np.arange(1,len(data)+1)/(len(data))
    ccdf = 1 - cdf
    plt.scatter(x, ccdf, color='skyblue', edgecolors='k', alpha=0.7, s=30, label='CCDF')
    # power law fit
    # => rescale to start fit from cut off
    index = np.where(x == m)[0][0]
    rescale = ccdf[index]
    power_law_fit = np.where(x >= m, np.power((m)/x,a-1)*rescale, np.nan)
    plt.plot(x, power_law_fit, color='limegreen', linewidth=3, label=r'PL ($\alpha$'f' = {round(a, 2)})')
    # power law cut off (mF)
    plt.axvline(m, color='k', linestyle=':', label=r'$m$'f' = {round(m, dp)}') # type: ignore
    # lognormal distribution
    estimates = stats.lognorm.fit(data)
    cdf = stats.lognorm.cdf(x, estimates[0], estimates[1], estimates[2])
    plt.plot(x, 1 - cdf, color='r', linestyle='--', linewidth=2, label='Log-Normal')
    # log-log axis
    plt.loglog()
    # legend
    plt.legend(loc='lower left', fontsize=fontsize)
    # tick size
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    # y limit 
    plt.ylim(ylim)
    # save figure
    plt.savefig(savefig, bbox_inches='tight')
    print(f'Power law exponent = {a}')
    print(f'Power law minimum = {m}')
    print(f"Distribution compare = {results.distribution_compare('power_law', 'lognormal')}\n")
    
def bank_debtrank(
        W_banks: NDArray[np.float64],        # shape (num_banks, num_firms)
        W_firms: NDArray[np.float64],        # shape (num_firms, num_banks)
        bank_assets: NDArray[np.float64],    # shape (num_banks,)
        firm_assets: NDArray[np.float64],    # shape (num_firms,)
        num_banks: int,
        num_firms: int,
        max_iterations: int = 500,
        epsilon: float = 1e-8
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Description
    -----------
    Function to calculate DebtRank if each bank when bankrupt for a bank-firm bipartitie credit network (Battiston et al, 2012; Aoyama et al, 2013).
    
    References
    ----------
    
    Battiston, S., Puliga, M., Kaushik, R., Tasca, P., & Caldarelli, G. (2012). Debtrank: Too central to fail? 
    financial networks, the fed and systemic risk. Scientific reports, 2(1), 541.
        
    Aoyama, H., Battiston, S., & Fujiwara, Y. (2013). DebtRank analysis of the Japanese credit network. 
    Discussion papers, Research Institute of Economy, Trade and Industry (RIETI).
    
    Parameters
    ----------
        W_banks : NDArray[float64]
            bank propagation matrix

        W_firms: NDArray[float64]
            firm propagation matrix
        
        bank_assets: NDArray[float64]
            bank asset values
        
        firm_assets: NDArray[float64]
            firm asset values
        
        max_iterations: int
            maximum number of iterations per loop
        
        epsilon: float
            convergence tolerance
            
    Returns
    -------
        (debtrank_bank, debtrank_firm) : tuple(NDArray[Float64], NDArray[float64])
            a tuple of bank DebtRanks and firm DebtRanks when each bank becomes bankrupt
    """
    debtrank_bank = np.zeros(num_banks, dtype=np.float64)
    debtrank_firm = np.zeros(num_banks, dtype=np.float64)

    for i in range(num_banks):
        # initialise distress and state variables
        bank_distress = np.zeros(num_banks, dtype=np.float64)
        firm_distress = np.zeros(num_firms, dtype=np.float64)
        bank_state = np.zeros(num_banks, dtype=np.int8)
        firm_state = np.zeros(num_firms, dtype=np.int8)

        # initial shock: bank i distressed (bankrupt)
        bank_distress[i] = 1.0
        bank_state[i] = 1

        for _ in range(max_iterations):
            # record previous distress for convergence check
            prev_total = np.sum(bank_distress) + np.sum(firm_distress)

            ### update distress ###
            # calculate new firm distress from currently distressed banks
            distressed_banks = (bank_state == 1).astype(np.float64)
            firm_distress_new = np.minimum(1.0, firm_distress + W_firms@(distressed_banks*bank_distress))

            # calculate new bank distress from currently distressed firms
            distressed_firms = (firm_state == 1).astype(np.float64)
            bank_distress_new = np.minimum(1.0, bank_distress + W_banks@(distressed_firms*firm_distress))

            ### update states ###
            # undistressed firms to distressed (U -> D)
            firm_state[(firm_distress_new > 0) & (firm_state == 0)] = 1
            # distressed firms to inactive (D -> I)
            firm_state[firm_state == 1] = 2
            # update firm distress
            firm_distress = firm_distress_new

            # undistressed banks to distressed (U -> D)
            bank_state[(bank_distress_new > 0) & (bank_state == 0)] = 1
            # distressed banks to inactive (D -> I)
            bank_state[bank_state == 1] = 2
            # update bank distress
            bank_distress = bank_distress_new

            ### check convergence ###
            total = np.sum(bank_distress) + np.sum(firm_distress)
            if abs(total - prev_total) < epsilon:
                break
        
        ### calculate DebtRank ###
        # bank-layer debtrank for bank i: exclude initially distressed bank i
        bank_mask = np.arange(num_banks) != i
        debtrank_bank[i] = np.sum(bank_distress[bank_mask]*bank_assets[bank_mask])/np.sum(bank_assets[bank_mask])
        # firm-layer debtrank for bank i: include all firms
        debtrank_firm[i] = np.sum(firm_distress*firm_assets)/np.sum(firm_assets)

    return debtrank_bank, debtrank_firm

def expected_systemic_loss(
        debtrank_bank: np.typing.NDArray[np.float64],
        debtrank_firm: np.typing.NDArray[np.float64],
        prob_default: np.typing.NDArray[np.float64],
        bank_assets: np.typing.NDArray[np.float64],
        firm_assets: np.typing.NDArray[np.float64]
    ) -> float:
    """
    Description
    -----------
    Function to calculate the expected systemic loss (ESL) approximation from Polenga et al (2015) 
    using DebtRank for a bipartite bank-firm credit network.
    
    References
    ----------
    Poledna, S., Molina-Borboa, J. L., Martínez-Jaramillo, S., Van Der Leij, M., & Thurner, S. (2015). 
    The multi-layer network nature of systemic risk and its implications for the costs of financial crises. 
    Journal of Financial Stability, 20, 70-81.
    
    Parameters
    ----------
        debtrank_bank : NDArray[float64]
            bank DebtRank 
            
        debtrank_firm : NDArray[float64]
            firm DebtRank
            
        prob_default : NDArray[float64]
            probability of default for each bank
            
        bank_assets: NDArray[float64]
            bank asset values
        
        firm_assets: NDArray[float64]
            firm asset values
            
    Returns
    -------
        esl : float
            Expected systemic loss approximation
    """
    # total assets
    total_bank_assets = np.sum(bank_assets)
    total_firm_assets = np.sum(firm_assets)

    # expected systemic loss approximation (ESL)
    esl = np.sum(prob_default*(debtrank_bank*total_bank_assets + debtrank_firm*total_firm_assets))

    # return esl calculation
    return esl