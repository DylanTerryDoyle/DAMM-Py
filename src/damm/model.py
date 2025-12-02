### Import Libraries ###
import copy
import sqlite3
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

### Import Agents ###
from damm.agents.bank import Bank
from damm.agents.household import Household
from damm.agents.capital_firm import CapitalFirm
from damm.agents.consumption_firm import ConsumptionFirm

### Model Class ###
class Model:
    
    """
    Market class
    ============
    Runs the macro ABM model. 
    
    Attributes
    ----------
        scenario : str
            name of scenario 
        
        params : dict
            model parameters
        
        simulations : int
            number of simulation
        
        steps : int
            number of time steps per year
        
        time : int
            total number of time steps 
        
        start : int
            simulation start cut-off
        
        dt : float
            time delta, inverse steps
        
        growth : float
            average growth rate
        
        num_households : int
            number of households in simulation
        
        num_banks : int
            number of banks in simulation
        
        num_cfirms : int
            number of consumption firms in simulation
        
        num_kfirms : int
            number of capital firm in simulation
        
        num_firms : int
            total number of firms in simulation
        
        cfirm_id : int
            current max cfirm id 
        
        kfirm_id : int
            current max kfirm id
        
        length : int
            length of banks dataset
        
        initial_output : float
            initial real gdp
    
    Data
    ----
        cfirm_data : pandas.DataFrame
            C-firm bankruptcy data
        
        kfirm_data : pandas.DataFrame
            K-firm bankrupcty data
        
        consumption : NDArray
            real consumption 
        
        nominal_consumption : NDArray
            nominal consumption
        
        investment : NDArray
            real investment
        
        nominal_investment : NDArray
            nominal investment
        
        real_gdp : NDArray
            real gross domestic product (GDP)
        
        nominal_gdp : NDArray
            nominal gross domestic product (GDP)
        
        capital : NDArray
            total capital stock
        
        cfirm_productivity : NDArray
            average C-firm productivity 
        
        kfirm_productivity : NDArray
            average K-firm productivity 
        
        debt : NDArray
            total nominal debt
        
        profits : NDArray
            total nominal profits
        
        cfirm_price_index : NDArray
            consumption good Paasche price index
        
        kfirm_price_index : NDArray
            capital good Paasche price index
        
        cfirm_nhhi : NDArray
            C-firm normalised Herfindahl-Hirschman Index
        
        kfirm_nhhi : NDArray
            K-firm normalised Herfindahl-Hirschman Index
        
        cfirm_hpi : NDArray
            C-firm Hymer-Pashigian Instability Index
        
        kfirm_hpi : NDArray
            K-firm Hymer-Pashigian Instability Index
        
        cfirm_bankruptcy : NDArray
            total C-firm bankruptcies per period
        
        kfirm_bankruptcy : NDArray
            total K-firm bankruptcies per period
        
        wages : NDArray
            total household wages
        
        avg_wage : NDArray
            average household wages
        
        employment : NDArray
            total number of employed households
        
        unemployment_rate : NDArray
            unemployment rate
        
        vacancy_ratio : NDArray
            ratio of vacancies to total households
        
        gini : NDArray
            Gini Coefficient
        
        bank_nhhi : NDArray
            bank normalised Herfindahl-Hirschman Index
        
        bank_hpi : NDArray
            bank Hymer-Pashigian Instability Index
        
        avg_loan_interest : NDArray
            average bank loan interest rate
        
        avg_reserve_ratio : NDArray
            average bank reserve ratio
        
        avg_capital_ratio : NDArray
            average bank capital ratio
        
        money_supply : numpy.array
            total M1 money supply
        
        bank_bankruptcy : NDArray
            number of bank bankruptcies per period
        
        bank_mean_degree : NDArray
            average bank to firm degree 
        
        cfirm_mean_degree : NDArray
            average C-firm to bank degree 
        
        kfirm_mean_degree : NDArray
            average K-firm to bank degree
    
    Methods
    -------
        __init__(self, params: dict) -> None
        
        initialise_database(self, cur: sql.Cursor) -> None
        
        instatiate_agents(self) -> None
        
        initialise_simulation(self) -> None
        
        new_entrants(self, s: int, t: int) -> None
        
        labour_market(self, s: int, t: int) -> None
        
        production(self, s: int, t: int) -> None
        
        consumption_market(self, t: int) -> None
        
        capital_market(self, t: int) -> None
        
        probability_default(self, t: int) -> None
        
        credit_market(self, s: int, t: int) -> None
        
        bankrupcty_data(self, t: int) -> None
        
        bankruptcies(self, s: int, t: int) -> None
        
        market_shares(self, t: int) -> None
                
        copy_cfirm(self, cfirm: ConsumptionFirm) -> ConsumptionFirm
        
        copy_kfirm(self, kfirm: CapitalFirm) -> CapitalFirm
        
        compute_gini(self, t: int) -> float
        
        compute_cfirm_probabilities(self, t: int) -> list[float]
        
        compute_kfirm_probabilities(self, t: int) -> list[float]
        
        compute_labour_probabilities(self, t: int) -> list[float]
        
        compute_loan_probabilities(self, t: int) -> list[float]
        
        indicators(self, s: int, t: int) -> None
        
        update_database(self, cur: sql.Cursor, s: int, t: int) -> None
        
        print_results(self, s: int, t: int) -> None
        
        run_simulation(self, cur: sql.Cursor) -> None
    """

    def __init__(self, scenario: str, params: dict) -> None:
        """
        Market class initialisation.
        
        Parameters
        ----------
            params : dict
                model parameters
        """
        ### Scenario Name ### 
        self.scenario:              str = scenario
        ### Parameters ###
        self.simulations:           int   = params['simulation']['num_sims']
        self.steps:                 int   = params['simulation']['steps']
        self.time:                  int   = (params['simulation']['years'] + params['simulation']['start'])*self.steps + 1
        self.start:                 int   = params['simulation']['start']*self.steps
        self.dt:                    float = 1/self.steps
        self.growth:                float = params['firm']['growth']*self.dt
        self.num_households:        int   = params['market']['num_households']
        self.num_banks:             int   = params['market']['num_banks']
        self.num_cfirms:            int   = params['market']['num_cfirms']
        self.num_kfirms:            int   = params['market']['num_kfirms']
        self.num_firms:             int   = self.num_cfirms + self.num_kfirms
        self.cfirm_id:              int   = params['market']['num_kfirms']
        self.kfirm_id:              int   = params['market']['num_cfirms']
        self.length:                int   = params['market']['length']
        self.initial_output:        float = np.floor(self.num_households/self.num_firms)
        ### Probability of default data ### 
        self.cfirm_data:            pd.DataFrame = pd.DataFrame({'default': {}, 'leverage': {}})
        self.kfirm_data:            pd.DataFrame = pd.DataFrame({'default': {}, 'leverage': {}})
        ### Market Data ##
        self.consumption:           NDArray = np.zeros(shape=(self.simulations, self.time))
        self.nominal_consumption:   NDArray = np.zeros(shape=(self.simulations, self.time))
        self.investment:            NDArray = np.zeros(shape=(self.simulations, self.time))
        self.nominal_investment:    NDArray = np.zeros(shape=(self.simulations, self.time))
        self.real_gdp:              NDArray = np.zeros(shape=(self.simulations, self.time))
        self.nominal_gdp:           NDArray = np.zeros(shape=(self.simulations, self.time))
        self.capital:               NDArray = np.zeros(shape=(self.simulations, self.time))
        self.cfirm_productivity:    NDArray = np.zeros(shape=(self.simulations, self.time))
        self.kfirm_productivity:    NDArray = np.zeros(shape=(self.simulations, self.time))
        ### Firm Data ###  
        self.debt:                  NDArray = np.zeros(shape=(self.simulations, self.time))
        self.profits:               NDArray = np.zeros(shape=(self.simulations, self.time))
        self.cfirm_price_index:     NDArray = np.zeros(shape=(self.simulations, self.time))
        self.kfirm_price_index:     NDArray = np.zeros(shape=(self.simulations, self.time))
        self.cfirm_nhhi:            NDArray = np.zeros(shape=(self.simulations, self.time))
        self.kfirm_nhhi:            NDArray = np.zeros(shape=(self.simulations, self.time))
        self.cfirm_hpi:             NDArray = np.zeros(shape=(self.simulations, self.time))
        self.kfirm_hpi:             NDArray = np.zeros(shape=(self.simulations, self.time))
        self.cfirm_bankruptcy:      NDArray = np.zeros(shape=(self.simulations, self.time))
        self.kfirm_bankruptcy:      NDArray = np.zeros(shape=(self.simulations, self.time))
        ### Household Data ###  
        self.wages:                 NDArray = np.zeros(shape=(self.simulations, self.time))
        self.avg_wage:              NDArray = np.zeros(shape=(self.simulations, self.time))
        self.employment:            NDArray = np.zeros(shape=(self.simulations, self.time))
        self.unemployment_rate:     NDArray = np.zeros(shape=(self.simulations, self.time))
        self.vacancy_ratio:         NDArray = np.zeros(shape=(self.simulations, self.time))
        self.gini:                  NDArray = np.zeros(shape=(self.simulations, self.time))
        ### Bank Data ###  
        self.bank_nhhi:             NDArray = np.zeros(shape=(self.simulations, self.time))
        self.bank_hpi:              NDArray = np.zeros(shape=(self.simulations, self.time))
        self.avg_loan_interest:     NDArray = np.zeros(shape=(self.simulations, self.time))
        self.avg_reserve_ratio:     NDArray = np.zeros(shape=(self.simulations, self.time))
        self.avg_capital_ratio:     NDArray = np.zeros(shape=(self.simulations, self.time))
        self.money_supply:          NDArray = np.zeros(shape=(self.simulations, self.time))
        self.bank_bankruptcy:       NDArray = np.zeros(shape=(self.simulations, self.time))
        ### Network Data ###  
        self.bank_mean_degree:      NDArray = np.zeros(shape=(self.simulations, self.time))
        self.cfirm_mean_degree:     NDArray = np.zeros(shape=(self.simulations, self.time))
        self.kfirm_mean_degree:     NDArray = np.zeros(shape=(self.simulations, self.time))
        ### Initial values ### 
        self.real_gdp[:,:]          = self.initial_output*self.num_firms
        self.nominal_gdp[:,0]       = self.real_gdp[:,0]
        self.consumption[:,0]       = self.initial_output*self.num_cfirms
        self.investment[:,0]        = self.initial_output*self.num_kfirms
        self.cfirm_price_index[:,:] = params['firm']['price']
        self.kfirm_price_index[:,:] = params['firm']['price']
        
        
    def initialise_database(self) -> None:
        """
        Iniatilise database and create tables to store macro and micro data.
        """
        ### Initialise Database ###
        # current working directory
        cwd_path = Path.cwd()
        # database path 
        data_path = cwd_path / "data"
        # create data folder if doesn't exist
        data_path.mkdir(parents=True, exist_ok=True)
        # create database connection
        self.connection = sqlite3.connect(data_path / f"{self.scenario}.db")
        # create database cursor
        self.cursor = self.connection.cursor()
        
        ### Initialise Tables ###
        """
        Iniitalise SQLite3 database tables:
        
        Parameters
        ----------
            cur : sqlite3.Cursor
                sqlite3 cursor object
        """
        
        ### 1. Macro Table ###
        # drop table if it already exists in the database 
        self.cursor.execute(
            """
                DROP TABLE IF EXISTS macro_data;
            """
        )
        # get all attributes of type np.ndarray
        macro_array_attrs = [name for name, value in self.__dict__.items() if isinstance(value, np.ndarray)]
        # create column names
        macro_columns = ",\n    ".join(f"{name} REAL NOT NULL" for name in macro_array_attrs)
        # create sql statement
        macro_sql = f"""
            CREATE TABLE macro_data (
                simulation INT NOT NULL,
                time INT NOT NULL,
                {macro_columns}
            );
        """
        # execute sql statement
        self.cursor.execute(macro_sql)
        
        ### 2. Firm Table ### 
        # drop table if it already exists in the database 
        self.cursor.execute(
            f"""
                DROP TABLE IF EXISTS firm_data;
            """
        )
        # temp instances of both firm types
        temp_cfirm = ConsumptionFirm(id=0, initial_output=10, initial_wage=1, params=self.params)
        temp_kfirm = CapitalFirm(id=0, initial_output=10, initial_wage=1, params=self.params)
        # get ndarray attributes from both firm types
        cfirm_attrs = {name for name, value in temp_cfirm.__dict__.items() if isinstance(value, np.ndarray)}
        kfirm_attrs = {name for name, value in temp_kfirm.__dict__.items() if isinstance(value, np.ndarray)}
        # take union so both types fit the schema
        all_firm_attrs = sorted(cfirm_attrs.union(kfirm_attrs))
        # create firm columns names
        firm_columns = ",\n    ".join(f"{name} REAL" for name in all_firm_attrs)
        # create sql statement
        firm_sql = f"""
            CREATE TABLE firm_data (
                simulation INT NOT NULL,
                time INT NOT NULL,
                id INT NOT NULL,
                firm_type TEXT NOT NULL,
                {firm_columns}
            );
        """
        # execute sql statement
        self.cursor.execute(firm_sql)
        
        ### 3. Household Table ###
        # drop table if it already exists in the database 
        self.cursor.execute(
            f"""
                DROP TABLE IF EXISTS household_data;
            """
        )
        # temporary household
        temp_household = Household(id=0, init_wage=temp_cfirm.wage[0], params=self.params)
        # get ndarray attributes of household class
        household_array_attrs = [name for name, value in temp_household.__dict__.items() if isinstance(value, np.ndarray)]
        # create household column names
        household_columns = ",\n    ".join(f"{name} REAL NOT NULL" for name in household_array_attrs)
        # create sql statement
        household_sql = f"""
            CREATE TABLE household_data (
                simulation INT NOT NULL,
                time INT NOT NULL,
                id INT NOT NULL,
                employed BOOL NOT NULL,
                {household_columns}
            );
        """
        # execute sql statement
        self.cursor.execute(household_sql)
        
        ### 4. Bank Table ###
        # drop table if it already exists in the database 
        self.cursor.execute(
            f"""
                DROP TABLE IF EXISTS household_data;
            """
        )
        # temporary household
        temp_bank = Bank(id=0, params=self.params)
        # get ndarray attributes of household class
        bank_array_attrs = [name for name, value in temp_bank.__dict__.items() if isinstance(value, np.ndarray)]
        # create household column names
        bank_columns = ",\n    ".join(f"{name} REAL NOT NULL" for name in bank_array_attrs)
        # create sql statement
        bank_sql = f"""
            CREATE TABLE bank_data (
                simulation INT NOT NULL,
                time INT NOT NULL,
                id INT NOT NULL,
                {bank_columns}
            );
        """
        # execute sql statement
        self.cursor.execute(bank_sql)
        
        ### 5. Edge Network Table ###
        # drop table if it already exists in the database 
        self.cursor.execute(
            f"""
                DROP TABLE IF EXISTS edge_data;
            """
        )
        # create new table
        self.cursor.execute(
            f"""
                CREATE TABLE edge_data (
                    simulation INT NOT NULL,
                    time INT NOT NULL,
                    source TEXT NOT NULL,
                    target TEXT NOT NULL,
                    loan REAL NOT NULL,
                    firm_assets REAL NOT NULL,
                    bank_assets REAL NOT NULL
                );
            """
        )
    
    def instatiate_agents(self, params: dict) -> None:
        """
        Instantiate agent objects with initial parameters.
        """
        # initial values 
        debt_ratio = (params['cfirm']['d0'] + params['cfirm']['d1']*params['firm']['growth'] + params['cfirm']['d2']*params['cfirm']['acceleration'] * (params['firm']['growth'] + params['firm']['depreciation'])) / (1 + params['cfirm']['d2'] * params['firm']['growth'])
        profit_share = params['cfirm']['acceleration']*(params['firm']['depreciation'] + params['firm']['growth']) - params['firm']['growth']*debt_ratio
        initial_wage = 1 - profit_share - params['bank']['loan_interest']*debt_ratio
        # initialse agent objects into lists 
        self.cfirms:       list[ConsumptionFirm] = [ConsumptionFirm(x, self.initial_output, initial_wage, params) for x in range(self.num_cfirms)]
        self.kfirms:       list[CapitalFirm] = [CapitalFirm(x, self.initial_output, initial_wage, params) for x in range(self.num_kfirms)]
        self.banks:        list[Bank] = [Bank(x, params) for x in range(self.num_banks)]
        self.households:   list[Household] = [Household(x, initial_wage, params) for x in range(self.num_households)] 
        # total firms list
        self.firms:        list[ConsumptionFirm | CapitalFirm] = self.cfirms + self.kfirms
        
    def initialise_simulation(self) -> None:
        """
        Initialise agent data in current simulation:
        
         - Firms randomly choose a loan & deposit bank
         - Firms hire initial number of employees
         - Household randomly choose a deposit bank
         - Banks initialise accounts
         - Initialise avg wage & total debt indices
        """
        # initialse firm banks and employees
        for firm in self.firms:
            firm_bank: Bank = self.banks[np.random.choice(np.arange(self.num_banks))]
            firm_bank.add_deposit_firm(firm)
            firm_bank.add_loan_firm(firm)
            firm.compute_new_loan(firm.loans[0], firm_bank.loan_interest[0], firm_bank.id, 0)
            initial_employees = self.initial_output
            for household in self.households:
                if not household.employed:
                    firm.hire(household)
                    initial_employees -= 1
                    if initial_employees == 0:
                        break
        # initialise household banks
        for household in self.households:
            household_bank: Bank = self.banks[np.random.choice(np.arange(self.num_banks))]
            household_bank.add_household(household)
        # initialise banks customer data
        for bank in self.banks:
            bank.initialise_accounts()
        # intialise average makret wage and total debt
        self.avg_wage[:,:] = self.cfirms[0].wage[0]
        self.debt[:,:] = self.cfirms[0].debt[0]*self.num_cfirms

    def new_entrants(self, s: int, t: int) -> None:
        """
        New Consumption Firms and Capital Firms enter the market at a 1:1 ratio of the previous period bankrupt firms.
        New entrants are a random copy of encombent firms with prices and wages equal to their market averages and no debt.
        
        Parameters
        ----------
            s : int 
                simulation number
            
            t : int
                time period
        """
        # new cfirms 
        if self.cfirm_bankruptcy[s,t-1] > 0:
            # randomly choose firms for new entrants to copy
            centrant_indices: list[np.int32] = list(np.random.choice(np.arange(len(self.cfirms)), size=int(self.cfirm_bankruptcy[s,t-1]))) 
            for i in centrant_indices:
                # cfirm at index i
                cfirm = self.cfirms[i]
                # copy randomly chosen firm
                new_cfirm = self.copy_cfirm(cfirm)
                # create new cfirm id
                new_cfirm.id = self.cfirm_id
                # increase cfirm id
                self.cfirm_id += 1
                # initialise a single employee
                new_cfirm.employees = []
                for household in self.households:
                    if not household.employed:
                        new_cfirm.hire(household)
                        break
                # randomly choose a new bank
                cfirm_bank: Bank = self.banks[np.random.choice(np.arange(self.num_banks))]
                cfirm_bank.add_deposit_firm(new_cfirm)
                cfirm_bank.add_loan_firm(new_cfirm)
                # update entrant data
                new_cfirm.price[t-1] = self.cfirm_price_index[s,t-1]
                new_cfirm.wage[t-1] = self.avg_wage[s,t-1]
                new_cfirm.loans[:] = 0
                new_cfirm.repayment[:] = 0
                new_cfirm.interest[:] = 0 
                new_cfirm.bank_ids[:] = np.nan
                new_cfirm.debt[:] = 0
                new_cfirm.age[:] = 0
                self.cfirms.append(new_cfirm) 
        # new kfirms
        if self.kfirm_bankruptcy[s,t-1] > 0:
            # randomly choose firms for new entrants to copy
            kentrant_indices: list[np.int32] = list(np.random.choice(np.arange(len(self.kfirms)), size = int(self.kfirm_bankruptcy[s,t-1]))) 
            for i in kentrant_indices:
                # cfirm at index i
                kfirm = self.kfirms[i]
                # copy randomly chosen firm
                new_kfirm = self.copy_kfirm(kfirm)
                # create new kfirm id
                new_kfirm.id = self.kfirm_id
                # increase kfirm id
                self.kfirm_id += 1
                # initialise a single employee
                new_kfirm.employees = []
                for household in self.households:
                    if not household.employed:
                        new_kfirm.hire(household)
                        break
                # randomly choose a new bank
                kfirm_bank: Bank = self.banks[np.random.choice(np.arange(self.num_banks))]
                kfirm_bank.add_deposit_firm(new_kfirm)
                kfirm_bank.add_loan_firm(new_kfirm)
                # update entrant data
                new_kfirm.price[t-1] = self.kfirm_price_index[s,t-1]
                new_kfirm.wage[t-1] = self.avg_wage[s,t-1]
                new_kfirm.loans[:] = 0
                new_kfirm.repayment[:] = 0
                new_kfirm.interest[:] = 0 
                new_kfirm.bank_ids[:] = np.nan
                new_kfirm.debt[:] = 0
                new_kfirm.age[:] = 0
                self.kfirms.append(new_kfirm) 
        # update new total firms list
        self.firms = self.cfirms + self.kfirms 
    
    def labour_market(self, s: int, t: int) -> None:
        """
        A decentralised labour market opens. Firms update their vacancies and wages, households send job applications 
        to firms with open vacancies, then firms hire households if they require more labour.
        
        Parameters
        ----------
            s : int 
                simulation number
            
            t : int
                time period
        """
        # firm labour demand & wages
        for firm in self.firms:
            firm.determine_vacancies(t)
            firm.determine_wages(self.avg_wage[s,t-1], t)
        # probabilities for households to select firms to send job applications to 
        probabilities = self.compute_labour_probabilities(t)
        # households send job applications
        for household in self.households:
            household.determine_applications(self.firms, probabilities, t)
        # firms determine new labour for production
        for firm in self.firms:
            firm.determine_labour(t)

    def production(self, s: int, t: int) -> None:
        """
        Firms engage in production, they first update their productivity then produce goods on either the consumption or capital
        market. They then update their inventories, output growth, cost of production, and prices.
        
        Parameters
        ----------
            s : int 
                simulation number
            
            t : int
                time period
        """
        # Consumption firm output, productivity, costs & prices
        for cfirm in self.cfirms:
            cfirm.determine_productivity(t)
            cfirm.determine_output(t)
            cfirm.determine_inventories(t)
            cfirm.determine_output_growth(t)
            cfirm.determine_costs(t)
            cfirm.determine_prices(self.cfirm_price_index[s,t-1], t)
        # Capital firm output, productivity, costs & prices
        for kfirm in self.kfirms:
            kfirm.determine_productivity(t)
            kfirm.determine_output(t)
            kfirm.determine_inventories(t)
            kfirm.determine_output_growth(t)
            kfirm.determine_costs(t)
            kfirm.determine_prices(self.kfirm_price_index[s,t-1], t)

    def consumption_market(self, t: int) -> None:
        """
        A decentralised consumption good market opens. Households visit each C-firm and consumes consumption goods, 
        they then update their deposit with any remaining income.
        
        Parameters
        ----------
            t : int
                time period
        """
        # probabilities for households to select cfirms to consume cgoods from 
        probabilities = self.compute_cfirm_probabilities(t)
        # household consumption and deposits
        for household in self.households:
            household.determine_consumption(self.cfirms, probabilities, t)
            household.determine_deposits(t)
    
    def capital_market(self, t: int) -> None:
        """
        A decentralised capital market opens. C-firms place orders of investment goods from K-firms.
        
        Parameters
        ----------
            t : int
                time period
        """
        # probabilities for cfirms to select kfirms to order kgoods from
        probabilities = self.compute_kfirm_probabilities(t) 
        # cfirm profits, investment, desired labour and desired loan
        for cfirm in self.cfirms:
            cfirm.determine_expected_demand(t)
            cfirm.determine_desired_output(t)
            cfirm.determine_profits(t)
            cfirm.determine_equity(t)
            cfirm.determine_investment(self.kfirms, probabilities, t)
            cfirm.determine_desired_labour(t)
            cfirm.determine_desired_loan(t)
            cfirm.determine_leverage(t)
        # kfirm profits, desired labour and desired loan
        for kfirm in self.kfirms:
            kfirm.determine_expected_demand(t)
            kfirm.determine_desired_output(t)
            kfirm.determine_profits(t)
            kfirm.determine_equity(t)
            kfirm.determine_desired_labour(t)
            kfirm.determine_desired_loan(t)
            kfirm.determine_leverage(t)
            
    def probability_default(self, t: int) -> None:
        """
        Calculates the probability of firm defaults using a logistic regression model of firm leverage ratios.
        
        Parameters
        ----------
            t : int
                time period
        """
        # cfirm probability of default calculation
        if self.cfirm_data['default'].sum() > 0:
            # instantiate sklearn logistic regression model
            cfirm_model = LogisticRegression()
            # fit model to cfirm data
            cfirm_model.fit(self.cfirm_data[['leverage']], self.cfirm_data['default'])
            # assign probability of default to loan cfirms
            for cfirm in self.cfirms:
                probability_default = cfirm_model.predict_proba(pd.DataFrame(np.array(cfirm.leverage[t]).reshape(1,-1), columns=['leverage']))
                cfirm.probability_default[t] = probability_default[0,1]
        # kfirm probability of default calculation
        if self.kfirm_data['default'].sum() > 0:
            # instantiate sklearn logistic regression model
            kfirm_model = LogisticRegression()
            # fit model to kfirm data
            kfirm_model.fit(self.kfirm_data[['leverage']], self.kfirm_data['default'])
            # assign probability of default to loan kfirms
            for kfirm in self.kfirms:
                probability_default = kfirm_model.predict_proba(pd.DataFrame(np.array(kfirm.leverage[t]).reshape(1,-1), columns=['leverage']))
                kfirm.probability_default[t] = probability_default[0,1]
        
    def credit_market(self, s: int, t: int) -> None:
        """
        A decentralised credit market opens. Firms visit banks and demand their desired loan, banks extend loans based on their 
        risk tolerence proxied by their capital ratio.
        
        Parameters
        ----------
            s : int 
                simulation number
            
            t : int
                time period
        """
        # banks update all data
        for bank in self.banks:
            # bank.determine_probability_default(t)
            bank.determine_loans(self.firms, t)
            bank.determine_deposits(t)
            bank.determine_profits(t)
            bank.determine_equity(t)
            bank.determine_reserves(t)
            bank.determine_capital(t)
            bank.determine_loan_interest(t)
        # probabilities for firms to select a bank to request a loan from
        probabilities = self.compute_loan_probabilities(t)
        # cfirm loan request and deposits
        for cfirm in self.cfirms:
            cfirm.determine_loan(self.banks, probabilities, t)
            cfirm.determine_deposits(t)
            cfirm.determine_bankruptcy(t)
        # kfirm loan request and deposits
        for kfirm in self.kfirms:
            kfirm.determine_loan(self.banks, probabilities, t)
            kfirm.determine_deposits(t)
            kfirm.determine_bankruptcy(t)
            
    def bankrupcty_data(self, t: int) -> None:
        """
        Updates firm bankruptcy data to be used in the estimation of the probability of firm defaults.
        
        Parameters
        ----------
            t : int
                time period
        """
        # firm probability of default data
        for cfirm in self.cfirms:
            # cfirm dataframe of new data 
            new_cfirm_data = pd.DataFrame([{'default': int(cfirm.bankrupt), 'leverage': cfirm.leverage[t]}])
            # concatonate to total cfirm dataframe
            self.cfirm_data = pd.concat([self.cfirm_data, new_cfirm_data], ignore_index=True)
            # reduce size if too large
            if len(self.cfirm_data) > self.length:
                self.cfirm_data = self.cfirm_data.iloc[-self.length:].copy()
        for kfirm in self.kfirms:
            # kfirm dataframe of new data 
            new_kfirm_data = pd.DataFrame([{'default': int(kfirm.bankrupt), 'leverage': kfirm.leverage[t]}])
            # concatonate to total kfirm dataframe
            self.kfirm_data = pd.concat([self.kfirm_data, new_kfirm_data], ignore_index=True)
            # reduce size if too large
            if len(self.kfirm_data) > self.length:
                self.kfirm_data = self.kfirm_data.iloc[-self.length:].copy()

    def bankruptcies(self, s: int, t: int) -> None:
        """
        Firms go bankrupt if they have zero or negative deposits and exit their markets. 
        Banks also go bankrupt if they have zero or negative equity and are bailed out by their depositors.
        
        Parameters
        ----------
            s : int 
                simulation number
            
            t : int
                time period
        """
        # cfirm bankruptcies
        for cfirm in self.cfirms.copy(): 
            if cfirm.bankrupt:
                self.cfirm_bankruptcy[s,t] += 1
                # banks update bad loans
                for bank in self.banks:
                    bank.determine_bad_loans(cfirm, t)
                # fire employees
                for household in cfirm.employees.copy():
                    cfirm.fire(household)
                # remove cfirm from simulation
                self.cfirms.remove(cfirm)
        # kfirm bankruptcies
        for kfirm in self.kfirms.copy():
            if kfirm.bankrupt:
                self.kfirm_bankruptcy[s,t] += 1
                # banks update bad loans
                for bank in self.banks:
                    bank.determine_bad_loans(kfirm, t)
                # fire employees
                for household in kfirm.employees.copy():
                    kfirm.fire(household)
                # remove kfirm from simulation
                self.kfirms.remove(kfirm)
        # update firms list
        self.firms = self.cfirms + self.kfirms 
        # bank bankruptcies
        for bank in self.banks:
            bank.determine_bankrupcty(self.avg_loan_interest[s,t-1], t)
            if bank.bankrupt:
                self.bank_bankruptcy[s,t] += 1

    def market_shares(self, t: int) -> None:
        """
        Calculates C-firms, K-firms and banks market shares.
        
        Parameters
        ----------
            t : int
                time period
        """
        # final market shares for firms and banks
        coutput = sum(cfirm.output[t] for cfirm in self.cfirms)
        koutput = sum(kfirm.output[t] for kfirm in self.kfirms)
        for cfirm in self.cfirms:
            cfirm.determine_balance_sheet(t)
            cfirm.determine_market_share(coutput, t)
        for kfirm in self.kfirms:
            kfirm.determine_balance_sheet(t)
            kfirm.determine_market_share(koutput, t)
        loans = sum(bank.loans[t] for bank in self.banks)
        for bank in self.banks:
            bank.determine_balance_sheet(t)
            bank.determine_market_share(loans, t)

    def copy_cfirm(self, cfirm: ConsumptionFirm) -> ConsumptionFirm:
        """
        Copies a consumption firm object.
        
        Parameters
        ----------
            cfirm : ConsumptionFirm
                consumption firm object
        
        Returns
        -------
            copy of cfirm object : ConsumptionFirm
        """
        # Shallow copy of large data file
        memo = {id(cfirm.employees): cfirm.employees}
        # Deep copy of everything else
        new_cfirm = copy.deepcopy(cfirm, memo)  
        return new_cfirm
    
    def copy_kfirm(self, kfirm: CapitalFirm) -> CapitalFirm:
        """
        Copies a capital firm object.
        
        Parameters
        ----------
            kfirm : CapitalFirm
                capital firm object
        
        Returns
        -------
            copy of kfirm object : CapitalFirm
        """
        # Shallow copy of large data file
        memo = {id(kfirm.employees): kfirm.employees}
         # Deep copy of everything else
        new_kfirm = copy.deepcopy(kfirm, memo)
        return new_kfirm
    
    def compute_cfirm_probabilities(self, t: int) -> list[float]:
        """
        Calculates probability of households to visit cfirms.
        
        Parameters
        ----------
            t : int
                time period
        
        Returns
        -------
            probabilities : list[float]
        """
        total = sum(cfirm.output[t] for cfirm in self.cfirms)
        probabilities = [cfirm.output[t]/total for cfirm in self.cfirms]
        return probabilities
    
    def compute_kfirm_probabilities(self, t: int) -> list[float]:
        """
        Calculates probability of C-firms to visit K-firms.
        
        Parameters
        ----------
            t : int
                time period
        
        Returns
        -------
            probabilities : list[float]
        """
        total = sum(kfirm.output[t] for kfirm in self.kfirms)
        probabilities = [kfirm.output[t]/total for kfirm in self.kfirms]
        return probabilities
    
    def compute_labour_probabilities(self, t: int) -> list[float]:
        """
        Calculates probability of households to visit firms.
        
        Parameters
        ----------
            t : int
                time period
        
        Returns
        -------
            probabilities : list[float]
        """
        total = sum(firm.labour[t] for firm in self.firms)
        probabilities = [firm.labour[t]/total for firm in self.firms]
        return probabilities
    
    def compute_loan_probabilities(self, t: int) -> list[float]:
        """
        Calculates probability of firms to visit banks.
        
        Parameters
        ----------
            t : int
                time period
        
        Returns
        -------
            probabilities : list[float]
        """
        avg_loan = sum(bank.loans[t] for bank in self.banks)/self.num_banks
        total = sum(bank.loans[t] if bank.loans[t] > 1 else avg_loan for bank in self.banks)
        probabilities = [bank.loans[t]/total if bank.loans[t] > 1 else avg_loan/total for bank in self.banks]
        return probabilities
    
    def compute_gini(self, t: int) -> float:
        """
        Calculates the Gini Coefficient between household, measures the amount of inequality.
        
        Parameters
        ----------
            t : int
                time period
        
        Returns
        -------
            Gini Coefficient : float
        """
        wealth = np.sort([household.deposits[t] for household in self.households])
        n = wealth.shape[0]
        index = np.arange(1,n+1)
        gini = ((np.sum((2 * index - n - 1)*wealth))/(n*np.sum(wealth)))
        return gini

    def indicators(self, s: int, t: int) -> None:
        """
        Calculates all macro indicators for the model.
        
        Parameters
        ----------
            s : int 
                simulation number
            
            t : int
                time period
        """
        # number of firms
        num_cfirms = len(self.cfirms)
        num_kfirms = len(self.kfirms)
        # Market Indicators
        self.consumption[s,t]           = sum(cfirm.output[t] for cfirm in self.cfirms)
        self.nominal_consumption[s,t]   = sum(cfirm.output[t]*cfirm.price[t] for cfirm in self.cfirms)
        self.investment[s,t]            = sum(kfirm.output[t] for kfirm in self.kfirms)
        self.nominal_investment[s,t]    = sum(kfirm.output[t]*kfirm.price[t] for kfirm in self.kfirms)
        self.real_gdp[s,t]              = self.consumption[s,t] + self.investment[s,t]
        self.nominal_gdp[s,t]           = self.nominal_consumption[s,t] + self.nominal_investment[s,t]
        self.capital[s,t]               = sum(cfirm.capital[t] for cfirm in self.cfirms)
        self.cfirm_productivity[s,t]    = sum(cfirm.productivity[t] for cfirm in self.cfirms)/num_cfirms
        self.kfirm_productivity[s,t]    = sum(kfirm.productivity[t] for kfirm in self.kfirms)/num_kfirms
        # Firm Indicators
        self.debt[s,t]                  = sum(firm.debt[t] for firm in self.firms)
        self.profits[s,t]               = sum(firm.profits[t] for firm in self.firms)
        self.cfirm_price_index[s,t]     = self.nominal_consumption[s,t]/self.consumption[s,t]
        self.kfirm_price_index[s,t]     = self.nominal_investment[s,t]/self.investment[s,t]
        self.cfirm_nhhi[s,t]            = (sum(cfirm.market_share[t]**2 for cfirm in self.cfirms) - 1/num_cfirms)/(1 - 1/num_cfirms)
        self.kfirm_nhhi[s,t]            = (sum(kfirm.market_share[t]**2 for kfirm in self.kfirms) - 1/num_kfirms)/(1 - 1/num_kfirms)
        self.cfirm_hpi[s,t]             = sum(abs(cfirm.market_share[t] - cfirm.market_share[t-self.steps]) for cfirm in self.cfirms)
        self.kfirm_hpi[s,t]             = sum(abs(kfirm.market_share[t] - kfirm.market_share[t-self.steps]) for kfirm in self.kfirms)
        # Household Indicators
        self.employment[s,t]            = sum(len(firm.employees) for firm in self.firms)
        self.unemployment_rate[s,t]     = (self.num_households - self.employment[s,t])/self.num_households
        self.wages[s,t]                 = sum(firm.wage_bill[t] for firm in self.firms)
        self.avg_wage[s,t]              = self.wages[s,t]/self.employment[s,t]
        self.vacancy_ratio[s,t]         = sum(firm.vacancies[t] for firm in self.firms)/self.num_households
        self.gini[s,t]                  = self.compute_gini(t)
        # Bank Indicators
        self.bank_nhhi[s,t]             = (sum(bank.market_share[t]**2 for bank in self.banks) - 1/self.num_banks)/(1 - 1/self.num_banks)
        self.bank_hpi[s,t]              = sum(abs(bank.market_share[t] - bank.market_share[t-self.steps]) for bank in self.banks)
        self.avg_loan_interest[s,t]     = sum(bank.loan_interest[t] for bank in self.banks)/self.num_banks
        self.avg_reserve_ratio[s,t]     = sum(bank.reserve_ratio[t] for bank in self.banks)/self.num_banks
        self.avg_capital_ratio[s,t]     = sum(bank.capital_ratio[t] for bank in self.banks)/self.num_banks
        self.money_supply[s,t]          = sum(bank.deposits[t] for bank in self.banks)
        # Network Indicators
        self.bank_mean_degree[s,t]      = sum(bank.degree[t] for bank in self.banks)/self.num_banks
        self.cfirm_mean_degree[s,t]     = sum(len(np.unique(cfirm.bank_ids,return_counts=True)[0])-1 for cfirm in self.cfirms)/self.num_cfirms 
        self.kfirm_mean_degree[s,t]     = sum(len(np.unique(kfirm.bank_ids,return_counts=True)[0])-1 for kfirm in self.kfirms)/self.num_kfirms 
        
    def update_database(self, s: int, t: int) -> None:
        """
        Saves data in SQLite3 database.
        
        Parameters
        ----------
            cur : sqlite3.Cursor
                SQLite3 database cursor object
            
            s : int 
                simulation number
            
            t : int
                time period
        """
        ### 1. Macro Table ###
        # insert values into macro table
        macro_array_attrs = [name for name, value in self.__dict__.items() if isinstance(value, np.ndarray)]
        macro_row = [s, t] + [float(getattr(self, attr)[s, t]) for attr in macro_array_attrs]
        # placeholder for macro insert values
        macro_placeholder = ", ".join(["?"] * len(macro_row))
        self.cursor.execute(
            f"INSERT INTO macro_data VALUES ({macro_placeholder});",
            macro_row
        )
        
        ### 2. Firm Table ###
        # union of attributes across firm types
        cfirm_attrs = {name for name, value in self.firms[0].__dict__.items() if isinstance(value, np.ndarray)}
        if any(type(firm).__name__ == "CapitalFirm" for firm in self.firms):
            kfirm = next(f for f in self.firms if type(f).__name__ == "CapitalFirm")
            kfirm_attrs = {name for name, value in kfirm.__dict__.items() if isinstance(value, np.ndarray)}
        else:
            kfirm_attrs = set()
        # all firm attributes (both cfirm & kfirm)
        all_firm_attrs = sorted(cfirm_attrs.union(kfirm_attrs))
        # firm rows to add to table
        firm_rows = []
        for firm in self.firms:
            row = [s, t, firm.id, firm.__str__()]
            for attr in all_firm_attrs:
                if hasattr(firm, attr):
                    row.append(float(getattr(firm, attr)[t]))
                else:
                    row.append(0.0)  # or None if you want NULLs
            firm_rows.append(row)
        # placeholder for firm insert values
        firm_placeholder = ", ".join(["?"] * len(firm_rows[0]))
        # execute sql statement
        self.cursor.executemany(
            f"INSERT INTO firm_data VALUES ({firm_placeholder});",
            firm_rows
        )

        
        ### 3. Household Table ###
        # household data for current period
        household_array_attrs = [name for name, value in self.households[0].__dict__.items() if isinstance(value, np.ndarray)]
        household_rows = []
        for household in self.households:
            row = [s, t, household.id, household.employed] + [float(getattr(household, attr)[t]) for attr in household_array_attrs]
            household_rows.append(row)
        # placeholder for household insert values
        household_placeholder = ", ".join(["?"] * len(household_rows[0]))
        self.cursor.executemany(
            f"INSERT INTO household_data VALUES ({household_placeholder});",
            household_rows
        )
        
        ### 4. Bank Table ###
        # bank data for current period
        bank_array_attrs = [name for name, value in self.banks[0].__dict__.items() if isinstance(value, np.ndarray)]
        bank_rows = []
        for bank in self.banks:
            row = [s, t, bank.id] + [float(getattr(bank, attr)[t]) for attr in bank_array_attrs]
            bank_rows.append(row)
        # placeholder for household insert values
        bank_placeholder = ", ".join(["?"] * len(bank_rows[0]))
        self.cursor.executemany(
            f"INSERT INTO bank_data VALUES ({bank_placeholder});",
            bank_rows
        )
        
        ### 5. Edge Table ###
        # node data for current period
        edge_rows = [
            [s, t, bank.__repr__(), firm.__repr__(), firm.compute_bank_loans(bank.id), firm.assets[t], bank.assets[t]]
            
            for bank in self.banks for firm in bank.loan_firms
        ]
        # insert values into edge table
        edge_placeholder = ", ".join(["?"] * len(edge_rows[0]))
        self.cursor.executemany(
            f"""INSERT INTO edge_data VALUES ({edge_placeholder});""",
            edge_rows
        )
    
    def close_databse(self) -> None:
        self.connection.commit()
        self.cursor.close()
        
    def step(self, s: int, t: int) -> None:
        """
        Update model time step.
        
        Parameters
        ----------
            s : int
                simulation index
            t : int 
                time period index
        """
        self.new_entrants(s, t)
        self.labour_market(s, t)
        self.production(s, t)
        self.consumption_market(t)
        self.capital_market(t)
        self.probability_default(t)
        self.credit_market(s, t)
        self.bankrupcty_data(t)
        self.bankruptcies(s, t)
        self.market_shares(t)
        self.indicators(s, t)
        self.update_database(s, t)

    def run(self, params: dict, init_seed: int | None = None) -> None:
        """
        Runs batch of num_sims simulations of the model, as defined in num_sims in the config/parameters.yaml file.
        
        Parameters
        ----------
            params: dict 
                dictionary of simulation parameters from config/parameters.yaml file
                
            init_seed : int | None
                initial random seed for the simulation (added to each simulation for a new random simulation)
        """
        # initialise sql database
        self.initialise_database()
        # start simulation
        for s in tqdm(range(self.simulations)):
            # set random seed if entered
            if not init_seed:
                np.random.seed(init_seed + s)
            # instantiate agents
            self.instatiate_agents(params)
            # initialise simulation s
            self.initialise_simulation()
            # start simulation s
            for t in tqdm(range(1, self.time), leave=True):
                self.step(s, t)