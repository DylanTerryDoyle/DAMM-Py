import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod

class Bank:
    
    """
    Bank class
    ==========
    Bank agent class used to simulate banks in the model.
    
    Attributes
    ----------
        id : int 
            bank object id
    
        steps : int
            number of time steps per year
    
        time : int
            total number of time steps
    
        dt : float
            time delta, inverse steps
    
        adjust : float
            speed of adjustment 
        
        deposit_interest : float
            deposit interest rate
        
        natural_interest : float
            initial loan ineterst rate 
        
        sigma : float
            loan interest rate standard deviation
        
        loan_periods : int
            number of loan repayment periods 
    
        capital_intercept : float
            risk free capital ratio
        
        capital_slope : float
            sensitivity of capital ratio to bad loans ratio
        
        bankruptcies : int
            total number of bankruptcies
    
        bankrupt : bool
            bankruptcy condition
    
    Data
    ----
        households : list[Household]
            households with deposits at the bank
    
        loan_firms : list[ConsumptionFirm | CapitalFirm]
            firms with loans at the bank
    
        deposit_firms : list[ConsumptionFirm | CapitalFirm]
            firms with deposits at the bank
        
        profits : numpy.ndarray
            nominal profits
    
        loan_interest : numpy.ndarray
            loan interest rate
        
        equity : numpy.ndarray
            nominal equity 
        
        deposits : numpy.ndarray
            total deposits
        
        loans : numpy.ndarray
            total loans
    
        bad_loans : numpy.ndarray
            total bad loans per period
        
        expected_bad_loans : numpy.ndarray
            expected bad loans 
        
        expected_bad_loans_ratio : numpy.ndarray
            expected bad loans ratio to total loans
        
        reserves : numpy.ndarray
            nominal reserves
    
        reserve_ratio : numpy.ndarray
            reserve to deposit ratio
        
        capital_ratio : numpy.ndarray
            capital ratio (equity to loans ratio)
             
        min_capital_ratio : numpy.ndarray
            minimum desired capital ratio
        
        market_share : numpy.ndarray
            market share of loans
        
        assets : numpy.ndarray
            value of assets
    
        liabilities : numpy.ndarray
            value of liabilities
    
    Methods
    -------
        __init__ (self, id: int, params: dict) -> None
        
        __repr__(self) -> str
        
        __str__(self) -> str
        
        initialise_accounts(self) -> None
        
        determine_loans(self, firms: list['ConsumptionFirm | CapitalFirm'], t: int) -> None
        
        determine_deposits(self, t: int) -> None
        
        determine_profits(self, t: int) -> None
        
        determine_equity(self, t: int) -> None
        
        determine_reserves(self, t: int) -> None
        
        determine_capital(self, t: int) -> None
        
        determine_loan_interest(self, t: int) -> None
        
        determine_bad_loans(self, bankrupt_firm: 'ConsumptionFirm | CapitalFirm', t: int) -> None
        
        determine_bankrupcty(self, avg_loan_interest: float, t: int) -> None
        
        determine_market_share(self, total_loans: float, t: int) -> None
        
        compute_loan_supply(self, firm: 'Firm', t: int) -> float
        
        compute_total_loan_interest(self) -> float
        
        compute_total_loans(self) -> float
        
        compute_expected_bad_loans(self, t: int) -> float
        
        compute_total_deposits(self, t: int) -> float
        
        add_household(self, household: 'Household') -> None
        
        remove_household(self, household: 'Household') -> None
        
        add_deposit_firm(self, firm: 'Firm | ConsumptionFirm | CapitalFirm') -> None
        
        remove_deposit_firm(self, firm: 'Firm | ConsumptionFirm | CapitalFirm') -> None
        
        add_loan_firm(self, firm: 'Firm | ConsumptionFirm | CapitalFirm') -> None
        
        remove_loan_firm(self, firm: 'Firm | ConsumptionFirm | CapitalFirm') -> None
        
        update_loan_firms(self, firms: list['ConsumptionFirm | CapitalFirm']) -> None
    """
    
    def __init__ (self, id: int, params: dict) -> None:
        """
        Bank class initialisation.
        
        Parameters
        ----------
            id : int
                unique id
            params : dict
                model parameters
        """
        # Parameters
        self.id:                        int   = id
        self.steps:                     int   = params['simulation']['steps']
        self.time:                      int   = (params['simulation']['years'] + params['simulation']['start'])*self.steps + 1
        self.dt:                        float = 1/self.steps
        self.adjust:                    float = params['bank']['adjust']*self.dt
        self.deposit_interest:          float = params['bank']['deposit_interest']*self.dt
        self.natural_interest:          float = params['bank']['loan_interest']
        self.sigma:                     float = params['bank']['sigma']*np.sqrt(self.dt)
        self.loan_periods:              int   = params['bank']['loan_years']*self.steps
        self.capital_intercept:         float = params['bank']['capital_intercept']
        self.capital_slope:             float = params['bank']['capital_slope']
        self.bankruptcies:              int = 0
        self.bankrupt:                  bool = False
        # Mutable data
        self.households:                list[Household] = []
        self.loan_firms:                list[Firm | ConsumptionFirm | CapitalFirm] = []
        self.deposit_firms:             list[Firm | ConsumptionFirm | CapitalFirm] = []
        # Data 
        self.profits:                   NDArray = np.zeros(shape=self.time)
        self.loan_interest:             NDArray = np.zeros(shape=self.time)
        self.equity:                    NDArray = np.zeros(shape=self.time)
        self.deposits:                  NDArray = np.zeros(shape=self.time)
        self.loans:                     NDArray = np.zeros(shape=self.time)
        self.bad_loans:                 NDArray = np.zeros(shape=self.time)
        self.expected_bad_loans:        NDArray = np.zeros(shape=self.time)
        self.expected_bad_loans_ratio:  NDArray = np.zeros(shape=self.time)
        self.reserves:                  NDArray = np.zeros(shape=self.time)
        self.reserve_ratio:             NDArray = np.zeros(shape=self.time)
        self.capital_ratio:             NDArray = np.zeros(shape=self.time)
        self.min_capital_ratio:         NDArray = np.zeros(shape=self.time)
        self.market_share:              NDArray = np.zeros(shape=self.time)
        self.degree:                    NDArray = np.zeros(shape=self.time)
        self.assets:                    NDArray = np.zeros(shape=self.time)
        self.liabilities:               NDArray = np.zeros(shape=self.time)
        # Initial values
        self.loan_interest[0]           = params['bank']['loan_interest']

    def __repr__(self) -> str:
        """
        Returns printable unique bank agent representaton.
        
        Returns
        -------
            representation : str
        """
        return f'Bank: {self.id}'
    
    def __str__(self) -> str:
        """ 
        Returns printable agent type.
        
        Returns
        -------
            agent type : str
        """
        return 'Bank'

    def initialise_accounts(self) -> None:
        """
        Calculate inital values for time series.
        """
        self.loans[0] = self.compute_total_loans()
        self.deposits[0] = self.compute_total_deposits(0)
        self.expected_bad_loans[0] = self.compute_expected_bad_loans(0)
        self.expected_bad_loans_ratio[0] = self.expected_bad_loans[0]/self.loans[0]
        self.min_capital_ratio[0] = self.capital_intercept + self.capital_slope*self.expected_bad_loans_ratio[0]
        self.equity[0] = self.min_capital_ratio[0]*self.loans[0]
        self.reserves[0] = self.deposits[0] + self.equity[0] - self.loans[0]
        self.reserve_ratio[0] = self.reserves[0]/self.deposits[0]
        self.profits[0] = self.compute_total_loan_interest() - self.deposit_interest*self.deposits[0]
    
    def determine_loans(self, firms: list['ConsumptionFirm | CapitalFirm'], t: int) -> None:
        """
        Update firms that still have outstanding loans, calculate total loans, expected bad loans, 
        and the expected bad loans ratio.
        
        Parameters
        ----------
            firms : list[ConsumptionFirm | CapitalFirm]
                list of all firms in the market
            
            t : int
                time period
        """
        # update loan firms, remove all firms that have repaid loans
        self.update_loan_firms(firms)
        # compute total loans extended to firms
        self.loans[t] = self.compute_total_loans()
        # compute expected bad loans, firm loans times probability of default
        self.expected_bad_loans[t] = self.compute_expected_bad_loans(t)
        if self.loans[t] != 0:
            # calculate expected bad loans as a ratio of total loans
            self.expected_bad_loans_ratio[t] = self.expected_bad_loans[t]/self.loans[t]
        # firm degree
        self.degree[t] = len(self.loan_firms)

    def determine_deposits(self, t: int) -> None:
        """
        Calculate total deposits of firms and households.
        
        Paremeters
        ----------
            t : int
                time period 
        """
        # compute firm and household all deposits
        self.deposits[t] = self.compute_total_deposits(t-1)

    def determine_profits(self, t: int) -> None:
        """
        Calculate total bank profits.

        Parameters
        ----------
            t : int
                time period
        """
        # profits as the difference between loan interest and deposit interest
        self.profits[t] = self.compute_total_loan_interest() - self.deposit_interest*self.deposits[t]
        
    def determine_equity(self, t: int) -> None:
        """
        Calculate bank equity.
        
        Parameters
        ----------
            t : int
                time period
        """
        # update equity with profits
        self.equity[t] = self.equity[t-1] + self.profits[t]
        
    def determine_reserves(self, t: int) -> None:
        """
        Calculate bank reserves and reserve to deposit ratio.
        
        Parameters
        ----------
            t : int
                time period
        """
        # calculate reserves from accounting identity
        self.reserves[t] = self.deposits[t] + self.equity[t] - self.loans[t]
        if self.deposits[t] != 0:
            # reserve as a ratio of total deposits
            self.reserve_ratio[t] = self.reserves[t]/self.deposits[t]

    def determine_capital(self, t: int) -> None:
        """
        Calculate bank capital ratio and minimum desired capital ratio.
        
        Parameters
        ----------
            t : int
                time period
        """
        # capital adequacy ratio, capital (equity) as a ratio of loans (assets) 
        if self.loans[t] != 0:
            self.capital_ratio[t] = self.equity[t]/self.loans[t]
        # minimum capital adequacy ratio, function of risk (expected bad loans ratio)
        self.min_capital_ratio[t] = self.capital_intercept + self.capital_slope*self.expected_bad_loans_ratio[t]

    def determine_loan_interest(self, t: int) -> None:
        """
        Calculate bank loan interest rate.
        
        Parameters
        ----------
            t : int 
                time period
        """
        # update loan interest rate
        if self.min_capital_ratio[t] >= self.capital_ratio[t] and self.loans[t] > 0:
            self.loan_interest[t] = self.loan_interest[t-1]*(1 + self.sigma*abs(np.random.randn())) + self.adjust*(self.natural_interest - self.loan_interest[t-1])
        else:
            self.loan_interest[t] = self.loan_interest[t-1]*(1 - self.sigma*abs(np.random.randn())) + self.adjust*(self.natural_interest - self.loan_interest[t-1])

    def determine_bad_loans(self, bankrupt_firm: 'ConsumptionFirm | CapitalFirm', t: int) -> None:
        """
        Calculate bank bad loans from bankrupt_firm.
        
        Parameters
        ----------
            bankrupt_firm : ConsumptionFirm | CapitalFirm
                bankrupt firm object
            
            t : int
                time period
        """
        # remove bankrupt firm from deposit accounts
        if bankrupt_firm in self.deposit_firms:
            self.remove_deposit_firm(bankrupt_firm)
        # remove bankrupt firm from loan accounts
        if bankrupt_firm in self.loan_firms:
            self.remove_loan_firm(bankrupt_firm)
            # all loans extended to bankrupt firm
            firm_loans = bankrupt_firm.compute_bank_loans(self.id)
            # increase bad loans 
            self.bad_loans[t] += firm_loans
            # reduce equity as banks absorb all risk
            self.equity[t] -= firm_loans

    def determine_bankrupcty(self, avg_loan_interest: float, t: int) -> None:
        """
        Determine if bank is bankrupt, bail out if the bank is bankrupt.
        
        Parameters
        ----------
            avg_loan_interest : float
                average market loan interest rate
            
            t : int
                time period
        """
        # bankrupcty condition when equity is zero or negative
        if self.equity[t] <= 0.001:
            self.bankrupt = True
            self.bankruptcies += 1
            # bail out amount
            bail_out = self.min_capital_ratio[t]*(self.loans[t] + self.reserves[t])
            # increase equity by bail out amoun
            self.equity[t] = bail_out
            # depositors pay for bail out
            depositors = self.deposit_firms + self.households
            for depositor in depositors:
                depositor_cost = bail_out*(depositor.deposits[t]/self.deposits[t])
                depositor.deposits[t] -= depositor_cost
            # reset loan interest in industry averae
            self.loan_interest[t] = avg_loan_interest
        # no bankruptcy
        else:
            self.bankrupt = False
            
    def determine_balance_sheet(self, t: int) -> None:
        # calculate assets
        self.assets[t] = self.loans[t] + self.reserves[t]
        # calculate liabilities
        self.liabilities[t] = self.deposits[t]
    
    def determine_market_share(self, total_loans: float, t: int) -> None:
        """
        Calculate bank loan market share.
        
        Parameters
        ----------
            total_loans : float
                total market loans
        
            t : int
                time period
        """
        self.market_share[t] = self.loans[t]/total_loans

    def compute_loan_supply(self, firm: 'Firm', t: int) -> float:
        """
        Calculates and returns loan supply to the input firm demand a loan.
        
        Parameters
        ----------
            firm : Firm
                firm object
            
            t : int
                time period
        
        Returns
        -------
            loan_supply : float
                loan extended to firm object
        """
        # loan supply condition
        if self.min_capital_ratio[t] < self.capital_ratio[t] or self.loans[t] <= 0.001:
            loan_supply = firm.desired_loan[t]
        else:
            loan_supply = 0
        return loan_supply

    def compute_total_loan_interest(self) -> float:
        """
        Calculates and returns total loan interest collected from all firms.
        
        Returns
        -------
            loan_interest : float
                total loan interest paid by firms
        """
        loan_interest = sum(firm.compute_bank_interest(self.id) for firm in self.loan_firms)
        return loan_interest
    
    def compute_total_loans(self) -> float:
        """
        Calculates and returns total outstanding loans.
        
        Returns
        -------
            loans : float
                total outstanding loans to firms 
        """
        loans = sum(firm.compute_bank_loans(self.id) for firm in self.loan_firms)
        return loans
    
    def compute_expected_bad_loans(self, t: int) -> float:
        """
        Calculates and returns expected bad loans.
        
        Parameters
        ----------
            t : int
                time period
        
        Returns
        -------
            exp_bad_loans : float
                banks expected bad loans 
        """
        exp_bad_loans = sum(firm.compute_bank_loans(self.id)*firm.probability_default[t] for firm in self.loan_firms)
        return exp_bad_loans
    
    def compute_total_deposits(self, t: int) -> float:
        """
        Calculates and returns total deposits held by firms and households.
        
        Parameters
        ----------
            t : int
                time period
        
        Returns
        -------
            deposits : float
                total deposits of firms and households
        """
        firm_deposits = sum(firm.deposits[t] for firm in self.deposit_firms)
        household_deposits = sum(household.deposits[t] for household in self.households)
        deposits = firm_deposits + household_deposits
        return deposits
    
    def add_household(self, household: 'Household') -> None:
        """
        Add household to list of deposit households.
        
        Parameters
        ----------
            household : Household
                household object 
        """
        if household not in self.households:
            self.households.append(household)

    def remove_household(self, household: 'Household') -> None:
        """
        Remove household from list of deposit households.
        
        Parameters
        ----------
            household : Household
                household object 
        """
        if household in self.households:
            self.households.remove(household)

    def add_deposit_firm(self, firm: 'Firm | ConsumptionFirm | CapitalFirm') -> None:
        """
        Add firm to list of deposit firms.
        
        Parameters
        ----------
            firm : Firm | ConsumptionFirm | CapitalFirm
                firm object 
        """
        if firm not in self.deposit_firms:
            self.deposit_firms.append(firm)

    def remove_deposit_firm(self, firm: 'Firm | ConsumptionFirm | CapitalFirm') -> None:
        """
        Remove firm from list of deposit firms.
        
        Parameters
        ----------
            firm : Firm | ConsumptionFirm | CapitalFirm
                firm object 
        """
        if firm in self.deposit_firms:
            self.deposit_firms.remove(firm)

    def add_loan_firm(self, firm: 'Firm | ConsumptionFirm | CapitalFirm') -> None:
        """
        Add firm to list of loan firms.
        
        Parameters
        ----------
            firm : Firm | ConsumptionFirm | CapitalFirm
                firm object 
        """
        if firm not in self.loan_firms:
            self.loan_firms.append(firm)

    def remove_loan_firm(self, firm: 'Firm | ConsumptionFirm | CapitalFirm') -> None:
        """
        Remove firm from list of loan firms.
        
        Parameters
        ----------
            firm : Firm | ConsumptionFirm | CapitalFirm
                firm object 
        """
        if firm in self.loan_firms:
            self.loan_firms.remove(firm)

    def update_loan_firms(self, firms: list['ConsumptionFirm | CapitalFirm']) -> None:
        """
        Update list of loan firms.
        
        Parameters
        ----------
            firms : list[ConsumptionFirm | CapitalFirm]
                list of all firms with loans
        """
        for firm in firms:
            firm_loans = firm.compute_bank_loans(self.id)
            if firm_loans <= 0.001:
                self.remove_loan_firm(firm)
            else:
                self.add_loan_firm(firm)


class Household:
    
    """
    Household Class
    ===============
    Household agent class used to simulate households in the model.
    
    Parameters
    ----------
        id : int
            unique id
        
        init_wage : float
            initial market wage rate
        
        params : dict
            model parameters
    
    Attributes
    ----------
        id : int
            unique id
    
        steps : int
            number of time steps per year
        
        time : int
            total number of time steps 
    
        dt : float
            time delta, inverse steps
    
        mpc_deposits : float
            marginal propensity to consume out of deposits
    
        num_firms : int
            number of applications sent on the labour market
    
        num_cfirms : int
            number of C-firms visited on the consumption market
    
        deposit_interest : float
            bank deposit interest rate
    
        employed : bool
            employment status
    
    Data
    ----
        wage : numpy.ndarray
            wage rate
            
        income : numpy.ndarray
            income (wage + deposit interest) 
    
        deposits : numpy.ndarray
            deposits held at the bank
        
        expenditure : numpy.ndarray
            expenditure on consumption goods
        
        desired_expenditure : numpy.ndarray
            desired expenditure on consumption goods
    
    Methods
    -------
        __init__(self, id: int, init_wage: float, params: dict) -> None
        
        __repr__(self) -> str    
        
        __str__(self) -> str
        
        determine_consumption(self, cfirms: list['ConsumptionFirm'], probabilities: list[float], t: int) -> None
        
        determine_applications(self, firms: list['ConsumptionFirm | CapitalFirm'], probabilities: list[float], t: int) -> None
        
        determine_deposits(self, t: int) -> None
    """
    
    def __init__(self, id: int, init_wage: float, params: dict) -> None:
        """
        Household class initialisation.
        
        Parameters
        ----------
            id : int
                unique id
            
            init_wage : float
                initial wage rate
            
            params : dict
                model parameters
        """
        # Parameters 
        self.id:                  int   = id
        self.steps:               int   = params['simulation']['steps']
        self.time:                int   = (params['simulation']['years'] + params['simulation']['start'])*self.steps + 1
        self.dt:                  float = 1/self.steps
        self.mpc_deposits:        float = params['household']['mpc_deposits']*self.dt
        self.num_firms:           int   = params['household']['num_firms']
        self.num_cfirms:          int   = params['household']['num_cfirms']
        self.deposit_interest:    float = params['bank']['deposit_interest']*self.dt
        self.employed:            bool  = False
        # Data
        self.wage:                NDArray = np.zeros(shape=self.time)
        self.income:              NDArray = np.zeros(shape=self.time)
        self.deposits:            NDArray = np.zeros(shape=self.time)
        self.expenditure:         NDArray = np.zeros(shape=self.time)
        self.desired_expenditure: NDArray = np.zeros(shape=self.time)
        # Initial values
        self.wage[0]              = init_wage
        self.income[0]            = init_wage
        self.deposits[0]          = init_wage
    
    def __repr__(self) -> str:
        """
        Returns printable unique household agent representaton.
        
        Returns
        -------
            representation : str
        """
        return f'Household: {self.id}'
    
    def __str__(self) -> str:
        """ 
        Returns printable agent type.
        
        Returns
        -------
            agent type : str
        """
        return 'Household'

    def determine_consumption(self, cfirms: list['ConsumptionFirm'], probabilities: list[float], t: int) -> None:
        """
        Household visit C-firms with probability given by probabilities list and consumes consumption goods.
        
        Parameters
        ----------
            cfirms : list[ConsumptionFirm]
                list of all C-firms in the market
            
            probabilities : list[float]
                list of probabilities that household visits a given C-firm
            
            t : int 
                time period
        """
        # household income for expenditure
        self.income[t] = self.wage[t] + self.deposit_interest*self.deposits[t-1]
        # desired expenditure as income plus deposit (savings) expenditure
        self.desired_expenditure[t] = self.income[t] + self.mpc_deposits*self.deposits[t-1]
        # randomly visits cfirms (index)
        visited_cfirms_indices: list[np.int32] = list(np.random.choice(np.arange(len(cfirms)), size=self.num_cfirms, replace=False, p=probabilities))
        # sorts cfirms indices by price
        sorted_visited_cfirms_indices = sorted(visited_cfirms_indices, key=lambda i: cfirms[i].price[t], reverse=False)
        # consumption from cfirms 
        for i in sorted_visited_cfirms_indices:
            # cfirm at index i
            cfirm = cfirms[i]
            # amount of goods demanded 
            goods_demanded = self.desired_expenditure[t]/cfirm.price[t]
            # amount of goods purchased, constrained by cfirm inventories
            goods_purchased = min(cfirm.inventories[t], goods_demanded)
            # cost of purchased goods
            goods_cost = goods_purchased*cfirm.price[t]
            # update cfirm demand
            cfirm.demand[t] += goods_demanded
            # update cfirm inventories
            cfirm.inventories[t] -= goods_purchased
            # update cfirm quantity sold
            cfirm.quantity[t] += goods_purchased
            # update expenditure
            self.expenditure[t] += goods_cost
            # update desired expenditure
            self.desired_expenditure[t] -= goods_cost
            # household stops consuming if desired expenditure is exceeded
            if self.desired_expenditure[t] <= 0.001:
                break
    
    def determine_applications(self, firms: list['ConsumptionFirm | CapitalFirm'], probabilities: list[float], t: int) -> None:
        """
        Unemployed household sends job applications to firms with open vacancies.
        
        Parameters
        ----------
            firms : list[ConsumptionFirm | CapitalFirm]
                list of all firms (C-firms + K-firms) in the market
            
            probabilities : list[float]
                list of probabilities that household sends an application to a given firm
        
            t : int
                time period
        """
        # only applies to jobs if unemployed
        if not self.employed:
            # randomly visits firms
            selected_firms_indices: list[np.int32] = list(np.random.choice(np.arange(len(firms)), size=self.num_firms, replace=False, p=probabilities))
            # sorts firms by wage
            sorted_selected_firms_indices = sorted(selected_firms_indices, key=lambda i: firms[i].wage[t], reverse=True)
            # applies to first firm with highest wage and open vacancies
            for i in sorted_selected_firms_indices:
                # firm at index i
                firm = firms[i]
                # check if firm has vacancies
                if firm.vacancies[t] > 0:
                    # append to firm applications
                    firm.applications.append(self)

    def determine_deposits(self, t: int) -> None:
        """
        Calculates household deposits in next period.
        
        Parameters
        ----------
            t : int
                time period
        """
        # updates deposits (savings) as earned income minus any expenditure
        self.deposits[t] = self.deposits[t-1] + self.income[t] - self.expenditure[t]
        


class Firm(ABC):
    
    """
    Firm Class
    ==========
    Base class of firm agent used to simulate firms in the model.
    
    Attributes
    ----------
        id : int
            bank object id
    
        steps : int
            number of time steps per year
        
        time : int
            total number of time steps
    
        dt : float  
            time delta, inverse steps
        
        adapt : float
            adaptive update speed
        
        adjust : float
            speed of adjustment 
        
        growth : float
            average firm productivity growth
        
        sigma : float
            firm productivity standard deviation
        
        sigma_p : float
            firm price standard deviation
        
        sigma_w : float
            firm wage standard deviation
        
        depreciation : float
            capital good depreciation rate

        num_banks : int
            number of banks visited
    
        deposit_interest : float
            bank deposit interest rate
        
        loan_periods : int
            number of repayment periods for a loan 
        
        repayment_rate : float
            loan repayment rate, inverse loan periods
        
        bankrupt : bool
            bankruptcy condition
    
    Data
    ----
        
        employees : list[Household]
            household employees
        
        applications : list[Household]
            applications from households
        
        productivity : numpy.array
            labour productivity
        
        productivity_growth : numpy.array
            labour productivity growth rate
        
        expected_productivity : numpy.array
            expected labour productivity 
        
        output : numpy.array
            production output 
        
        output_growth : numpy.array
            output growth rate
        
        desired_output : numpy.array
            desired output 
        
        demand : numpy.array
            demand for goods
        
        expected_demand : numpy.array
            expected demand for goods
        
        quantity : numpy.array
            quantity of goods sold
        
        inventories : numpy.array
            inventories of goods
        
        desired_inventories : numpy.array
            desired inventories 
        
        labour : numpy.array
            number of employees
        
        desired_labour : numpy.array
            desired number of employees
        
        labour_demand : numpy.array
            demand for new employees
        
        vacancies : numpy.array
            number of vacant positions 
        
        wage : numpy.array
            wage rate 
        
        wage_bill : numpy.array
            wage bill (wage x labour)
        
        price : numpy.array
            price of goods
        
        profits : numpy.array
            profits 
        
        profit_share : numpy.array
            profit share of revenue 
        
        equity : numpy.array
            equity 
        
        deposits : numpy.array
            deposits at bank
        
        desired_loan : numpy.array
            desired loan from banks
        
        debt : numpy.array
            debt
        
        total_repayment : numpy.array
            total cost of loan repayment 
        
        total_interest : numpy.array
            total cost of interest payments 
        
        leverage : numpy.array
            leverage ratio
        
        probability_default : numpy.array
            probability of defaulting 
        
        age : numpy.array
            age in years
        
        market_share : numpy.array
            market share 
        
        assets : numpy.ndarray
            value of assets
        
        liabilities : numpy.ndarray
            value of liabilities
        
        loans : numpy.array
            current outstanding loans
        
        repayment : numpy.array
            current repayment amount for each loan
        
        interest : numpy.array
            current interest cost on each loan 
        
        bank_ids : numpy.array
            bank id for each loan
    
    Methods 
    -------
        determine_vacancies(self, t: int) -> None
        
        determine_wages(self, avg_wage: float, t: int) -> None
        
        determine_labour(self, t: int) -> None
        
        determine_productivity(self, t: int) -> None
        
        determine_output(self, t: int) -> None
        
        determine_inventories(self, t: int) -> None
        
        determine_market_share(self, market_output: float, t: int) -> None
        
        determine_output_growth(self, t: int) -> None
        
        determine_costs(self, t: int) -> None
        
        determine_prices(self, avg_price: float, t: int) -> None
        
        determine_profits(self, t: int) -> None
        
        determine_equity(self, t: int) -> None
        
        determine_expected_demand(self, t: int) -> None
        
        determine_desired_output(self, t: int) -> None
        
        determine_desired_labour(self, t: int) -> None
        
        determine_desired_loan(self, t: int) -> None
        
        determine_leverage(self, t) -> None
        
        determine_loan(self, banks: list[Bank], probabilities: list[float], t: int) -> None
        
        determine_deposits(self, t: int) -> None
        
        determine_bankruptcy(self, t: int) -> None
        
        determine_balance_sheet(self, t: int) -> None
        
        compute_total_debt(self, t: int) -> float
        
        compute_amortisation(self, loan: float, interest: float) -> float
        
        compute_new_loan(self, loan: float, interest: float, bank_id: int, t: int) -> None
        
        compute_total_repayment(self) -> float
        
        compute_total_interest(self) -> float
        
        compute_bank_loans(self, bank_id: int) -> float
        
        compute_bank_interest(self, bank_id: int) -> float
        
        hire(self, household: Household) -> None
        
        fire(self, household: Household) -> None
    """
    
    def __init__(self, id: int, initial_output: float, initial_wage: float, params: dict) -> None:
        """
        Firm class initialisation.
        
        Parameters
        ----------
            id : int
                unique id
        
            init_output : float
                initial output
        
            init_wage : float
                initial wage rate
        
            params : dict
                model parameters
        """
        # Parameters
        self.id:                        int   = id
        self.steps:                     int   = params['simulation']['steps']
        self.time:                      int   = (params['simulation']['years'] + params['simulation']['start'])*self.steps + 1
        self.dt:                        float = 1/self.steps
        self.adapt:                     float = params['firm']['adapt']*self.dt
        self.adjust:                    float = params['firm']['adjust']*self.dt
        self.growth:                    float = params['firm']['growth']*self.dt
        self.sigma:                     float = params['firm']['sigma']*np.sqrt(self.dt)
        self.sigma_p:                   float = params['firm']['sigma_p']*np.sqrt(self.dt)
        self.sigma_w:                   float = params['firm']['sigma_w']*np.sqrt(self.dt)
        self.depreciation:              float = params['firm']['depreciation']*self.dt
        self.num_banks:                 int   = params['firm']['num_banks']
        self.deposit_interest:          float = params['bank']['deposit_interest']*self.dt
        self.loan_periods:              int   = params['bank']['loan_years']*self.steps
        self.repayment_rate:            float = 1/self.loan_periods
        self.bankrupt:                  bool  = False
        # Mutable Data      
        self.employees:                 list[Household] = []
        self.applications:              list[Household] = []
        # Timeseries Data   
        self.productivity:              NDArray = np.zeros(shape=self.time)
        self.productivity_growth:       NDArray = np.zeros(shape=self.time)
        self.expected_productivity:     NDArray = np.zeros(shape=self.time)
        self.output:                    NDArray = np.zeros(shape=self.time)
        self.output_growth:             NDArray = np.zeros(shape=self.time)
        self.desired_output:            NDArray = np.zeros(shape=self.time)
        self.demand:                    NDArray = np.zeros(shape=self.time)
        self.expected_demand:           NDArray = np.zeros(shape=self.time)
        self.quantity:                  NDArray = np.zeros(shape=self.time)
        self.inventories:               NDArray = np.zeros(shape=self.time)
        self.desired_inventories:       NDArray = np.zeros(shape=self.time)
        self.labour:                    NDArray = np.zeros(shape=self.time)
        self.desired_labour:            NDArray = np.zeros(shape=self.time)
        self.labour_demand:             NDArray = np.zeros(shape=self.time)
        self.vacancies:                 NDArray = np.zeros(shape=self.time)
        self.wage:                      NDArray = np.zeros(shape=self.time)
        self.wage_bill:                 NDArray = np.zeros(shape=self.time)
        self.price:                     NDArray = np.zeros(shape=self.time)
        self.profits:                   NDArray = np.zeros(shape=self.time)
        self.profit_share:              NDArray = np.zeros(shape=self.time)
        self.equity:                    NDArray = np.zeros(shape=self.time)
        self.deposits:                  NDArray = np.zeros(shape=self.time)
        self.desired_loan:              NDArray = np.zeros(shape=self.time)
        self.debt:                      NDArray = np.zeros(shape=self.time)
        self.total_repayment:           NDArray = np.zeros(shape=self.time)
        self.total_interest:            NDArray = np.zeros(shape=self.time)
        self.leverage:                  NDArray = np.zeros(shape=self.time)
        self.probability_default:       NDArray = np.zeros(shape=self.time)
        self.age:                       NDArray = np.zeros(shape=self.time)
        self.market_share:              NDArray = np.zeros(shape=self.time)
        self.assets:                    NDArray = np.zeros(shape=self.time)
        self.liabilities:               NDArray = np.zeros(shape=self.time)
        # accounts      
        self.loans:                     NDArray = np.zeros(shape=self.time)
        self.repayment:                 NDArray = np.zeros(shape=self.time)
        self.interest:                  NDArray = np.zeros(shape=self.time)
        self.bank_ids:                  NDArray = np.zeros(shape=self.time)
        # Initial values        
        self.output[:]                  = initial_output
        self.desired_output[0]          = initial_output
        self.demand[0]                  = initial_output
        self.expected_demand[0]         = initial_output
        self.productivity[:]            = params['firm']['productivity']
        self.expected_productivity[0]   = params['firm']['productivity']
        self.desired_labour[0]          = initial_output/self.productivity[0]
        self.price[0]                   = params['firm']['price']
        self.wage[0]                    = initial_wage
        self.bank_ids[:]                = np.nan

    def determine_vacancies(self, t: int) -> None:
        """
        Calculate vacancies.
        
        Parameters
        ----------
            t : int
                time period
        """
        # reset applications list
        self.applications = []
        # calculate total employees/labour
        self.labour[t] = len(self.employees)
        # new labour demand
        self.labour_demand[t] = self.desired_labour[t-1] - self.labour[t]
        # number of vacancies
        self.vacancies[t] = max(self.labour_demand[t], 0)
        
    def determine_wages(self, avg_wage: float, t: int) -> None:
        """
        Calculate wage rate. 
        
        Parameters
        ----------
            avg_wage : float
                average market wage rate
            
            t : int
                time period
        """
        # update wage rate
        if self.labour_demand[t] >= 0:
            self.wage[t] = self.wage[t-1]*(1 + self.sigma_w*abs(np.random.randn())) + self.adjust*(avg_wage - self.wage[t-1])
        else:
            self.wage[t] = self.wage[t-1]*(1 - self.sigma_w*abs(np.random.randn())) + self.adjust*(avg_wage - self.wage[t-1])

    def determine_labour(self, t: int) -> None:
        """
        Hire or fire employees.
        
        Parameters
        ----------
            t : int
                time period
        """
        # firm hires new employees
        if self.labour_demand[t] > 0 and len(self.applications) > 0:
            # number of employees to hire
            num_households_hired = min(int(abs(self.labour_demand[t])), len(self.applications))
            # randomly choose household to hire from applications list
            hire_household_indices: list[np.int32] = list(np.random.choice(np.arange(len(self.applications)), size=num_households_hired, replace=False)) 
            # hire households from applications 
            for i in hire_household_indices:
                # household at index i
                household = self.applications[i]
                # if housheold is unemployed 
                if not household.employed:
                    # hire household
                    self.hire(household)
        # firm fires current employees 
        elif self.labour_demand[t] < 0:
            # number of employees to fire
            num_households_fired = min(int(abs(self.labour_demand[t])), len(self.employees)-1)
            # fire households from employees
            for _ in range(num_households_fired):
                # random household index
                index: np.int32 = np.random.choice(np.arange(len(self.employees)))
                # household at index i
                household = self.employees[index]
                # fire household
                self.fire(household)

    def determine_productivity(self, t: int) -> None:
        """
        Calculate productivity.
        
        Parameters
        ----------
            t : int
                time period
        """
        # update productivity
        self.productivity[t] = self.productivity[t-1]*np.exp(self.growth - 0.5*(self.sigma**2) + self.sigma*np.random.randn())
        # annual productivity growth as the difference in the natural log
        self.productivity_growth[t] = np.log(self.productivity[t]) - np.log(self.productivity[t-self.steps])
        # expected productivity in the next period
        self.expected_productivity[t] = self.productivity[t]*np.exp(self.growth)
    
    @abstractmethod
    def determine_output(self, t: int) -> None:
        """
        Calculate output.
        
        Parameters
        ----------
            t : int
                time period
        """
        # To be overwritten
        pass

    @abstractmethod
    def determine_inventories(self, t: int) -> None:
        """
        Calculate inventories.
        
        Parameters
        ----------
            t : int
                time period
        """
        # To be overwritten
        pass 
    
    def determine_market_share(self, market_output: float, t: int) -> None:
        """
        Calculate market share.
        
        Parameters
        ----------
            market_output : float
                total market output (consumption or investment)
        
            t : int
                time period
        """
        if market_output != 0:
            self.market_share[t] = self.output[t]/market_output

    def determine_output_growth(self, t: int) -> None:
        """
        Calculate output growth rate.
        
        Parameters
        ----------
            t : int
                time period
        """
        # annual output growth as the difference in the natural log
        if self.output[t-self.steps] > 0 and self.age[t-1] > 1:
            self.output_growth[t] = np.log(self.output[t]) - np.log(self.output[t-self.steps])

    def determine_costs(self, t: int) -> None:
        """
        Calculate firm debt, total repayment, total interest, wage bill and pay employees.
        
        Parameters
        ----------
            t : int
                time period 
        """
        # compute total debt owed to all banks
        self.debt[t] = self.compute_total_debt(t)
        # compute total repayment costs
        self.total_repayment[t] = self.compute_total_repayment()
        # compute total interest costs
        self.total_interest[t] = self.compute_total_interest()
        # compte total wage bill
        self.wage_bill[t] = self.wage[t]*self.labour[t]
        # pay employees
        for household in self.employees:
            household.wage[t] = self.wage[t]

    def determine_prices(self, avg_price: float, t: int) -> None:
        """
        Calculate prices.
        
        Parameters
        ----------
            t : int
                time period
        """
        # update prices
        if self.desired_inventories[t-1] >= self.inventories[t-1]:
            self.price[t] = self.price[t-1]*(1 + self.sigma_p*abs(np.random.randn())) + self.adjust*(avg_price - self.price[t-1])
        else:
            self.price[t] = self.price[t-1]*(1 - self.sigma_p*abs(np.random.randn())) + self.adjust*(avg_price - self.price[t-1])

    def determine_profits(self, t: int) -> None:
        """
        Calculate profits and profit share.
        
        Parameters
        ----------
            t : int
                time period
        """
        # calculate profits
        self.profits[t] = self.price[t]*self.quantity[t] + self.deposit_interest*self.deposits[t-1] - self.wage_bill[t] - self.total_interest[t]
        # calculate profit share of revenue
        if self.output[t] != 0:
            self.profit_share[t] = self.profits[t]/(self.price[t]*self.output[t])

    def determine_equity(self, t: int) -> None:
        """
        Calculate equity.
        
        Parameters
        ----------
            t : int 
                time period
        """
        # update equity with profits
        self.equity[t] = self.equity[t-1] + self.profits[t]

    def determine_expected_demand(self, t: int) -> None:
        """
        Calculate expected demand.
        
        Parameters
        ----------
            t : int 
                time period
        """
        # update expected demand in next period
        self.expected_demand[t] = self.expected_demand[t-1] + self.adapt*(self.demand[t] - self.expected_demand[t-1])

    @abstractmethod
    def determine_desired_output(self, t: int) -> None:
        """
        Calculate desired output.
        
        Parameters
        ----------
            t : int
                time period
        """
        # To be overwritten
        pass

    @abstractmethod
    def determine_desired_labour(self, t: int) -> None:
        """
        Calculate desired labour.
        
        Parameters
        ----------
            t : int 
                time period
        """
        # To be overwritten
        pass

    @abstractmethod
    def determine_desired_loan(self, t: int) -> None:
        """
        Calculate desired loan.
        
        Parameters
        ----------
            t : int
                time period
        """
        # To be overwritten
        pass

    def determine_leverage(self, t) -> None:
        """
        Calculate leverage ratio.
        
        Parameters
        ----------
            t : int
                time period
        """
        # expected debt in next period
        expected_debt = self.debt[t]*(1 - self.repayment_rate) + self.desired_loan[t]
        if self.deposits[t-1] + self.profits[t] > 0:
            # leverage ratio between (0,1)
            self.leverage[t] = expected_debt/(self.deposits[t-1] + self.profits[t] + expected_debt)
        else:
            self.leverage[t] = 1
    
    def determine_loan(self, banks: list[Bank], probabilities: list[float], t: int) -> None:
        """
        Visit banks and demand loans, banks supply loans based on their credit risk.
        
        Parameters
        ----------
            banks : list[Bank]
                list of all banks
            
            probabilities : list[float]
                probability of visiting a given banks 
            
            t : int
                time period
        """
        # visit banks for a new loan if desired loan is positive
        if self.desired_loan[t] > 0:
            # randomly visit banks
            visited_bank_indices: list[np.int32] = list(np.random.choice(np.arange(len(banks)), size=self.num_banks, p=probabilities)) 
            # bank with lowest interest rate index
            i = sorted(visited_bank_indices, key=lambda i: banks[i].loan_interest[t], reverse=False)[0]
            # bank at index i
            bank = banks[i]
            # bank computes loan supplied to firm
            loan_supply = bank.compute_loan_supply(self, t)
            # if the banks choses to supply the loan  
            if loan_supply > 0:
                # bank adds firm to loan accounts
                bank.add_loan_firm(self)
                # firm computes new loan costs
                self.compute_new_loan(loan_supply, bank.loan_interest[t], bank.id, t)

    @abstractmethod
    def determine_deposits(self, t: int) -> None:
        """
        Calculate deposits.
                
        Parameters
        ----------
            t : int
                time period
        """
        # To be overwritten
        pass

    def determine_bankruptcy(self, t: int) -> None:
        """
        Calculate age and bankruptcy condition.
        
        Parameters
        ----------
            t : int
                time period
        """
        # increase age by simulation time
        self.age[t] = self.age[t-1] + self.dt
        # bankruptcy condition when deposits are zero or negative
        if self.deposits[t] <= 0.001:
            self.bankrupt = True
    
    @abstractmethod
    def determine_balance_sheet(self, t: int) -> None:
        """
        Calculate firm balance sheet, assets and liabilities.
        
        Parameters
        ----------
            t : int
                time period
        """
        # To be overwritten
        pass

    def compute_total_debt(self, t: int) -> float:
        """
        Calculate total debt.
        
        Parameters
        ----------
            t : int
                time period
        
        Returns
        -------
            total debt : float
        """
        # updates loans and compute total debt
        for tau in range(t):
            # update loan by repayment amount 
            self.loans[tau] = max(self.loans[tau] - self.repayment[tau], 0)
            # if the loan is repaid remove from accounts
            if self.loans[tau] <= 0.001:
                self.repayment[tau] = 0
                self.interest[tau] = 0
                self.bank_ids[tau] = np.nan
        # total debt as sum of all outstanding loans
        return self.loans.sum()
    
    def compute_amortisation(self, loan: float, interest: float) -> float:
        """
        Calculate amortisation cost of a loan
        
        Parameters
        ----------
            loan : float
                loan amount
            
            interest : float
                interest rate on the loan
        
        Returns
        -------
            amortisation cost : float
        """
        # period interest rate 
        period_interest = interest*self.dt
        # amortisation cost each period
        return loan*((period_interest*(1 + period_interest)**self.loan_periods)/((1 + period_interest)**self.loan_periods - 1))

    def compute_new_loan(self, loan: float, interest: float, bank_id: int, t: int) -> None:
        """
        Calculate new loan, repayment cost, interest cost, and bank id.
        
        Parameters
        ---------- 
            loan : float 
                loan amount
            
            interest : float 
                interest rate on the loan
            
            bank_id : int 
                id of the lender bank
            
            t : int 
                time period
        """
        # compute amortisation cost of the new loan
        amortisation = self.compute_amortisation(loan, interest)
        # compute repayment cost of the new loan
        repayment_amount = self.repayment_rate*loan
        # compute interest cost of the new loan
        interest_amount = amortisation - repayment_amount
        # add loan to accounts
        self.loans[t] = loan
        self.repayment[t] = repayment_amount
        self.interest[t] = interest_amount
        self.bank_ids[t] = bank_id
    
    def compute_total_repayment(self) -> float:
        """
        Calculate total repayment cost. 
        
        Returns
        -------
            total repayment cost : float
        """
        return self.repayment.sum()

    def compute_total_interest(self) -> float:
        """
        Calculate total interest cost.
        
        Returns
        -------
            total interest cost : float
        """
        return self.interest.sum()

    def compute_bank_loans(self, bank_id: int) -> float:
        """
        Calculate loan to a specific bank with id bank_id.
        
        Parameters
        ----------
            bank_id : int
                id of lender bank
        
        Returns
        -------
            loans to bank : float
        """
        return np.where(self.bank_ids == bank_id, self.loans, 0).sum() 
        
    def compute_bank_interest(self, bank_id: int) -> float:
        """
        Calculate interest payments to a specific bank with id bank_id.
        
        Parameters
        ----------
            bank_id : int
                id of lender bank
        
        Returns
        -------
            interest to bank : float
        """
        return np.where(self.bank_ids == bank_id, self.interest, 0).sum() 
    
    def hire(self, household: Household) -> None:
        """
        Hire household.
        
        Parameters
        ----------
            housheold : Household
                household to hire
        """
        household.employed = True
        self.employees.append(household)

    def fire(self, household: Household) -> None:
        """
        Fire household
        
        Parameters
        ----------
            household : Household
                household to fire
        """
        household.employed = False
        self.employees.remove(household)
    

class CapitalFirm(Firm):
    
    """
    CapitalFirm Class
    =================
    Capital firm (K-firm) class used to simulate K-firms in the model, inherits from Firm class. 
    
    Attributes
    ----------
        id : int
            bank object id
    
        steps : int
            number of time steps per year
        
        time : int
            total number of time steps
    
        dt : float  
            time delta, inverse steps
        
        adapt : float
            adaptive update speed
        
        adjust : float
            speed of adjustment 
        
        growth : float
            average firm productivity growth
        
        sigma : float
            firm productivity standard deviation
        
        sigma_p : float
            firm price standard deviation
        
        sigma_w : float
            firm wage standard deviation
        
        depreciation : float
            capital good depreciation rate

        num_banks : int
            number of banks visited
    
        deposit_interest : float
            bank deposit interest rate
        
        loan_periods : int
            number of repayment periods for a loan 
        
        repayment_rate : float
            loan repayment rate, inverse loan periods
        
        bankrupt : bool
            bankruptcy condition
            
        excess_output : float
            desired percentage of excess output over demand 
    
    Data
    ----
        
        employees : list[Household]
            household employees
        
        applications : list[Household]
            applications from households
        
        productivity : numpy.array
            labour productivity
        
        productivity_growth : numpy.array
            labour productivity growth rate
        
        expected_productivity : numpy.array
            expected labour productivity 
        
        output : numpy.array
            production output 
        
        output_growth : numpy.array
            output growth rate
        
        desired_output : numpy.array
            desired output 
        
        demand : numpy.array
            demand for goods
        
        expected_demand : numpy.array
            expected demand for goods
        
        quantity : numpy.array
            quantity of goods sold
        
        inventories : numpy.array
            inventories of goods
        
        desired_inventories : numpy.array
            desired inventories 
        
        labour : numpy.array
            number of employees
        
        desired_labour : numpy.array
            desired number of employees
        
        labour_demand : numpy.array
            demand for new employees
        
        vacancies : numpy.array
            number of vacant positions 
        
        wage : numpy.array
            wage rate 
        
        wage_bill : numpy.array
            wage bill (wage x labour)
        
        price : numpy.array
            price of goods
        
        profits : numpy.array
            profits 
        
        profit_share : numpy.array
            profit share of revenue 
        
        equity : numpy.array
            equity 
        
        deposits : numpy.array
            deposits at bank
        
        desired_loan : numpy.array
            desired loan from banks
        
        debt : numpy.array
            debt
        
        total_repayment : numpy.array
            total cost of loan repayment 
        
        total_interest : numpy.array
            total cost of interest payments 
        
        leverage : numpy.array
            leverage ratio
        
        probability_default : numpy.array
            probability of defaulting 
        
        age : numpy.array
            age in years
        
        market_share : numpy.array
            market share 
        
        assets : numpy.ndarray
            value of assets
        
        liabilities : numpy.ndarray
            value of liabilities
        
        loans : numpy.array
            current outstanding loans
        
        repayment : numpy.array
            current repayment amount for each loan
        
        interest : numpy.array
            current interest cost on each loan 
        
        bank_ids : numpy.array
            bank id for each loan
    
    Methods 
    -------
        determine_vacancies(self, t: int) -> None
        
        determine_wages(self, avg_wage: float, t: int) -> None
        
        determine_labour(self, t: int) -> None
        
        determine_productivity(self, t: int) -> None
        
        determine_output(self, t: int) -> None
        
        determine_inventories(self, t: int) -> None
        
        determine_market_share(self, market_output: float, t: int) -> None
        
        determine_output_growth(self, t: int) -> None
        
        determine_costs(self, t: int) -> None
        
        determine_prices(self, avg_price: float, t: int) -> None
        
        determine_profits(self, t: int) -> None
        
        determine_equity(self, t: int) -> None
        
        determine_expected_demand(self, t: int) -> None
        
        determine_desired_output(self, t: int) -> None
        
        determine_desired_labour(self, t: int) -> None
        
        determine_desired_loan(self, t: int) -> None
        
        determine_leverage(self, t) -> None
        
        determine_loan(self, banks: list[Bank], probabilities: list[float], t: int) -> None
        
        determine_deposits(self, t: int) -> None
        
        determine_bankruptcy(self, t: int) -> None
        
        determine_balance_sheet(self, t: int) -> None
        
        compute_total_debt(self, t: int) -> float
        
        compute_amortisation(self, loan: float, interest: float) -> float
        
        compute_new_loan(self, loan: float, interest: float, bank_id: int, t: int) -> None
        
        compute_total_repayment(self) -> float
        
        compute_total_interest(self) -> float
        
        compute_bank_loans(self, bank_id: int) -> float
        
        compute_bank_interest(self, bank_id: int) -> float
        
        hire(self, household: Household) -> None
        
        fire(self, household: Household) -> None
    """
    
    def __init__(self, id: int, initial_output: float, initial_wage: float, params: dict) -> None:
        """
        CapitalFirm class initialisation, inherits from Firm class.
        
        Parameters
        ----------
            id : int
                unique id
        
            init_output : float
                initial output
        
            init_wage : float
                initial wage rate
        
            params : dict
                model parameters
        """
        super().__init__(id, initial_output, initial_wage, params)
        # Parameters
        self.excess_output:           float = params['kfirm']['excess_output']
        # Initial values
        self.market_share[0]          = 1/params['market']['num_kfirms']
        self.desired_inventories[0]   = self.excess_output*initial_output
        self.wage_bill[0]             = self.wage[0]*self.desired_labour[0]
        self.profits[0]               = initial_output - self.wage_bill[0]
        self.profit_share[0]          = self.profits[0]/initial_output
        self.deposits[0]              = self.profits[0]
        self.equity[0]                = self.deposits[0]

    def __repr__(self) -> str:
        """
        Returns printable unique K-firm agent representaton.
        
        Returns
        -------
            representation : str
        """
        return f'Kfirm: {self.id}'
    
    def __str__(self) -> str:
        """ 
        Returns printable agent type.
        
        Returns
        -------
            agent type : str
        """
        return 'CapitalFirm'

    def determine_output(self, t: int) -> None:
        """
        Calculate K-firm output as a linear production function of labour productivity and the number of employees.
        
        Parameters
        ----------
            t : int
                time period
        """
        # calculate total labour 
        self.labour[t] = len(self.employees)
        # output as total labour times the productivity they make goods
        self.output[t] = self.productivity[t]*self.labour[t]

    def determine_inventories(self, t: int) -> None:
        """
        Calculate K-firm inventories.
        
        Parameters
        ----------
            t : int
                time period
        """
        # update total inventories from new output and previous inventories
        self.inventories[t] = self.output[t] + (1 - self.depreciation)*self.inventories[t-1]
        # desired level of inventories as a ration of current output
        self.desired_inventories[t] = self.output[t]*self.excess_output
    
    def determine_desired_output(self, t: int) -> None:
        """
        Calculate K-firm desired output.
        
        Parameters
        ----------
            t : int
                time period
        """
        # desired output next period
        self.desired_output[t] = max(self.expected_demand[t]*(1 + self.excess_output) - (1 - self.depreciation)*self.inventories[t], self.expected_productivity[t])
    
    def determine_desired_labour(self, t: int) -> None:
        """
        Calculate K-firm desired labour.
        
        Parameters
        ----------
            t : int
                time period
        """
        # desired amount of labour next period
        self.desired_labour[t] = int(self.desired_output[t]/self.expected_productivity[t])

    def determine_desired_loan(self, t: int) -> None:
        """
        Calculate K-firm desired loan.
        
        Parameters
        ----------
            t : int
                time period
        """
        # desired loan next period
        self.desired_loan[t] = max(self.wage_bill[t] - self.profits[t] - self.deposits[t-1], 0)
        
    def determine_deposits(self, t: int) -> None:
        """
        Calculate K-firm deposits.
        
        Parameters
        ----------
            t : int
                time period
        """
        # update deposits by accounting identity 
        self.deposits[t] = self.deposits[t-1] + self.profits[t] + self.loans[t] - self.total_repayment[t]
        
    def determine_balance_sheet(self, t: int) -> None:
        """
        Calculate K-firm balance sheet, assets and liabilities.
        
        Parameters
        ----------
            t : int
                time period
        """
        # calculate assets
        self.assets[t] = self.deposits[t]
        # calculate liabilities
        self.liabilities[t] = self.debt[t]


class ConsumptionFirm(Firm):
    
    """
    ConsumptionFirm Class
    =====================
    Conumption firm (C-firm) class used to simulate C-firms in the model, inherits from Firm class. 
    
    Attributes
    ----------
        id : int
            bank object id
    
        steps : int
            number of time steps per year
        
        time : int
            total number of time steps
    
        dt : float  
            time delta, inverse steps
        
        adapt : float
            adaptive update speed
        
        adjust : float
            speed of adjustment 
        
        growth : float
            average firm productivity growth
        
        sigma : float
            firm productivity standard deviation
        
        sigma_p : float
            firm price standard deviation
        
        sigma_w : float
            firm wage standard deviation
        
        depreciation : float
            capital good depreciation rate

        num_banks : int
            number of banks visited
    
        deposit_interest : float
            bank deposit interest rate
        
        loan_periods : int
            number of repayment periods for a loan 
        
        repayment_rate : float
            loan repayment rate, inverse loan periods
        
        bankrupt : bool
            bankruptcy condition
            
        acceleration : float 
            capital acceleration (capital to output ratio)
        
        num_kfirms : int
            number of K-firms visited on the capital market
        
        d0 : float
            desired debt ratio parameter
        
        d1 : float
            desired debt ratio parameter
        
        d2 : float
            desired debt ratio parameter
    
    Data
    ----
        
        employees : list[Household]
            household employees
        
        applications : list[Household]
            applications from households
        
        productivity : numpy.array
            labour productivity
        
        productivity_growth : numpy.array
            labour productivity growth rate
        
        expected_productivity : numpy.array
            expected labour productivity 
        
        output : numpy.array
            production output 
        
        output_growth : numpy.array
            output growth rate
        
        desired_output : numpy.array
            desired output 
        
        demand : numpy.array
            demand for goods
        
        expected_demand : numpy.array
            expected demand for goods
        
        quantity : numpy.array
            quantity of goods sold
        
        inventories : numpy.array
            inventories of goods
        
        desired_inventories : numpy.array
            desired inventories 
        
        labour : numpy.array
            number of employees
        
        desired_labour : numpy.array
            desired number of employees
        
        labour_demand : numpy.array
            demand for new employees
        
        vacancies : numpy.array
            number of vacant positions 
        
        wage : numpy.array
            wage rate 
        
        wage_bill : numpy.array
            wage bill (wage x labour)
        
        price : numpy.array
            price of goods
        
        profits : numpy.array
            profits 
        
        profit_share : numpy.array
            profit share of revenue 
        
        equity : numpy.array
            equity 
        
        deposits : numpy.array
            deposits at bank
        
        desired_loan : numpy.array
            desired loan from banks
        
        debt : numpy.array
            debt
        
        total_repayment : numpy.array
            total cost of loan repayment 
        
        total_interest : numpy.array
            total cost of interest payments 
        
        leverage : numpy.array
            leverage ratio
        
        probability_default : numpy.array
            probability of defaulting 
        
        age : numpy.array
            age in years
        
        market_share : numpy.array
            market share 
        
        assets : numpy.ndarray
            value of assets
        
        liabilities : numpy.ndarray
            value of liabilities
        
        loans : numpy.array
            current outstanding loans
        
        repayment : numpy.array
            current repayment amount for each loan
        
        interest : numpy.array
            current interest cost on each loan 
        
        bank_ids : numpy.array
            bank id for each loan

        capital : numpy.array
            capital stock
        
        capital_cost : numpy.array
            cost of capital stock
        
        desired_utilisation : numpy.array
            desired capital utilisation rate
        
        investment : numpy.array
            investment in capital 
        
        investment_cost : numpy.array
            cost of investment in capotal
        
        desired_investment_cost : numpy.array
            desired cost of investment in capital
        
        desired_investment_loan : numpy.array
            desired loan for investment in capital
        
        desired_debt_ratio : numpy.array
            desired debt ratio for investment in capital
        
        desired_debt : numpy.array
            desired nominal debt for investment in capital
    
    Methods 
    -------
        determine_vacancies(self, t: int) -> None
        
        determine_wages(self, avg_wage: float, t: int) -> None
        
        determine_labour(self, t: int) -> None
        
        determine_productivity(self, t: int) -> None
        
        determine_output(self, t: int) -> None
        
        determine_inventories(self, t: int) -> None
        
        determine_market_share(self, market_output: float, t: int) -> None
        
        determine_output_growth(self, t: int) -> None
        
        determine_costs(self, t: int) -> None
        
        determine_prices(self, avg_price: float, t: int) -> None
        
        determine_profits(self, t: int) -> None
        
        determine_equity(self, t: int) -> None
        
        determine_expected_demand(self, t: int) -> None
        
        determine_desired_output(self, t: int) -> None
        
        determine_desired_labour(self, t: int) -> None
        
        determine_desired_loan(self, t: int) -> None
        
        determine_leverage(self, t) -> None
        
        determine_loan(self, banks: list[Bank], probabilities: list[float], t: int) -> None
        
        determine_deposits(self, t: int) -> None
        
        determine_bankruptcy(self, t: int) -> None
        
        determine_balance_sheet(self, t: int) -> None
        
        compute_total_debt(self, t: int) -> float
        
        compute_amortisation(self, loan: float, interest: float) -> float
        
        compute_new_loan(self, loan: float, interest: float, bank_id: int, t: int) -> None
        
        compute_total_repayment(self) -> float
        
        compute_total_interest(self) -> float
        
        compute_bank_loans(self, bank_id: int) -> float
        
        compute_bank_interest(self, bank_id: int) -> float
        
        hire(self, household: Household) -> None
        
        fire(self, household: Household) -> None
        
        determine_investment(self, kfirms: list[CapitalFirm], probabilities: list, t: int) -> None
    """
    
    def __init__(self, id: int, initial_output: float, initial_wage: float, params: dict) -> None:
        """
        CapitalFirm class initialisation, inherits from Firm class.
        
        Parameters
        ----------
            id : int
                unique id
        
            init_output : float
                initial output
        
            init_wage : float
                initial wage rate
        
            params : dict
                model parameters
        """
        super().__init__(id, initial_output, initial_wage, params)
        # Parameters
        self.acceleration:              float = params['cfirm']['acceleration']
        self.num_kfirms:                int   = params['cfirm']['num_kfirms']
        self.d0:                        float = params['cfirm']['d0']
        self.d1:                        float = params['cfirm']['d1']
        self.d2:                        float = params['cfirm']['d2']
        # Data  
        self.capital:                   NDArray = np.zeros(shape=self.time)
        self.capital_cost:              NDArray = np.zeros(shape=self.time)
        self.desired_utilisation:       NDArray = np.zeros(shape=self.time)
        self.investment:                NDArray = np.zeros(shape=self.time)
        self.investment_cost:           NDArray = np.zeros(shape=self.time)
        self.desired_investment_cost:   NDArray = np.zeros(shape=self.time)
        self.desired_investment_loan:   NDArray = np.zeros(shape=self.time)
        self.desired_debt_ratio:        NDArray = np.zeros(shape=self.time)
        self.desired_debt:              NDArray = np.zeros(shape=self.time)
        # Initial values
        self.market_share[0]            = 1/params['market']['num_cfirms']
        self.capital[0]                 = initial_output*self.acceleration
        self.desired_debt_ratio[0]      = (self.d0 + self.d1*self.growth*self.steps + self.d2*self.acceleration*(self.growth*self.steps + self.depreciation*self.steps))/(1 + self.d2*self.growth*self.steps)
        self.loans[0]                   = self.desired_debt_ratio[0]*initial_output
        self.repayment[0]               = self.repayment_rate*self.loans[0]
        self.interest[0]                = self.compute_amortisation(self.loans[0], params['bank']['loan_interest']) - self.repayment[0]
        self.debt[0]                    = self.loans[0]
        self.total_interest[0]          = self.compute_total_interest()
        self.total_repayment[0]         = self.compute_total_repayment()
        self.profit_share[0]            = self.acceleration*(self.growth*self.steps + self.depreciation*self.steps) - self.growth*self.steps*self.desired_debt_ratio[0]
        self.profits[0]                 = self.profit_share[0]*initial_output
        self.wage_bill[0]               = self.wage[0]*initial_output
        self.deposits[0]                = self.profits[0] + self.debt[0]
        self.equity[0]                  = self.capital[0] + self.deposits[0] - self.debt[0]
        self.leverage[0]                = self.debt[0]/(self.equity[0] + self.debt[0])

    def __repr__(self) -> str:
        """
        Returns printable unique C-firm agent representaton.
        
        Returns
        -------
            representation : str
        """
        return f'Cfirm: {self.id}'
    
    def __str__(self) -> str:
        """ 
        Returns printable agent type.
        
        Returns
        -------
            agent type : str
        """
        return 'ConsumptionFirm'
        
    def determine_output(self, t: int) -> None:
        """
        Calculate C-firm output as a Leonteif production function between labour and capital.
        
        Parameters
        ----------
            t : int
                time period
        """
        # calculate total labour
        self.labour[t] = len(self.employees)
        # output as minimum of labour times productivity or capital times productivity (1/accelation)
        self.output[t] = min(self.productivity[t]*self.labour[t], self.capital[t-1]/self.acceleration)

    def determine_inventories(self, t: int) -> None:
        """
        Calculate C-firm inventories.
        
        Parameters
        ----------
            t : int
                time period
        """
        # calculate inventories for current period
        self.inventories[t] = self.output[t]

    def determine_desired_output(self, t: int) -> None:
        """
        Calculates C-firm desired output.
        
        Parameters
        ----------
            t : int
                time period
        """
        # desired output next period
        self.desired_output[t] = self.expected_demand[t]
    
    def determine_investment(self, kfirms: list[CapitalFirm], probabilities: list, t: int) -> None:
        """
        C-firms visit K-firms with probability given by probabilities list and place investment orders for capital goods.
        
        Parameters
        ----------
            kfirms : list[CapitalFirm]  
                list of K-firms in the market
                
            probabilities : list[float] 
                list of probabilities that C-firm visits a given K-firm 

            t : int
                time period
        """
        # calculate desired debt ratio
        self.desired_debt_ratio[t] = self.d0 + self.d1*self.productivity_growth[t] + self.d2*self.profit_share[t]
        # calculate desired debt level next period
        self.desired_debt[t] = self.output[t]*self.price[t]*self.desired_debt_ratio[t]
        # caluclate desired investment loan
        self.desired_investment_loan[t] = max(self.desired_debt[t] - self.debt[t], 0)
        # calulate desired investment cost
        self.desired_investment_cost[t] = max(self.desired_investment_loan[t] + self.profits[t] + self.deposits[t-1] - self.wage_bill[t], 0)
        # if desired investment cost
        if self.desired_investment_cost[t] > 0:
            # randomly visits kfirms (index)
            visited_kfirm_indices: list[np.int32] = list(np.random.choice(np.arange(len(kfirms)), size=self.num_kfirms, replace=False, p=probabilities)) 
            # sorts cfirms indices by price
            sorted_visited_kfirm_indices = sorted(visited_kfirm_indices, key=lambda i: kfirms[i].price[t], reverse=False)
            # order capital goods from kfirms
            for i in sorted_visited_kfirm_indices:
                # kfirm at index i
                kfirm = kfirms[i]
                # amount of goods demanded
                goods_demanded = self.desired_investment_cost[t]/kfirm.price[t]
                # amount of goods purchased, constrained by kfirm inventories
                goods_purchased = min(kfirm.inventories[t], goods_demanded)
                # cost of purchased goods
                goods_cost = goods_purchased*kfirm.price[t]
                # update kfirm demand
                kfirm.demand[t] += goods_demanded
                # update kfirm quantity sold
                kfirm.quantity[t] += goods_purchased
                # update kfirm inventories
                kfirm.inventories[t] -= goods_purchased
                # update investment amount
                self.investment[t] += goods_purchased
                # update investment cost
                self.investment_cost[t] += goods_cost
                # reduce desired investment
                self.desired_investment_cost[t] -= goods_cost
                # cfirm stops ordering investment goods if desired investment is exceeded
                if self.desired_investment_cost[t] <= 0.001:
                    break
        # update capital stock
        self.capital[t] = self.capital[t-1]*(1 - self.depreciation) + self.investment[t]
        self.capital_cost[t] = self.capital_cost[t-1]*(1 - self.depreciation) + self.investment_cost[t]

    def determine_desired_labour(self, t: int) -> None:
        """
        Calculate C-firm desired labour.
        
        Parameters
        ----------
            t : int
                time period
        """
        # calculate desired capital utilisation rate
        self.desired_utilisation[t] = min(1, (self.acceleration*self.desired_output[t])/self.capital[t])
        # calculate desired investment
        self.desired_labour[t] = int(self.desired_utilisation[t]*(self.capital[t]/(self.expected_productivity[t]*self.acceleration)))
    
    def determine_desired_loan(self, t: int) -> None:
        """
        Calculate C-firm desired loan.
        
        Parameters
        ----------
            t : int
                time period
        """
        # calculate desired loan
        self.desired_loan[t] = max(self.investment_cost[t] + self.wage_bill[t] - self.profits[t] - self.deposits[t-1], 0)

    def determine_deposits(self, t: int) -> None:
        """
        Calculate C-firm deposits.
        
        Parameters
        ----------
            t : int
                time period
        """
        # update deposits by accounting identity 
        self.deposits[t] = self.deposits[t-1] + self.profits[t] + self.loans[t] - self.investment_cost[t] - self.total_repayment[t]
        
    def determine_balance_sheet(self, t: int) -> None:
        """
        Calculate C-firm balance sheet, assets and liabilities.
        
        Parameters
        ----------
            t : int
                time period
        """
        # calculate assets
        self.assets[t] = self.deposits[t] + self.capital_cost[t]
        # calculate liabilities
        self.liabilities[t] = self.debt[t]