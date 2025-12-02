### Import Libraries
import numpy as np
from numpy.typing import NDArray

### Import Parent AbstractFirm for Inheritance ###
from damm.agents.abstract_firm import AbstractFirm

### Import Agents for Typing ###
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from damm.agents.capital_firm import CapitalFirm

### Conumption Firm Class ###
class ConsumptionFirm(AbstractFirm):
    
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
    
    def determine_investment(self, kfirms: list['CapitalFirm'], probabilities: list, t: int) -> None:
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