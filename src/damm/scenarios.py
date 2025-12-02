
from pathlib import Path
from damm.model import Model
from damm.utils import load_yaml

def main():
    """
    Run this file to run the main function which runs the 4 scenarios of the DAMM model.
    
    Growth Scenario 1: low debt 
        - 2% productivity growth => growth = 0.02 
        - low debt volatility => d1 = 3 & d2 = 2
        
    Growth scenario 2: high debt
        -  2% productivity growth => growth = 0.02
        - high debt volatility => d1 = 5 & d2 = 3
        
    Zero-Growth Scenario 1: low debt 
        - 0% productivity growth => growth = 0.0
        - low debt volatility => d1 = 3 & d2 = 2
    
    Zero-Growth Scenario 2: high debt 
        - 0% productivity growth => growth = 0.0
        - high debt volatility => d1 = 5 & d2 = 3
    """

    ### paths ###

    # current working directory
    cwd_path = Path.cwd()
    
    # parameters path 
    params_path = cwd_path / "src" / "config" / "parameters.yaml"


    ### Model Parameters ###
    params = load_yaml(params_path)
    
    ### Start Scenario Simulations ###
    
    for i, scenario in enumerate(params["scenarios"]):
        print(f"Running scenario {scenario}")
        # scenario specific parameters
        scenario_params = params.copy()
        scenario_params["firm"]["growth"] = params["firm"]["growth"][i]
        scenario_params["cfirm"]["d1"] = scenario_params["cfirm"]["d1"][i]
        scenario_params["cfirm"]["d2"] = scenario_params["cfirm"]["d2"][i]
        
        # instantiate model object for scenario
        model = Model(scenario, scenario_params)
        
        # run model
        model.run(init_seed=0)

# run main function
if __name__ == '__main__':
    main()
