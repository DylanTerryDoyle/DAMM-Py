import sqlite3
from market import Market
from utils import * 

def main():
    """
    Run this file to run the main function which starts the macro ABM simulation.
    """
    # load model parameters
    params = load_parameters('parameters.yaml')
    # create market object
    market = Market(params)
    # connect/create database
    try:
        # connect/create specified database path and name
        conn = sqlite3.connect(f"{params['database_path']}\\{params['database_name']}.db")
    except sqlite3.OperationalError:
        # connect/create to database called data in current src folder if database details are not given
        conn = sqlite3.connect(f"data.db")
    # create cursor 
    cur = conn.cursor()
    # run simulation
    market.run_simulation(cur)
    # close database connection 
    cur.close()
    conn.commit()
    conn.close()

# run main function
if __name__ == '__main__':
    main()
