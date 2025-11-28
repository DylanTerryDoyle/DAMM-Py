import yaml
import numpy as np
from numba import njit
from pathlib import Path

def load_yaml(file: Path | str):
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
