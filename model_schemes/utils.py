import os
import pandas as pd

def get_file_paths(directory = ".", extension = ".csv"):
    """
    Get all the file paths in the given directory.
    
    Parameters
    ----------
    directory : str
        The directory to search for files.
        
    Returns
    -------
    list
        A list of file paths in the given directory.
    """
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                file_paths.append(os.path.join(root, file))
    return file_paths

def load_dfs_by_source(data_dir:str,data_source:str)-> pd.DataFrame:
    """
    Load all dfs by source ('valve1','valve2','other')

    Args:
        data_dir (str): _description_
        data_source (str): _description_

    Returns:
        pd.DataFrame: _description_
    """
    all_files=[]
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".csv"):
                all_files.append(os.path.join(root, file))
    all_files.sort()
    
    
    dfs_list = [pd.read_csv(file, 
                          sep=';', 
                          index_col='datetime', 
                          parse_dates=True) for file in all_files if data_source in file]
    
    return (
        pd.concat(dfs_list)
        # .drop(columns=['changepoint'])
        .sort_index()
    )
