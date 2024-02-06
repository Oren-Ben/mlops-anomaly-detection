import os

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
