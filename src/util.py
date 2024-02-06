import os
dirname = os.path.dirname(__file__)

def get_file_paths(path_to_files = ".", extension = ".csv"):
    """
    Get all the file paths in the given directory.
    
    Parameters
    ----------
    path_to_files : str
        The path to search for files.
        
    Returns
    -------
    list
        A list of file paths in the given directory.
    """
    file_paths = []
    for root, _, files in os.walk(os.path.join(dirname,path_to_files)):
        for file in files:
            if file.endswith(extension):
                file_paths.append(os.path.join(root, file))
    return file_paths
