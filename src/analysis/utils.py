import numpy as np


def np_string_to_array(npstring: str) -> np.ndarray:
    """Convert a numpy array string back to numpy array.

    Args:
        npstring (str): numpy array in string format to be saved
            in a pd.DataFrame; e.g. [1 0 1 0 0 0 1 0 1].

    Returns
        np.ndarray: numpy array of the string representation.
    """
    # remove braces
    clean_string = npstring.strip()[1:-1]

    # create np array
    return np.fromstring(clean_string, dtype=int, sep=" ")


def list_string_to_array(npstring: str) -> np.ndarray:
    """Convert a numpy array string back to numpy array.

    Args:
        npstring (str): numpy array in string format to be saved
            in a pd.DataFrame; e.g. [1 0 1 0 0 0 1 0 1].

    Returns
        np.ndarray: numpy array of the string representation.
    """
    # remove braces
    clean_string = npstring.strip()[1:-1]

    # create np array
    return np.fromstring(clean_string, dtype=int, sep=", ")
