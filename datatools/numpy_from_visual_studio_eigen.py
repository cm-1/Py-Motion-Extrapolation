
#%%
# Generated using Olmo 3.1 32B. Still need to fiddle manually with the regex.
# Converts values copied from Eigen matrices in Visual Studio's debugger into
# numpy matrices, which are easier to do some testing on. 
import re
import numpy as np

def eigen_debug_to_numpy(text_or_path=None, shape=None):
    """
    Extracts floating-point numbers from Eigen debug output.

    Args:
        text_or_path (str or None): Either a string containing the debug output,
                                     or a path to a file with the output.
                                     If None, attempts to read from clipboard.
        shape (tuple, optional): Desired shape for the output numpy array.

    Returns:
        numpy.ndarray: 1D array of floats (or reshaped if `shape` provided)
    """
    # Get the text from various sources
    if text_or_path is None:
        # Try to get from clipboard
        try:
            import tkinter
            from tkinter import Tk
            root = Tk()
            root.withdraw()
            text = root.clipboard_get()
            root.destroy()
        except Exception:
            raise ValueError("Could not get text from clipboard. Provide text or a file path.")
    elif isinstance(text_or_path, str):
        # Assume it's a file path if it looks like one, otherwise treat as text
        import os
        if os.path.isfile(text_or_path):
            with open(text_or_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            text = text_or_path
    else:
        raise TypeError("text_or_path must be a string (text or file path) or None (use clipboard).")

    # print("Working with text:", text)

    # Extract all floats using regex (handles scientific notation)
    # This regex matches numbers like 202186.766, 0.00000000, 8.13220659e+09, -5.16269824e+09
    numbers = re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?', text)
    print("numbers:", len(numbers), numbers)

    # Convert to floats
    try:
        floats = [float(x) for x in numbers]
    except ValueError as e:
        raise ValueError(f"Could not convert all matches to float: {e}")

    arr = np.array(floats, dtype=np.float64)

    # Reshape if shape provided
    if shape is not None:
        if len(floats) != np.prod(shape):
            raise ValueError(f"Number of extracted floats ({len(floats)}) does not match desired shape {shape}")
        arr = arr.reshape(shape)

    return arr