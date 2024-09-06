import ast
import pandas as pd


def safe_literal_eval(val: str) -> any:
    """
    Safely evaluate a string literal to a Python object to infer the correct datatype. Returns original value if error
    is raised.
    :param val: The string to be evaluated.
    :return: The evaluated Python object, or the original string if evaluation fails.
    """
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val


def load_csv_to_df(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file into a Pandas DataFrame with safe literal evaluation to ensure correct datatypes.
    :param file_path: The path to the file to laod.
    :return: A Pandas DataFrame with correct datatypes
    """
    df = pd.read_csv(file_path)
    df = df.applymap(safe_literal_eval)
    return df
