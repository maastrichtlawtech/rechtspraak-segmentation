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
    :param file_path: The path to the file to load.
    :return: A Pandas DataFrame with correct datatypes
    """
    df = pd.read_csv(file_path)
    df = df.applymap(safe_literal_eval)
    return df

def load_txt_file(file_path: str):
    """
    Loads and processes text patterns from a specified file.
    This function reads a text file line by line, strips each line of leading and trailing whitespace,
    and removes the first two and the last character from each line. It also replaces any double
    backslashes ('\\\\') with a single backslash ('\\') in the processed lines.
    :param file_path: The path to the file to load.
    :return:  A list of processed string patterns from the file.
    """
    # Load patterns from the file
    txt_file = open(file_path, 'r')
    split_lines = txt_file.readlines()

    # Process each line: strip whitespaces, remove the first two and last characters,
    # and replace double backslashes with a single backslash.
    split_patterns = [line.strip()[2:-1].replace('\\\\', '\\') for line in split_lines]
    return split_patterns