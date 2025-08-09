import pandas as pd

def load_data(df):
    """
    Load the Brent oil prices data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: DataFrame containing the oil prices.
    """
    return pd.read_csv(df, parse_dates=['Date'], dayfirst=True)