import pandas as pd

def get_brent_data():
    # Load the dataset
    df = pd.read_csv('../../Data/BrentOilPrices.csv')  
    # Process the data as needed
    return df.to_dict(orient='records')  # Convert to a dictionary for JSON response