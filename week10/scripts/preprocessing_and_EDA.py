import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def preprocessing(df):

    """
    Preprocess the data: handle missing values and ensure correct types.
    
    Args:
        df (pd.DataFrame): DataFrame containing the oil prices.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """

    df.dropna(inplace=True) # removes missing values from the original data

    df['Price'] = df['Price'].astype(float)

    # Replace outliers with median
    median = np.median(df['Price'])
    threshold = 3 * np.std(df['Price'])
    df['Price'] = np.where(np.abs(df['Price'] - median) > threshold, median, df['Price'])
    
    return df

def EDA(df):

    """
    Plot the trends of Brent oil prices over time.
    
    Args:
        df (pd.DataFrame): DataFrame containing the oil prices.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(df['Date'], df['Price'], label='Brent Oil Price')
    plt.title('Brent Oil Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price (USD per barrel)')
    plt.legend()
    plt.show()

    # Frequency with the Date
    plt.figure(figsize=(14, 7))
    plt.xlabel('Date')
    plt.ylabel('Frequency')
    plt.hist(df)    