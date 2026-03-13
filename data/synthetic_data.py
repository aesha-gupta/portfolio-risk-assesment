import pandas as pd

def load_synthetic_data():
    """
    Loads the synthetic portfolio dataset
    and returns it as a pandas DataFrame.
    """
    # read the CSV file
    df = pd.read_csv("data/portfolio_dataset.csv")
    # display basic information
    print("Synthetic dataset loaded successfully")
    print("Number of rows:", df.shape[0])             # print the number of rows 
    print("Number of columns:", df.shape[1])          # print the number of columns 
    return df