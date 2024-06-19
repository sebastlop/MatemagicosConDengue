import pandas as pd
import numpy as np

def read_table():
    dataframe = pd.read_excel('./weekly_data.xlsx')
    values = dataframe.values
    return values

if __name__ == '__main__':
    print(read_table().shape)