import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib
import numpy as np
import datetime

def read_data_from_csv(*paths):
    assert len(paths) in [1, 2]
    data = pd.read_csv(path[0])
    prices = pd.read_csv(path[1]) if len(path) == 2

    return data, prices



if __name__ == '__main__':
    data_path = 
    read_data_from_csv('1','2')
