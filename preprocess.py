import pandas as pd
import numpy as np
import datetime
import sys
sys.path.insert(0, '../')

def read_data_from_csv(path: str):

    data = pd.read_csv(path)

    return data

def create_categorical(df, which_columns, one_hot=True):
    '''
    Returns a data frame df and replaces with one-hot encodings the specified columns
    in which_columns
    '''
    df = df.copy()
    columns = list(df.columns)
    assert set(which_columns).issubset(set(columns))
    if one_hot:
        Y = []
        for column in which_columns:
            Y.append(pd.get_dummies(df[column], prefix=column))
            columns.remove(column)
        df = df[columns]
        for y in Y:
            df = pd.concat([df, y], axis=1, sort=False)
        return df
    else:
        for column in which_columns:
            N = sorted(list(df[column].unique()))
            y = {val:int(key) for (key,val) in enumerate(N)}
            new_col = []
            for d in df[column]:
                new_col.append(y[d])
            df[column] = new_col
        return df


if __name__ == '__main__':
    data_path = "C:\Users\GeorgeYiasemis\OneDrive - Tickmill Ltd\Töölaud\Data\APPL_sentiment.csv"
    print(read_data_from_csv(data_path))
