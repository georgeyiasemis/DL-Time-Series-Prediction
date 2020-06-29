import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib
import numpy as np
import datetime

def read_data_from_csv(path: str):
    """Read csv file.

    Parameters
    ----------
    path : str
        Path to csv data.

    Returns
    -------
    DataFrame

    """

    df = pd.read_csv(path)

    return df

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

def adjust_dates_to_trading_dates(df):
    """Adjusts sentiment data to which Day/Date they affect:

        Friday 19:55 - Monday 13:30 ---> Monday
        Day 00:00 - Day 19:55 ---> Day
        Day 19:55 - Day 23:59 ---> Day + 1

    Parameters
    ----------
    df : pd.Dataframe

    Returns
    -------
    DF : pd.DataFrame
        Added column with number of day in a week, and to which day the news
        correspond to.

    Example
    -------
    2000-01-03 11:21:55.000	corresponds to [Date] 2000-01-03 [Day] 1

    2000-01-03 20:40:00.000	corresponds to [Date] 2000-01-04 [Day] 2

    """

    DF = df.copy()

    open_time = datetime.time(13,30,0)
    close_time = datetime.time(19,55,0)

    days = []
    dates = []
    # Adjust Dates of weekend to Monday's news
    # Adjust afternoon's news dates to next day's date
    for dt in DF['TIMESTAMP_UTC']:

        t = np.datetime64(dt)
        t = t.astype(datetime.timedelta)
        day = t.isoweekday()

        if (day in [6, 7]) or ( day == 5 and t.time() > close_time): # if in weekend
            while day != 1: # Monday
                t += datetime.timedelta(1)
                day = t.isoweekday()

        else:
            if t.time() > close_time: # between 13:30-19:55 UTC
                day += 1
                t += datetime.timedelta(1)


        dates.append(t.date())

        days.append(day)
    DF['Day of week'] = days
    DF['Date'] = dates

    assert set(np.unique(DF['Day of week'])) == {1, 2, 3, 4, 5} # {Monday,...,Friday}
    DF = DF.set_index(pd.Index(list(range(0, len(DF)))))

    return DF

def remove_prices_no_news(prices_df, news_df):
    # Check for which days we don't have news and remove those
    # who don't.

    unique_dates = np.unique(news_df['Date'])
    unique_dates = [str(date) for date in unique_dates]

    df_prices = pd.DataFrame(columns=prices_df.columns) # Empty dataframe to append
    # only days which we have news

    n = len(prices_df)
    j = 0
    for i in range(n):
        if prices_df['Date'][i] in unique_dates:
            df_prices.loc[j] = prices_df.iloc[i]
            j += 1
    df_prices.set_index(pd.Index(list(range(0, len(df_prices)))), inplace=True)
    return df_prices


def remove_non_trading_dates(prices_df, news_df):
    # Check which days are not trading days
    trading_dates = np.unique(prices_df['Date'])
    trading_dates = [np.datetime64(date).astype(datetime.timedelta) for date in trading_dates]

    DF_news = pd.DataFrame(columns=news_df.columns) # Empty dataframe to append only days which are trading days

    for i in range(len(trading_dates)):
        date = trading_dates[i]
        DF_news = DF_news.append(news_df.loc[news_df['Date'] == date])
    DF_news.set_index(pd.Index(list(range(0, len(DF_news)))), inplace=True)
    return DF_news

def group_by_date(df):
    # Find counts of data for each date and the respective index of the 1st
    # sample with that date
    print(df.head())
    counts = []
    indices = [0]
    current_day = df['Date'][0]
    print(current_day)
    count = 1
    i = 0
    n = len(df['Date'])

    for day in df['Date'][1:]:
        i += 1
        if current_day == day:
            count += 1
            if i == n - 1:
                counts.append(count)
        else:
            counts.append(count)
            indices.append(i)
            current_day = day
            count = 1

    assert len(counts) == len(indices)

    print('Max sequence length: ', max(counts))
    print('Min sequence length: ', min(counts))
    print('Max appears at index: ', counts.index(max(counts)))
    print('Min appears at index: ', counts.index(min(counts)))

    return counts, indices

def weight_news_by_time(news, counts, indices):
    """Wheighs features by time. Newer features are assigned a higher weight.

    X = [x1,...,xT]^T with times [d1,...,dT]

    W_t = exp(-dt/K)
    W_t = W_i / sum(W)
    Z_t = W_t * X_t

    Y = mean(Z)


    Parameters
    ----------
    news : type
        Description of parameter `news`.
    counts : type
        Description of parameter `counts`.
    indices : type
        Description of parameter `indices`.

    Returns
    -------
    type
        Description of returned object.

    """

    W = []
    news['Weight'] = 0
    secs_in_day = 24 * 60 * 60
    # Average news with respect to closing time
    for j in range(len(indices)):
        i = indices[j]
        t = np.datetime64(str(news.loc[i]['Date']) + ' 19:55')
        t = t.astype(datetime.datetime)
        w = []
        for k in range(i, i+counts[j]):
            time = np.datetime64(news.iloc[k]['TIMESTAMP_UTC'])
            time = time.astype(datetime.datetime)
            time_elapsed = (t - time).seconds
            w.append(np.exp(-time_elapsed/secs_in_day))
        w = np.array(w) / sum(w)
        W += list(w)

    for j in range(len(indices)):
        assert np.abs(sum(W[indices[j]:indices[j] + counts[j]]) - 1.0) <= 0.00001

    news['Weight'] = W
    df = news[set(news.columns) - {'TIMESTAMP_UTC', 'Day of week', 'Date'}]
    df.set_index(pd.Index(list(range(0, len(df)))), inplace=True)
    df = df.apply(lambda x: x * df.Weight)
    df = df[set(df.columns) - {'Weight'}]

    columns = df.columns
    # Average
    values = [] # Data tensor X
    for i in range(len(indices)):
        index = indices[i]
        values.append(df.loc[index:index+counts[i] -1 ].mean(0).values)
        values[i] = values[i].astype(np.float32)
    X = np.array(values)

    return pd.DataFrame(X, columns = columns)

def concat_with_prices(prices, news):
    y = prices[['Open', 'High', 'Low']]
    X_data = pd.concat((news, y), axis=1)

    return X_data



if __name__ == '__main__':

    data_path = "../Data/IBM_sentiment.csv"
    prices_path = "../Data/IBM_prices.csv"
    df = read_data_from_csv(data_path)
    df = df[df['EVENT_SIMILARITY_DAYS'] > 1.0000]
    df = df[df['RELEVANCE'] > 60]
    # cols = ['TIMESTAMP_UTC', 'EVENT_SENTIMENT_SCORE', 'RELEVANCE',
    #     'EVENT_RELEVANCE', 'EVENT_SIMILARITY_KEY', 'TOPIC',
    #     'GROUP', 'TYPE', 'CSS', 'NIP']
    cols = ['TIMESTAMP_UTC', 'EVENT_SENTIMENT_SCORE',
        'EVENT_RELEVANCE', 'CSS', 'NIP']
    df = df [cols]

    prices = read_data_from_csv(prices_path)
    prices['Date'] = prices['Date'].apply(lambda x: x[:10]) # ****-**-** format

    news = adjust_dates_to_trading_dates(df)
    prices = remove_prices_no_news(prices, news)
    news = remove_non_trading_dates(prices, news)

    assert prices.shape[0] == np.unique(news['Date']).shape[0]
    assert list(np.unique(prices['Date'])) == [str(d) for d in np.unique(news['Date'])]

    counts, indices = group_by_date(news)
    assert len(counts) == prices.shape[0]

    news = weight_news_by_time(news, counts, indices)
    print(news.head())

    X = concat_with_prices(prices, news)
    print(X.head())
    pd.to_pickle(X, '../Data/IBM_X_data.pkl', protocol=4)
    y = prices['Close']
    pd.to_pickle(y, '../Data/IBM_close_data.pkl', protocol=4)
