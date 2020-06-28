import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch


def create_sequence(X:np.array, y:np.array, seq_len:int):
    '''
    Creates sequential data of length seq_len given input data X and labels y.

    Input:
        X: 2D array of shape (n, num_features)
        y: 1D array of shape (n,)
    Returns:
        X: past features of shape (n-seq_len, seq_len, num_features)
        y: past target values of shape (n-seq_len, seq_len)
        labels_y: target of shape (n-seq_len)

    Example:
    --------
    >>> x = np.random.randn(40,10)
    >>> y = np.random.randn(40,1)
    >>> sequence_len = 3
    >>> X, y, y_true = create_sequence(x, y, sequence_len)
    >>> print(X.shape, y.shape, y_true.shape)
    (37, 3, 10) (37, 3, 1) (37, 1)
    '''
    dt = []
    dt_y = []
    labels_y = []
    n = X.shape[0]

    assert n == y.shape[0]

    for i in range(n - seq_len):
        dt.append(X[i:i+seq_len])
        dt_y.append(y[i:i+seq_len])
        labels_y.append(y[i + seq_len])

    return np.array(dt), np.array(dt_y), np.array(labels_y)


def split_data(*inputs, train_ratio = 0.65, val_ratio = 0.145,
                test_ratio = 0.145, leave_out = 0.03):
    '''
    Splits data into training, validation and test sets. If leave_out > 0,
    a portion of the data is left out between training/validation and
    validation/test splits.

    Example:
    --------
    >>> x = np.random.randn(40,20,10)
    >>> y = np.random.randn(40,1)
    >>> out = split_data(x, y)
    >>> x_train, x_val, x_test = out[0]
    >>> y_train, y_val, y_test = out[1]
    >>> print(x_train.shape, x_val.shape, x_test.shape)
    (27, 20, 10) (6, 20, 10) (5, 20, 10)
    '''
    assert len(inputs) <= 3
    assert len(inputs[0]) == len(inputs[1])

    assert 1 - (train_ratio + val_ratio + test_ratio + 2 * leave_out) <= 0.00001

    n = len(inputs[0])
    N_train_start = 0
    N_train_end = int(train_ratio * n + 1)

    N_val_start = N_train_end + int(leave_out * n)
    N_val_end = N_val_start + int(val_ratio * n + 1)

    N_test_start = N_val_end + int(leave_out * n)

    outs = []
    for inp in inputs:
        outs.append((inp[N_train_start: N_train_end],
                    inp[N_val_start: N_val_end],
                    inp[N_test_start:]))

    return outs


def create_dataloader(*inputs: torch.tensor, batch_size=64, shuffle=False):
    '''
    Example:
    --------
    >>> x = torch.randn(40,20,10)
    >>> y = torch.randn(40,20,1)
    >>> y_true = torch.randn(40,1)
    >>> data_loader = create_dataloader(x, y, y_true, batch_size=16)
    >>> for (a,b,c) in data_loader:
    >>>     print(a.shape, b.shape, c.shape)
    torch.Size([16, 20, 10]) torch.Size([16, 20, 1]) torch.Size([16, 1])
    torch.Size([16, 20, 10]) torch.Size([16, 20, 1]) torch.Size([16, 1])
    torch.Size([8, 20, 10]) torch.Size([8, 20, 1]) torch.Size([8, 1])
    '''
    dataset = TensorDataset(*[inp.float() for inp in inputs])
    data_loader = DataLoader(dataset=dataset,batch_size=batch_size,
                            shuffle=False)
    return data_loader


# 
#
#
# X = torch.randn(40,20,10)
# y = torch.randn(40,20,1)
# yy = torch.randn(40,1)
#
# d = create_dataloader(X,y,yy, batch_size=16)
# for (q,w,e) in d:
#     print(q.shape, w.shape, e.shape)
