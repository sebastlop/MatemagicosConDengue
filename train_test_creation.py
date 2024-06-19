import numpy as np
from read import read_table
# El primer dataset recibe #temp, hum, precip, casos

def liquidar_timestamps(data):
    return data[:,1:]

def create_xy(data, lookback=4, horizon=1):
    x = []
    y = []
    for i in range(len(data)-lookback-horizon):
        x.append(data[i:i+lookback,:])
        y.append(data[i+lookback:i+lookback+horizon, -1])
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)

def train_test_split(data, ratio = 0.2):
    n = len(data)
    n_test = np.int32(0.2*n)
    return data[:n-n_test], data[n-n_test: n]

def mlp_prepare(x):
    '''esta funcion es para acondicionar los datos para entrenar un mlp'''
    return x.reshape(-1, x.shape[1]*x.shape[2])


if __name__ == '__main__':

    data = read_table()
    train, test = train_test_split(data)
    print(train.shape, test.shape)
    x,y  = create_xy(train, lookback=5, horizon=1)
    print(x.shape, y.shape)
