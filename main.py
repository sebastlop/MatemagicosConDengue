import numpy as np
import torch
from read import read_table
from mlp import MLP
from lstm import LSTM
from sklearn.preprocessing import MinMaxScaler
from train_test_creation import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


def pipe_para_mlp(raw_data, lookback=8, horizon= 1):
    #liquidamos los timestamps
    raw_data = liquidar_timestamps(raw_data)
    # separamos train de test
    train, test = train_test_split(raw_data, 0.3)
    # escalamos las variables
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    # crear las x y las y
    x_train, y_train = create_xy(train_scaled, lookback=lookback, horizon= horizon)
    x_test, y_test = create_xy(test_scaled, lookback=lookback, horizon= horizon)
    #no olvidar el preape
    x_train = mlp_prepare(x_train)   
    x_test = mlp_prepare(x_test)
    return x_train, y_train, x_test, y_test, scaler

def descaling(y, scaler):
    '''
    scaler interpret 4 column arrays
    si se predice un horizonte, hay que desenvolver los arrays y volverlos a la forma previa
    y debe ser un array de python
    '''
    n_batch = y.shape[0]
    n_horizon = y.shape[1]
    y = y.reshape(n_batch*n_horizon, 1)
    y = scaler.inverse_transform(np.hstack([np.empty(shape=(n_batch*n_horizon,3)), y]))
    return y[:,-1].reshape(n_batch, n_horizon)




if __name__ == '__main__':

    #leemos y tenemos un array de numpy
    raw_data = read_table()
    #----- MLP CASO ------
    lookback = 8
    horizon = 2
    x_tr, y_tr, x_tst, y_tst, scaler = pipe_para_mlp(raw_data=raw_data, lookback=lookback, horizon=horizon)
    is_train = True
    if is_train:
        # creamos un model

        mlp = MLP(lookback= lookback, features= 4, horizon= horizon, n_hidden= 64)

        # creamos un optimizador
        # elegimos un criterio de perdida

        optim = torch.optim.Adam(mlp.parameters(), lr = 1e-3)
        criterio = torch.nn.MSELoss()

        # definimos el numero epocas y entrenamos
        epochs = 1000
        history = []
        history_test = []
        best_test_loss = 100
        for e in range(epochs):
            x = torch.FloatTensor(x_tr)
            y_pred = mlp(x)
            loss = criterio(torch.FloatTensor(y_tr), y_pred)

            optim.zero_grad()  # pongo en 0 los gradiente
            loss.backward()     # calculo las derivadas respecto de los parametros
            optim.step()        #hago un descenso en la direccion opuesta al gradiente
            if e%10 == 0:
                print(loss.item())
            history.append(loss.item())

            with torch.no_grad():
                mlp.eval()
                x = torch.FloatTensor(x_tst)
                y_pred = mlp(x)
                tst_loss = criterio(torch.FloatTensor(y_tst), y_pred).item()
                history_test.append(tst_loss)
                if tst_loss < best_test_loss:
                    torch.save(mlp.state_dict(), './mlp_statedict.pth')
                    best_test_loss = tst_loss


        plt.semilogy(history)
        plt.semilogy(history_test)
        plt.show()

        torch.save(mlp.state_dict(), './mlp_statedict.pth')

    else:

        mlp = MLP(lookback=8, features=4 , horizon= 1, n_hidden=64)
        mlp.load_state_dict(torch.load('./mlp_statedict.pth'))
        mlp.train()

    # desescalamos y predecimos
    mlp.eval()
    y_pred = mlp(torch.FloatTensor(x_tst)).detach().numpy()
    y_pred_descaled = descaling(y_pred,scaler)
    y_tst_descaled = descaling(y_tst,scaler)

    plt.plot(y_pred_descaled, 'bo', label= 'prediccion')
    plt.plot(y_tst_descaled, 'rx', label= 'casos verdaderos')
    plt.legend()
    plt.show()
    mae_test = mean_absolute_error(y_tst_descaled,y_pred_descaled )
    print(mae_test)




