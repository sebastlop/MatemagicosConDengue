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



if __name__ == '__main__':

    is_train = False
    if is_train:
        #leemos y tenemos un array de numpy
        raw_data = read_table()
        #----- MLP CASO ------
        x_tr, y_tr, x_tst, y_tst, scaler = pipe_para_mlp(raw_data=raw_data, lookback=8, horizon=1)

        # creamos un model

        mlp = MLP(lookback= 8, features= 4, horizon= 1, n_hidden= 64)

        # creamos un optimizador
        # elegimos un criterio de perdida

        optim = torch.optim.Adam(mlp.parameters(), lr = 1e-3)
        criterio = torch.nn.MSELoss()

        # definimos el numero epocas y entrenamos
        epochs = 2000
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
        #leemos y tenemos un array de numpy
        raw_data = read_table()
        #----- MLP CASO ------
        x_tr, y_tr, x_tst, y_tst, scaler = pipe_para_mlp(raw_data=raw_data, lookback=8, horizon=1)
        mlp = MLP(lookback=8, features=4 , horizon= 1, n_hidden=64)
        mlp.load_state_dict(torch.load('./mlp_statedict.pth'))
        mlp.train()

    # desescalamos y predecimos
    y_pred = mlp(torch.FloatTensor(x_tst)).detach().numpy()
    
    y_pred_descaled = scaler.inverse_transform(np.hstack([np.empty(shape=(y_pred.shape[0],3)), y_pred]))[:,-1]
    y_tst_descaled = scaler.inverse_transform(np.hstack([np.empty(shape=(y_tst.shape[0],3)), y_tst]))[:,-1]

    plt.plot(y_pred_descaled, 'bo', label= 'prediccion')
    plt.plot(y_tst_descaled, 'rx', label= 'casos verdaderos')
    plt.legend()
    plt.show()
    mae_test = mean_absolute_error(y_tst_descaled,y_pred_descaled )
    print(mae_test)




