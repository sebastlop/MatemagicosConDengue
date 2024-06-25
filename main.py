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
    train, test = train_test_split(raw_data, 0.15)
    # escalamos las variables
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    # crear las x y las y
    x_train, y_train = create_xy(train_scaled, lookback=lookback, horizon= horizon)
    x_test, y_test = create_xy(test_scaled, lookback=lookback, horizon= horizon)
    #no olvidar el prepare
    x_train = mlp_prepare(x_train)   
    x_test = mlp_prepare(x_test)
    return x_train, y_train, x_test, y_test, scaler

def pipe_para_lstm(raw_data, lookback=8, horizon= 1):
    #liquidamos los timestamps
    raw_data = liquidar_timestamps(raw_data)
    # separamos train de test
    train, test = train_test_split(raw_data, 0.15)
    # escalamos las variables
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    # crear las x y las y
    x_train, y_train = create_xy(train_scaled, lookback=lookback, horizon= horizon)
    x_test, y_test = create_xy(test_scaled, lookback=lookback, horizon= horizon)

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
    y = scaler.inverse_transform(np.hstack([np.empty(shape=(n_batch*n_horizon,4)), y]))
    return y[:,-1].reshape(n_batch, n_horizon)


def confidence_estimation(model, scaler, x_test, y_test):
    maes = []
    model.train()
    for i in range(1000):
        y_pred = model(torch.FloatTensor(x_test)).detach().numpy()
        y_pred_descaled = descaling(y_pred,scaler)
        y_tst_descaled = descaling(y_test,scaler)
        mae_test = mean_absolute_error(y_tst_descaled,y_pred_descaled )
        maes.append(mae_test)

    model.eval()
    y_pred = model(torch.FloatTensor(x_test)).detach().numpy()
    y_pred_descaled = descaling(y_pred,scaler)
    y_tst_descaled = descaling(y_test,scaler)
    best_pred = mean_absolute_error(y_tst_descaled,y_pred_descaled )

    return best_pred, np.array(maes) 


if __name__ == '__main__':

    #leemos y tenemos un array de numpy
    raw_data = read_table()
    #----- MLP /LSTM  ------
    model_type = 'LSTM' 

    lookback = 48
    horizon = 8
    is_train = True  # con este entrenan modelos

    # creamos un modelo
    if model_type == 'MLP':
        x_tr, y_tr, x_tst, y_tst, scaler = pipe_para_mlp(raw_data=raw_data, lookback=lookback, horizon=horizon)
        model = MLP(lookback= lookback, features= 5, horizon= horizon, n_hidden= 64)
        filename = './mlp_statedict.pth'

    elif model_type == 'LSTM':
        x_tr, y_tr, x_tst, y_tst, scaler = pipe_para_lstm(raw_data=raw_data, lookback=lookback, horizon=horizon)
        model = LSTM(features= 5, horizon= horizon, n_hidden= 64)
        filename = './lstm_statedict.pth'

    if is_train:
        # creamos un optimizador
        # elegimos un criterio de perdida

        optim = torch.optim.Adam(model.parameters(), lr = 1e-3)
        criterio = torch.nn.MSELoss()

        # definimos el numero epocas y entrenamos
        epochs = 3000
        history = []
        history_test = []
        best_test_loss = 100
        for e in range(epochs):
            model.train()
            x = torch.FloatTensor(x_tr)
            y_pred = model(x)
            loss = criterio(torch.FloatTensor(y_tr), y_pred)

            optim.zero_grad()  # pongo en 0 los gradiente
            loss.backward()     # calculo las derivadas respecto de los parametros
            optim.step()        #hago un descenso en la direccion opuesta al gradiente
            history.append(loss.item())

            with torch.no_grad():
                model.eval()
                x = torch.FloatTensor(x_tst)
                y_pred = model(x)
                tst_loss = criterio(torch.FloatTensor(y_tst), y_pred).item()
                history_test.append(tst_loss)
                if tst_loss < best_test_loss:
                    torch.save(model.state_dict(), filename)
                    best_test_loss = tst_loss
                if e%100 == 0:
                    print(f'epoch: {e} - training loss: {loss.item()} - test loss: {tst_loss}')


        plt.semilogy(history)
        plt.semilogy(history_test)
        plt.show()

    # desescalamos y predecimos
    model.load_state_dict(torch.load(filename))
    model.eval()
    y_pred = model(torch.FloatTensor(x_tst)).detach().numpy()
    y_pred_descaled = descaling(y_pred,scaler)
    y_tst_descaled = descaling(y_tst,scaler)

    for i in range(10):
        plt.plot(np.round(y_pred_descaled[-i],0), 'bo', label= 'prediccion')
        plt.plot(y_tst_descaled[-i], 'rx', label= 'casos verdaderos')
        plt.legend()
        plt.show()
    mae_test = mean_absolute_error(y_tst_descaled,y_pred_descaled )
    print(f'MAE sobre test = {mae_test}')

    # calculos de intervalos de confianza
    best_mae, results = confidence_estimation(model, scaler, x_tst, y_tst)
    print(f'MAE: media sobre 1000 evaluaciones con dropout: {results.mean()}. Desviacion estandar {results.std()}')
    print(f'MAE del mejor modelo: {best_mae}')
    plt.title('Realizacion de 1000 evaluaciones')
    plt.hist(results, 50)
    plt.show()


# Tomar un modelo y predecir una porcion del año que esté como test