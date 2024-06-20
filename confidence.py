import numpy
import torch
import matplotlib.pyplot as plt
from main import descaling
from sklearn.metrics import mean_absolute_error


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