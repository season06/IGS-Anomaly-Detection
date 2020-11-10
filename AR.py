import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tqdm import tqdm
import statsmodels.api as sm
import itertools
import threading
import time

def read_file():
    train_set = pd.read_csv('dataset/TrainingSet.csv')
    test_set = pd.read_csv('dataset/TestingSet.csv')

    print(train_set.shape)
    print(test_set.shape)

    train = list(train_set.ErrorCount)
    test = list(test_set.ErrorCount)

    train_error_date = list(train_set.DateTime)
    test_error_date = list(test_set.DateTime)

    return train, test, train_error_date, test_error_date

def AR_model(train, test, train_error_date, test_error_date, pdq):
    # for p in p_list:
    p, d, q = pdq[0], pdq[1], pdq[2]
    predict = []
    test_mse = []

    print(f"------------p = {p} | d = {d} | q = {q}-----------")
    np.random.seed(0)

    model = SARIMAX(train, trend='c', order=pdq, enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit()

    start = 168 # actual is p
    model_fit = model_fit.append(test[0:start])

    for t in range(start, len(test)):
        output = model_fit.forecast()

        pred = output[0]
        truth = test[t]
        
        loss = abs(pred - truth)
        predict.append(pred)
        test_mse.append(loss)

        if (loss < 5000):
            model_fit = model_fit.append([truth])
        else:
            model_fit = model_fit.append([pred])

    plt.plot(test_error_date[start:], test[start:], label='Ground truth')
    plt.plot(test_error_date[start:], predict, color='red', label='Prediction')
    plt.plot(test_error_date[start:], test_mse, color='black', label='Error')
    plt.xticks(test_error_date[start:][::24], fontsize=12, rotation = 60)
    plt.legend(loc='best')
    plt.title(f'AR p={p}, d={d}, q={q} | Error={np.asarray(test_mse).mean():.02f}',fontsize=18)
    # plt.show()

    plt.savefig(f"./fig/{p}-{d}-{q}.png")
    print('save')
    plt.close()

if __name__ == '__main__':
    train, test, train_error_date, test_error_date = read_file()

    # p_list = [24, 48, 72, 96, 120, 144, 168]
    # d_list = [0]
    # q_list = [0, 1, 2, 3, 4, 5, 6]
    p_list = [24]
    d_list = [0]
    q_list = range(1,24)
    pdq_list = list(itertools.product(p_list, d_list, q_list)) 

    dd = datetime.now()

    for pdq in pdq_list:
        AR_model(train, test, train_error_date, test_error_date, pdq)

    print(datetime.now())
    print(datetime.now() - dd)

    print('done')