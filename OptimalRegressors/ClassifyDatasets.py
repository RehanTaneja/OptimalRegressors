import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score

def classify(train_X):
    samples = train_X.shape[0]
    cols = train_X.shape[1]
    if samples<1000 or cols <20:
        return 0
    elif samples<100000 or cols<500:
        return 1
    else:
        return 2

def get_accuracy_regressors(model,X,y):
    pred = model.predict(X)
    mse = mean_squared_error(y,pred)
    return mse

def get_accuracy_classifiers(model,X,y):
    pred = model.predict(X)
    acc = accuracy_score(y,pred)
    return acc

def compare(m1,m2,m3,m4,val_X,val_y,type_):
    model_lst = [m1,m2,m3,m4]
    if type_ == "regressor":
        acc_lst = [get_accuracy_regressors(i,val_X,val_y) for i in model_lst]
        max_acc = min(acc_lst)
        if max_acc == acc_lst[0]:
            return 0
        elif max_acc == acc_lst[1]:
            return 1
        elif max_acc == acc_lst[2]:
            return 2
        else:
            return 3
    elif type_ == "classifier":
        acc_lst = [get_accuracy_classifiers(i,val_X,val_y) for i in model_lst]
        max_acc = max(acc_lst)
        if max_acc == acc_lst[0]:
            return 0
        elif max_acc == acc_lst[1]:
            return 1
        elif max_acc == acc_lst[2]:
            return 2
        else:
            return 3