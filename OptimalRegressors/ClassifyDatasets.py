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

def compare(m1,m2,val_X,val_y,type_):
    if type_ == "regressor":
        if get_accuracy_regressors(m1,val_X,val_y)>=get_accuracy_regressors(m2,val_X,val_y):
            return m1
        else:
            return m2
    elif type_ == "classifier":
        if get_accuracy_classifiers(m1,val_X,val_y)>=get_accuracy_classifiers(m2,val_X,val_y):
            return m1
        else:
            return m2