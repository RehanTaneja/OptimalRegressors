from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def OptimalMaxLeafNodes(candidate_nodes,trainX,trainy,valX,valy,mae):
    best_node = candidate_nodes[0]
    for i in candidate_nodes:
        model = RandomForestRegressor(max_leaf_nodes=i,random_state=0)
        model.fit(trainX,trainy)
        preds = model.predict(valX)
        err = mean_absolute_error(preds,valy)
        if err<mae:
            mae = err
            best_node = i
    return best_node

def OptimalNEstimators(candidate_estimators,node,trainX,trainy,valX,valy,mae):
    best_estimator = candidate_estimators[0]
    for i in candidate_estimators:
        model = RandomForestRegressor(max_leaf_nodes=node,n_estimators=i,random_state=0)
        model.fit(trainX,trainy)
        preds = model.predict(valX)
        err = mean_absolute_error(preds,valy)
        if err<mae:
            mae = err
            best_estimator = i
    return best_estimator

def OptimalBootstrap(candidate_bootstrap,estimator,node,trainX,trainy,valX,valy,mae):
    best_bootstrap = candidate_bootstrap[0]
    for i in candidate_bootstrap:
        model = RandomForestRegressor(max_leaf_nodes=node,n_estimators=estimator,bootstrap=i,random_state=0)
        model.fit(trainX,trainy)
        preds = model.predict(valX)
        err = mean_absolute_error(preds,valy)
        if err<mae:
            mae = err
            best_bootstrap = i
    return best_bootstrap