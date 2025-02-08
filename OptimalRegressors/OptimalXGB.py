from ClassifyDatasets import classify, get_accuracy_regressors, get_accuracy_classifiers
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

xgb_small = {
    'n_estimators':[50,100,200],
    'learning_rate':[0.1,0.2,0.3],
    'max_depth':[3,5,7],
    'subsample':[0.8,1.0],
    'colsample_bytree':[0.8,1.0]
}

xgb_medium = {
    'n_estimators':[100,300,500],
    'learning_rate':[0.01,0.05,0.1],
    'max_depth':[3,5,10],
    'subsample':[0.6,0.8,1.0],
    'colsample_bytree':[0.6,0.8,1.0]
}

xgb_large = {
    'n_estimators':[500,1000,2000],
    'learning_rate':[0.001,0.01,0.05],
    'max_depth':[5,10,20],
    'subsample':[0.5,0.7,1.0],
    'colsample_bytree':[0.5,0.7,1.0]
}

xgbc_small = {
    "n_estimators": [50, 100, 200], 
    "learning_rate": [0.1, 0.05, 0.01], 
    "max_depth": [3, 5, 7]
}

xgbc_medium = {
    "n_estimators": [100, 300, 500], 
    "learning_rate": [0.1, 0.05, 0.01], 
    "max_depth": [3, 5, 10]
}

xgbc_large = {
    "n_estimators": [500, 1000, 1500], 
    "learning_rate": [0.05, 0.01, 0.005], 
    "max_depth": [5, 10, 15]
}

class OptimalXGBoost:
    def __init__(self,train_X,train_y,val_X,val_y,type_,param_grid=None):
        self.train_X = train_X
        self.train_y = train_y
        self.val_X = val_X
        self.val_y = val_y
        self.type_ = type_
        self.param_grid = param_grid
        self.niter = None
        self.model = None

    def set_param_grid(self,size):
        if self.param_grid == None:
            if self.type_ == "regressor":
                if size == 0:
                    self.param_grid = xgb_small
                elif size == 1:
                    self.param_grid = xgb_medium
                else:
                    self.param_grid = xgb_large
            elif self.type_ == "classifier":
                if size == 0:
                    self.param_grid = xgbc_small
                elif size == 1:
                    self.param_grid = xgbc_medium
                else:
                    self.param_grid = xgbc_large

# sets param_grid and n_iter
    def configure(self):
        size = classify(self.train_X)
        if self.type_ == "regressor":
            if size == 0:
                self.niter = 20
            elif size == 1:
                self.niter = 20
            else:
                self.niter = 30
        elif self.type_ == "classifier":
            if size == 0:
                self.niter = 20
            elif size == 1:
                self.niter = 20
            else:
                self.niter = 30
# makes the model using randomized search cv
    def make_model(self):
        if self.type_ == "regressor":
            model = RandomizedSearchCV(estimator=XGBRegressor(),param_distributions=self.param_grid,n_iter=self.niter,cv=5,n_jobs=-1,scoring='neg_mean_squared_error')
        elif self.type_ == "classifier":
            model = RandomizedSearchCV(estimator=XGBClassifier(),param_distributions=self.param_grid,n_iter=self.niter,cv=5,n_jobs=-1,scoring='accuracy')
        model.fit(self.train_X,self.train_y)
        self.model = model.best_estimator_

    def evaluate(self):
        if self.type_ == "regressor":
            training_accuracy = get_accuracy_regressors(self.model,self.train_X,self.train_y)
            validation_accuracy = get_accuracy_regressors(self.model,self.val_X,self.val_y)
        elif self.type_ == "classifier":
            training_accuracy = get_accuracy_classifiers(self.model,self.train_X,self.train_y)
            validation_accuracy = get_accuracy_classifiers(self.model,self.val_X,self.val_y)
        return training_accuracy,validation_accuracy
    
# final function to be called
    def optimize(self):
        self.configure()
        self.make_model()
        return self.model