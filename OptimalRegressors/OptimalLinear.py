from sklearn.linear_model import Lasso,Ridge, LogisticRegression
from ClassifyDatasets import classify, get_accuracy_regressors, get_accuracy_classifiers, compare
from sklearn.model_selection import RandomizedSearchCV

rl_small = {
    'alpha':[1e-5,1e-3,0.01,0.1,1,10]
}

rl_medium = {
    'alpha':[1e-6,1e-4,1e-2,1,10,100]
}

rl_large = {
    'alpha':[1e-7,1e-5,1e-3,1,100,1000]
}

log_small = {
    "C": [0.1, 1, 10], 
    "penalty": ["l1", "l2"], 
    "solver": ["liblinear"]
}

log_medium = {
    "C": [0.1, 1, 10, 100], 
    "penalty": ["l1", "l2", "elasticnet"], 
    "solver": ["liblinear", "saga"]
}

log_large = {
    "C": [0.01, 0.1, 1, 10, 100, 1000], 
    "penalty": ["l1", "l2", "elasticnet"], 
    "solver": ["liblinear", "saga"]
}

class OptimalLinear:
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
                    self.param_grid = rl_small
                elif size == 1:
                    self.param_grid = rl_medium
                else:
                    self.param_grid = rl_large
            elif self.type_ == "classifier":
                if size == 0:
                    self.param_grid = log_small
                elif size == 1:
                    self.param_grid = log_medium
                else:
                    self.param_grid = log_large
# sets param_grid and n_iter
    def configure(self):
        size = classify(self.train_X)
        self.set_param_grid(size)
        if self.type_ == "regressor":
            if size == 0:
                self.niter = 20
            elif size == 1:
                self.niter = 20
            else:
                self.niter = 20
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
            model1 = RandomizedSearchCV(estimator=Ridge(),param_distributions=self.param_grid,n_iter=self.niter,cv=5,n_jobs=-1,scoring='neg_mean_squared_error')
            model2 = RandomizedSearchCV(estimator=Lasso(),param_distributions=self.param_grid,n_iter=self.niter,cv=5,n_jobs=-1,scoring='neg_mean_squared_error')
            model1.fit(self.train_X,self.train_y)
            model2.fit(self.train_X,self.train_y)
            model1 = model1.best_estimator_
            model2 = model2.best_estimator_
            self.model = compare(model1,model2,self.val_X,self.val_y)
        elif self.type_ == "classifier":
            model = RandomizedSearchCV(estimator=LogisticRegression(),param_distributions=self.param_grid,n_iter=self.niter,cv=5,n_jobs=-1,scoring='accuracy')
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