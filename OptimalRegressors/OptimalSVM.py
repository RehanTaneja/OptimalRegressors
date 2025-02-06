from sklearn.svm import SVR,SVC
from ClassifyDatasets import classify,get_accuracy_regressors,get_accuracy_classifiers
from sklearn.model_selection import RandomizedSearchCV

svr_small = {
    'kernel':['linear','rbf','poly'],
    'C':[0.1,1,10,100],
    'epsilon':[0.01,0.1,0.5],
    'gamma':['scale','auto']
}

svr_medium = {
    'kernel':['linear','rbf'],
    'C':[1,10,100],
    'epsilon':[0.01,0.1,0.5],
    'gamma':[0.01,0.1,1]
}

svc_small = {
    "C": [0.1, 1, 10], 
    "kernel": ["linear", "rbf"], 
    "gamma": ["scale", "auto"]
}

svc_medium = {
    "C": [0.1, 1, 10, 100], 
    "kernel": ["linear", "rbf", "poly"], 
    "gamma": ["scale", "auto"]
}

svc_large = {
    "C": [0.01, 0.1, 1, 10, 100, 1000], 
    "kernel": ["linear", "rbf", "poly", "sigmoid"], 
    "gamma": ["scale", "auto"]
}

class OptimalSVM:
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
                    self.param_grid = svr_small
                elif size == 1:
                    self.param_grid = svr_medium
                else:
                    self.param_grid = svr_medium
            elif self.type_ == "classifier":
                if size == 0:
                    self.param_grid = svc_small
                elif size == 1:
                    self.param_grid = svc_medium
                else:
                    self.param_grid = svc_large

# sets param_grid and n_iter
    def configure(self):
        size = classify(self.train_X)
        self.set_param_grid(size)
        if self.type_ == "regressor":
            if size == 0:
                self.niter = 10
            elif size == 1:
                self.niter = 50
            else:
                self.niter = 100
        elif self.type_ == "classifier":
            if size == 0:
                self.niter = 20
            elif size == 1:
                self.niter = 50
            else:
                self.niter = 100
# makes the model using randomized search cv
    def make_model(self):
        if self.type_ == "regressor":
            model = RandomizedSearchCV(estimator=SVR(),param_distributions=self.param_grid,n_iter=self.niter,cv=5,n_jobs=-1,scoring='neg_mean_squared_error')
        elif self.type_ == "classifier":
            model = RandomizedSearchCV(estimator=SVC(),param_distributions=self.param_grid,n_iter=self.niter,cv=5,n_jobs=-1,scoring='accuracy')
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