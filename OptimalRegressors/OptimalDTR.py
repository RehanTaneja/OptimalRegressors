from ClassifyDatasets import classify, get_accuracy_classifiers, get_accuracy_regressors
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

# param_grid for small regressor
dtr_small = {
    'max_depth':[3,5,10,None],
    'min_samples_split':[2,5,10],
    'min_samples_leaf':[1,2,5],
    'criterion':['squared_error','friedman_mse']
}

# param_grid for medium regressor
dtr_medium = {
    'max_depth':[5,10,20,None],
    'min_samples_split':[2,5,10,20],
    'min_samples_leaf':[1,2,5,10],
    'max_features':['sqrt','log2',None],
    'criterion':['squared_error','friedman_mse']
}
# param_grid for large regressor
dtr_large = {
    'max_depth':[10,20,50,None],
    'min_samples_split':[2,10,50,100],
    'min_samples_leaf':[1,10,50],
    'max_features':['sqrt','log2'],
    'criterion':['squared_error','friedman_mse']
}
# for small classifier
dtc_small = {
    "max_depth": [3, 5, 10], 
    "criterion": ["gini", "entropy","log_loss"], 
    "min_samples_split": [2, 5, 10]
}
# for medium classifier
dtc_medium = {
    "max_depth": [5, 10, 20,None], 
    "criterion": ["gini", "entropy","log_loss"], 
    "min_samples_split": [2, 5, 10, 20]
}
# for large classifier
dtc_large = {
    "max_depth": [10, 20, 30,50, None], 
    "criterion": ["gini", "entropy","log_loss"], 
    "min_samples_split": [2, 5, 10, 20, 50,100]
}

class OptimalDecisionTree:
    def __init__(self,train_X,train_y,val_X,val_y,type_of,param_grid=None):
        self.train_X = train_X
        self.train_y = train_y
        self.val_X = val_X
        self.val_y = val_y
        self.type_ = type_of
        self.param_grid = param_grid
        self.niter = None
        self.model = None

    def set_param_grid(self,size):
        if self.param_grid==None:
            if self.type_ == "regressor":
                if size == 0:
                    self.param_grid  = dtr_small
                elif size ==1:
                    self.param_grid = dtr_medium
                else:
                    self.param_grid = dtr_large
            elif self.type_ == "classifier":
                if size == 0:
                    self.param_grid  = dtc_small
                elif size ==1:
                    self.param_grid = dtc_medium
                else:
                    self.param_grid = dtc_large
# sets the param grid and n_iter values 
    def configure(self):
        size = classify(self.train_X)
        self.set_param_grid(size)
        if self.type_ == "regressor":
            if size == 0:
                self.niter = 10
            elif size ==1:
                self.niter = 15
            else:
                self.niter = 20
        elif self.type_ == "classifier":
            if size == 0:
                self.niter = 10
            elif size ==1:
                self.niter = 15
            else:
                self.niter = 20
# literally makes the model by returning the best_estimator_
    def make_model(self):
        if self.type_ == "regressor":
            model = RandomizedSearchCV(estimator=DecisionTreeRegressor(),param_distributions=self.param_grid,n_iter=self.niter,cv=5,n_jobs=-1,scoring='neg_mean_squared_error')
        elif self.type_ == "classifier":
            model = RandomizedSearchCV(estimator=DecisionTreeClassifier(),param_distributions=self.param_grid,n_iter=self.niter,cv=5,n_jobs=-1,scoring='accuracy')
        model.fit(self.train_X,self.train_y)
        self.model = model.best_estimator_
    # accuracy metrics
    def evaluate(self):
        if self.type_ == "regressor":
            training_accuracy = get_accuracy_regressors(self.model,self.train_X,self.train_y)
            validation_accuracy = get_accuracy_regressors(self.model,self.val_X,self.val_y)
        elif self.type_ == "classifier":
            training_accuracy = get_accuracy_classifiers(self.model,self.train_X,self.train_y)
            validation_accuracy = get_accuracy_classifiers(self.model,self.val_X,self.val_y)
        return training_accuracy,validation_accuracy
    # final function to be called to get best model
    def optimize(self):
        self.configure()
        self.make_model()
        return self.model