from ClassifyDatasets import classify, get_accuracy_regressors, get_accuracy_classifiers
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# param_grid for small regressor
rf_small_reg = {
    'n_estimators': [10, 50, 100],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
}
# param_grid for medium regressor
rf_medium_reg = {
    'n_estimators': [100, 300, 500],
    'max_depth': [10, 20, 50, None],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10]
}
# param_grid for large regressor
rf_large_reg = {
    'n_estimators': [500, 1000, 2000],
    'max_depth': [20, 50, 100, None],
    'min_samples_split': [2, 20, 50, 100],
    'min_samples_leaf': [1, 10, 50],
    'max_features': ['sqrt', 'log2']
}
# param_grid for small classifier
rf_small_clf = {
    'n_estimators': [10, 50, 100],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'criterion': ['gini', 'entropy']
}
# param_grid for medium classifier
rf_medium_clf = {
    'n_estimators': [100, 300, 500],
    'max_depth': [10, 20, 50, None],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10],
    'criterion': ['gini', 'entropy']
}
# param_grid for large classifier
rf_large_clf = {
    'n_estimators': [500, 1000, 2000],
    'max_depth': [20, 50, 100, None],
    'min_samples_split': [2, 20, 50, 100],
    'min_samples_leaf': [1, 10, 50],
    'max_features': ['sqrt', 'log2', None],
    'criterion': ['gini', 'entropy']
}

class OptimalRandomForest:
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
                    self.param_grid = rf_small_reg
                elif size == 1:
                    self.param_grid = rf_medium_reg
                else:
                    self.param_grid = rf_large_reg
            elif self.type_ == "classifier":
                if size == 0:
                    self.param_grid = rf_small_clf
                elif size == 1:
                    self.param_grid = rf_medium_clf
                else:
                    self.param_grid = rf_large_clf


# sets param_grid and n_iter
    def configure(self):
        size = classify(self.train_X)
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
            model = RandomizedSearchCV(estimator=RandomForestRegressor(),param_distributions=self.param_grid,n_iter=self.niter,cv=5,n_jobs=-1,scoring='neg_mean_squared_error')
        elif self.type_ == "classifier":
            model = RandomizedSearchCV(estimator=RandomForestClassifier(),param_distributions=self.param_grid,n_iter=self.niter,cv=5,n_jobs=-1,scoring='accuracy')
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