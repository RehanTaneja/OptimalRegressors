from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from ClassifyDatasets import classify, get_accuracy_classifiers,get_accuracy_regressors
from sklearn.model_selection import RandomizedSearchCV

knn_small = {
    'n_neighbors':[2,3,5,7,10],
    'weights':['uniform','distance'],
    'p':[1,2]
}

knn_medium = {
    'n_neighbors':[5,10,20],
    'weights':['uniform','distance'],
    'p':[1,2]
}

knn_large = {
    'n_neighbors':[10,20,50],
    'weights':['distance'],
    'p':[2]
}

knnc_small = {
    "n_neighbors": [3, 5, 7, 10], 
    "weights": ["uniform", "distance"], 
    "metric": ["euclidean", "manhattan"]
}

knnc_medium = {
    "n_neighbors": [3, 5, 10, 15, 20], 
    "weights": ["uniform", "distance"], 
    "metric": ["euclidean", "manhattan", "minkowski"]
}

knnc_large = {
    "n_neighbors": [5, 10, 20, 30, 50], 
    "weights": ["uniform", "distance"], 
    "metric": ["euclidean", "manhattan", "minkowski"]
}

class OptimalKNN:
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
                    self.param_grid = knn_small
                elif size == 1:
                    self.param_grid = knn_medium
                else:
                    self.param_grid = knn_large
            elif self.type_ == "classifier":
                if size == 0:
                    self.param_grid = knnc_small
                elif size == 1:
                    self.param_grid = knnc_medium
                else:
                    self.param_grid = knnc_large

# sets param_grid and n_iter
    def configure(self):
        size = classify(self.train_X)
        self.set_param_grid(size)
        if self.type_ == "regressor":
            if size == 0:
                self.niter = 10
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
            model = RandomizedSearchCV(estimator=KNeighborsRegressor(),param_distributions=self.param_grid,n_iter=self.niter,cv=5,n_jobs=-1,scoring='neg_mean_squared_error')
        elif self.type_ == "classifier":
            model = RandomizedSearchCV(estimator=KNeighborsClassifier(),param_distributions=self.param_grid,n_iter=self.niter,cv=5,n_jobs=-1,scoring='accuracy')
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

