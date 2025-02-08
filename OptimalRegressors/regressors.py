from ClassifyDatasets import classify,get_accuracy_classifiers,get_accuracy_regressors,compare
from OptimalDTR import OptimalDecisionTree
from OptimalLinear import OptimalLinear
from OptimalRF import OptimalRandomForest
from OptimalXGB import OptimalXGBoost

class BestModel:
    def __init__(self,train_X,train_y,val_X,val_y,type_,param_grid=None):
        self.train_X = train_X
        self.train_y = train_y
        self.val_X = val_X
        self.val_y = val_y
        self.type_ = type_
        self.param_grid = param_grid
        self.model = None
        self.reg = None

    def get_model(self):
        dtr = OptimalDecisionTree(self.train_X,self.train_y,self.val_X,self.val_y,self.type_,self.param_grid)
        rf = OptimalRandomForest(self.train_X,self.train_y,self.val_X,self.val_y,self.type_,self.param_grid)
        xgb = OptimalXGBoost(self.train_X,self.train_y,self.val_X,self.val_y,self.type_,self.param_grid)
        li = OptimalLinear(self.train_X,self.train_y,self.val_X,self.val_y,self.type_,self.param_grid)

        dtr_model = dtr.optimize()
        rf_model = rf.optimize()
        xgb_model = xgb.optimize()
        li_model = li.optimize()

        comparison = compare(li_model,dtr_model,rf_model,xgb_model,self.val_X,self.val_y)

        if comparison == 0:
            self.model = li_model
            self.reg = li
        elif comparison == 1:
            self.model = dtr_model
            self.reg = dtr
        elif comparison == 2:
            self.model = rf_model
            self.reg = rf
        else:
            self.model = xgb_model
            self.reg = xgb
        
    def evaluate(self):
        return self.reg.evaluate()
    
    def optimize(self):
        self.get_model()
        return self.model

