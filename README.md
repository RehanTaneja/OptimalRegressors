# Machine Learning Model Optimization
This repository contains a set of utilities designed to automate the process of optimizing machine learning models using various regression and classification algorithms. The code implements hyperparameter tuning for multiple models using RandomizedSearchCV and provides a framework to compare the performance of these models to find the best one for a given dataset.

## Features
 - Hyperparameter Optimization: Automatically tunes the hyperparameters of several models (e.g., XGBoost, Decision Trees, Random Forest, KNN, SVM, and more).
 - Model Selection: Compares the performance of different models and returns the best performing regressor or classifier based on validation data.
 - Dataset Classification: Automatically classifies datasets to determine whether they are suited for regression or classification tasks, adjusting the optimization approach accordingly.
 - Cross-validation: Uses cross-validation to ensure robust evaluation during hyperparameter search.
Supported Models
 - XGBoost (Regressor and Classifier)
 - Decision Tree Regressor (DTR)
 - Gradient Boosting Regressor (GBR)
 - K-Nearest Neighbors (KNN)
 - Random Forest (RF)
 - Support Vector Machine (SVM)
 - Linear Regression (Ridge,Lasso,Logistic)
## Installation
To use the model optimization tools, clone this repository and install the necessary dependencies:
```bash
git clone https://github.com/yourusername/ml-model-optimization.git
cd ml-model-optimization
pip install -r requirements.txt
```
## Usage
### Importing the Modules
In your project, import the necessary modules:
```Python
from OptimalXGB import OptimalXGBoost
from OptimalDTR import OptimalDTR
from OptimalGBR import OptimalGBR
from OptimalKNN import OptimalKNN
from OptimalLinear import OptimalLinear
from OptimalRF import OptimalRF
from OptimalSVM import OptimalSVM
from ClassifyDatasets import classify, get_accuracy_classifiers, get_accuracy_regressors
from regressors import BestModel
```
### Optimizing a Model
To optimize a specific model (e.g., XGBoost Regressor), follow these steps:

 - Prepare your data: Ensure you have the training and validation datasets ready (train_X, train_y, val_X, val_y).
 - Create the optimal model object:
```Python
# For regression (XGBoost Regressor example)
optimizer = OptimalXGBoost(train_X, train_y, val_X, val_y, type_='regressor')
```
 - Run the optimization:
```Python
# Perform the optimization
best_model = optimizer.optimize()

# Evaluate the model
train_accuracy, val_accuracy = optimizer.evaluate()

print("Training Accuracy: ", train_accuracy)
print("Validation Accuracy: ", val_accuracy)
```
This will automatically perform the necessary hyperparameter tuning and return the best regressor or classifier.

### Getting Best Model
To get the overall best model by comparing multiple models. follow these steps:

 - Prepare your data: Ensure you have the training and validation datasets ready (train_X, train_y, val_X, val_y).
 - Create the best model object:
```Python
# For regression
best_model = BestModel(train_X,train_y,val_X,val_y,type_='regressor')

# Get the best model
best_overall_model = best_model.optimize() # This will directly store the model (like DecisionTreeRegressor(),Ridge()...etc)

#Evaluate Model
training_accuracy,validation_accuracy = best_model.evaluate()
```
## File Structure
```bash
ml-model-optimization/
├── __init__.py            # Package initialization
├── ClassifyDatasets.py    # Dataset classification and accuracy calculation
├── OptimalDTR.py          # Decision Tree Regressor optimization
├── OptimalGBR.py          # Gradient Boosting Regressor optimization
├── OptimalKNN.py          # K-Nearest Neighbors optimization
├── OptimalLinear.py       # Linear Regression optimization
├── OptimalRF.py           # Random Forest optimization
├── OptimalSVM.py          # Support Vector Machine optimization
├── OptimalXGB.py          # XGBoost optimization
├── regressors.py          # comparing models and returning best overall model
├── requirements.txt       # Dependencies
├── README.md              # Documentation
```
## Requirements
This project requires the following Python libraries:

 - scikit-learn: For machine learning algorithms and cross-validation.
 - xgboost: For XGBoost regressor and classifier.
 - numpy: For numerical operations.
 - pandas: For data manipulation.
You can install the dependencies using pip:
```bash
pip install -r requirements.txt
```
## License
This project is licensed under the MIT License - see the LICENSE file for details.








