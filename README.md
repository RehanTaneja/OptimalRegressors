# OptimalRegressors

**OptimalRegressors** is a Python library designed to find the optimal configurations for decision tree and random forest regressors. It automates hyperparameter tuning, such as determining the optimal number of leaf nodes, using validation data to improve model performance.

## Features

- **Automatically optimizes hyperparameters for:**
  - DecisionTreeRegressor
  - RandomForestRegressor
- **Allows custom candidate values for hyperparameter tuning.**
- **Optionally fits the optimized model for direct use.**
- **Simple, lightweight, and easy to use.**

## Installation

To install **OptimalRegressors**, follow the steps below:

### Clone the Repository

```bash
git clone https://github.com/RehanTaneja/OptimalRegressors.git
cd OptimalRegressors
```

### Install Dependencies

Use pip to install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### regressors.py

#### OptimalDecisionTreeRegressor

- **Parameters:**
  - trainX,trainy Training data (features & labels)
  - valX,valy Validation  data (features & labels)
  - candidate_nodes (list, optional): A list of max_leaf_nodes values to test. Default: [5, 50, 500, 5000]
  - candidate_splits (list, optional): A list of min_sample_split values to test. Default: [2,3,4,5,6,7,8,9,10]
  - candidate_leaves (list,optional): A list of min_sample_leaf values to test. Default: [1,2,3,4,5]
  - candidate_depths (list,optional): A list of max_depth values to test. Default: [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
  - fit (bool, optional): Whether to fit the returned model on the training data. Default: False
- **Returns:**
  - model: A DecisionTreeRegressor instance with the optimal configuration

Find the best max_leaf_nodes for a decision tree:

```python
from OptimalRegressor import OptimalDecisionTreeRegressor

# Example data (replace with your dataset)
trainX, trainy = [[1], [2], [3]], [2.5, 3.5, 5.0]
valX, valy = [[1.5], [2.5]], [3.0, 4.0]

# Get the optimal DecisionTreeRegressor
model, nodes = OptimalDecisionTreeRegressor(trainX, trainy, valX, valy)

print("Optimal max_leaf_nodes:", nodes)
```

#### OptimalRandomForestRegressor

- **Parameters:**
  - trainX, trainy: Training data (features and labels)
  - vsalX, valy: Validation data (features and labels)
  - candidate_nodes (list, optional): A list of max_leaf_nodes values to test. Default: [5, 50, 500, 5000]
  - candidate_estimators (list,optional): A list of n_estimator values to test. Default: [50,100,200,300,400,500]
  - fit (bool, optional): Whether to fit the returned model on the training data. Default: False
- **Returns:**
  - model: A RandomForestRegressor instance with the optimal configuration

Find the best max_leaf_nodes value for a random forest regressor

```python
from OptimalRegressor import OptimalRandomForestRegressor

# Example data (replace with your dataset)
trainX, trainy = [[1], [2], [3]], [2.5, 3.5, 5.0]
valX, valy = [[1.5], [2.5]], [3.0, 4.0]

# Get the optimal RandomForestRegressor
model, nodes = OptimalRandomForestRegressor(trainX, trainy, valX, valy)

print("Optimal max_leaf_nodes:", nodes)
```
### OptimalDecisionTreeRegressors.py

#### OptimalMaxLeafNodes

- **Parameters:**
 - candidate_nodes: A list of max_leaf_nodes values to test
 - trainX, trainy: Training data (features and labels)
 - valX, valy: Validation data (features and labels)
 - mae: Original Mean Absolute Error
- **Returns:**
  - Optimal value for max_leaf_node with minimum mean absolute error

#### OptimalMinSampleSplit

- **Parameters:**
 - candidate_splits: A list of min_sample_split values to test
 - trainX, trainy: Training data (features and labels)
 - valX, valy: Validation data (features and labels)
 - mae: Original Mean Absolute Error
- **Returns:**
  - Optimal value for min_sample_split with minimum mean absolute error

#### OptimalMinSampleLeaf

- **Parameters:**
 - candidate_nleaves: A list of min_sample_leaf values to test
 - trainX, trainy: Training data (features and labels)
 - valX, valy: Validation data (features and labels)
 - mae: Original Mean Absolute Error
- **Returns:**
  - Optimal value for min_sample_leaf with minimum mean absolute error

#### OptimalMaxDepth

- **Parameters:**
 - candidate_nodes: A list of max_depth values to test
 - trainX, trainy: Training data (features and labels)
 - valX, valy: Validation data (features and labels)
 - mae: Original Mean Absolute Error
- **Returns:**
  - Optimal value for max_depth with minimum mean absolute error

### OptimalRandomForestRegressors.py

#### OptimalMaxLeafNodes

- **Parameters:**
 - candidate_nodes: A list of max_leaf_nodes values to test
 - trainX, trainy: Training data (features and labels)
 - valX, valy: Validation data (features and labels)
 - mae: Original Mean Absolute Error
- **Returns:**
  - Optimal value for max_leaf_node with minimum mean absolute error

#### OptimalNEstimators

- **Parameters:**
 - candidate_estimators: A list of n_estimators values to test
 - trainX, trainy: Training data (features and labels)
 - valX, valy: Validation data (features and labels)
 - mae: Original Mean Absolute Error
- **Returns:**
  - Optimal value for n_estimators with minimum mean absolute error

#### OptimalBootstrap

- **Parameters:**
 - trainX, trainy: Training data (features and labels)
 - valX, valy: Validation data (features and labels)
 - mae: Original Mean Absolute Error
- **Returns:**
  - Optimal value for bootstrap with minimum mean absolute error

## Repository Structure

```
OptimalRegressors/
│
├── OptimalDecisionTreeRegressors.py # Methods for getting optimal hyperparameters for DecisionTree
├── OptimalRandomForestRegressors.py # Methods for getting optimal hyperparameters for RandomForest
├── regressor.py      # Main library file
├── requirements.txt         # Python dependencies
├── LICENSE                  # License information
└── README.md                # Project documentation
```
