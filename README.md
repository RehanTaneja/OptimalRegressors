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

### Optimal Decision Tree Regressor

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

### Optimal Random Forest Regressor

- **Parameters:**
  - trainX, trainy: Training data (features and labels)
  - trainX, trainy: Training data (features and labels)
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

## Repository Structure

OptimalRegressors/
│
├── OptimalRegressor.py      # Main library file
├── requirements.txt         # Python dependencies
├── LICENSE                  # License information
└── README.md                # Project documentation

