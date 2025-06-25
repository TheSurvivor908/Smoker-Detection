# Smoker Detection
## Overview
This project aims to check is the person a smoker or non smoker with health dataset

## Dataset
- Data source : Kaggle
##  Tools & Technologies Used
- Environment: Jupyter Notebook
- Modeling Tools: XGBoost and CatBoost
## Data Preprocessing
- Fill Missing Value(If needed)
- Handle duplicate Data
- Handle Anomaly Data
- Convert Categorical Value into numerical value
- Feature engineering
- Feature Selection with correlation matrix
## Model Development
### XGBoost
- Hyperparameter Tuning: Optimal parameters (max depth, learning rate, sub samble, etc) selected based on Baynessian Opt.
### CATBoost
- Hyperparameter Tuning: Employed Bayesian optimization to fine-tune parameters such as tree depth and number of estimators.

## Model Evaluation 
- Evaluation metrics: RoC Auc
