# Smoker Detection
## Overview
-"This project is designed to accurately determine whether individuals are smokers or non-smokers, utilizing a sophisticated data analysis approach. Additionally, we've developed an interactive website that runs locally with react."
## Dataset
- Data source : Kaggle
##  Tools & Technologies Used
- Environment: Jupyter Notebook, React
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
- Hyperparameter Tuning: Employed Bayesian optimization to fine-tune parameters.

## Model Evaluation 
- Evaluation metrics: RoC Auc
