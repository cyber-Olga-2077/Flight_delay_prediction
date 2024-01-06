# All the models with cross - validation

#### 1. Linear regression model ####
# * computationally efficient
# * simple and easy to understand
# * only few assumptions to test, which are well documented in literature
# * can serve as a baseline model that will be compared to more complex models

#### 2. Lasso and ridge regressions ####
# * built on linear regression, but with regularization
# * helps with dealing with outliers and prevents overfitting to data
# * not much more complex than linear regression, but can yield better results if the data is skewed, exhibits multicolinearity or have redundant features

#### 3. Decision trees / random forest / XGBoost ####
# * able to capture complex patterns, including non-linear relationships
# * not too complicated and easy to understand and visualize
# * single decision tree is prone to overfitting, if not pruned
# * random forest is ensemble learning made of decision trees, so it also tackles the issue of overfitting
# * (preffered) XGBoost is one of the most successful models (high preditictive accuracy) in recent data science hackatons, able to handle high dimensional and large datasets. Is also robust to outliers.

#### Other models not considered ####
# * SVM (Support Vector Machines) for regression - very computationally heavy on larger datasets - timely to train. Performance and training time sensitive to hyperparameters
# * K-Nearest Neighbors (KNN) for Regression - effective assuming that the local relations are important (if plotted values would be present in some clusters), hard to assign the number of neighbours, many distance metrics available. Computationally heavy for larger datasets
# * Neural Networks for regression - computationally heavy with bigger datasets, requires a lot of data to be accurate, "black-box" (not easily interpretable). Performance and training time sensitive to hyperparameters


