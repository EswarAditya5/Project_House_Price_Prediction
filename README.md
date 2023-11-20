# House_Prices-Advanced_Regression_Techniques

### The project involves the following key steps:
**Data Collection: Kaggle Data link:** https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

### Competition Description:
Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.<br>

With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.<br>

### File descriptions
**train.csv** - the training set.<br>
**test.csv** - the test set.<br>
**data_description.txt** - full description of each column<br>

### Comprehensive Data Exploration with Python.
- Understand how variables are distributed and how they interact
- Apply different transformations before training machine learning models

### House Prices EDA
- Learn to use visualization techniques to study missing data and distributions
- Includes correlation heatmaps, bar plots, and pie plots to help inform appropriate inputs to a linear model

### A Study on Regression Applied to the Dataset
- Demonstrate effective tactics for feature engineering
- Explore linear regression with different regularization methods including ridge, LASSO, and ElasticNet using scikit-learn

### Regularized Linear Models
- Build a basic linear model

**Data Preprocessing:** Cleaning and preprocessing the dataset to handle missing values, outliers, and inconsistencies. This step also involves transforming categorical variables into numerical representations, normalizing numeric features, and splitting the dataset into training and testing subsets.

**Model Training:** Implementing linear regression using appropriate libraries or frameworks. The training process involves fitting the model to the training data, estimating the coefficients (slope and intercept), and optimizing the model's performance by minimizing the residual errors between the predicted and actual values.

**Model Evaluation:** Assessing the performance of the trained linear regression model using evaluation metrics such as mean squared error (MSE), root mean squared error (RMSE), and R-squared. These metrics provide insights into how well the model predicts the values and indicate its overall accuracy.

### Objective:
Predict the secondhand price of a House using 79 Independent variables describing (almost) every feature of the residential home.

### Goal
We need to predict the sales price for each house. For each Id in the test set, we need to predict the value of the SalePrice variable. 

### Learning:
Missing Value Imputation Dealing with noise, understanding model performance, and comparing them.

### License:
This project is licensed under the GPU License.

### Acknowledgments
- The dataset used in this project is sourced from: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
- The Gradient Boosting Machine algorithm is implemented using the scikit-learn library.

