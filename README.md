# Overview
Use xgboost library to perform predictive analytics on bank loan approval based on information on applicant (personal information, loan information, and credit history)


## Data Source

The dataset contains information regarding 45,000 loan applicants and if their loan application was approved or not.

[Source Link](https://www.kaggle.com/datasets/udaymalviya/bank-loan-data)



## Data Exploration
Before we jump into performing analysis using the xgboost library, lets take a moment to learn about the data we are working with.

1. [Full dataset exploration](./code/data_exploration.ipynb)

    Separate out the numeric and catagorical data and perform exploration using correlation matrix and barcharts.
    
    Based on this exploration, it is noted that all loan applications that corresponds with an applicant with a previous loan default are denied of their loan.


2. [Data exploration, excluding observations without previous loan defaults](./code/data_exploration_exclude_previous_defults.ipynb)
 
    Similar analysis can be performed based on the subset data, excluding observations with previous loan defaults


## Model 

Use XGboost to predict the loan_status categorization based on the data give. For comparison we will also use the Logistic Regression model from sklearn.

1. [Use of normalize Data](./code/normalize_data.ipynb)

    During testing of the code it was identifed that the logistic regression model perfomed significantly worse when using the raw data. However, this performance issue disappread when the data was normalized (Z-score). Most likely the cuase of this is due to the scaling that is present in the dataset. For example, the income would be on the order of thousands while the person's age would be on the order of tens. 

     * Another key differnece in the data is that XGboost is able to use the categorical data as-is while the logistic regression required each categorical value to be represented as a column (Use of [pd.get_dummies](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html)). 

2. [Comparison of models](./code/analysis_of_models.ipynb)

    In this notebook, I compare the probability prediction between the 2 models. 
    In order to visualize how the model differs in their prediction, I use use the [confusion matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) from Sklearn.
    The number of applicants it approves correctly (Predicted Label: 1 & True Label: 1) are similar, while logistic regression approves more applicants that should have been denied (Predicted Label: 1 & True Label: 0).
    