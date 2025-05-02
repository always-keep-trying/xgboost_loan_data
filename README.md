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
