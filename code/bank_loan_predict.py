import os
import json

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def prob_to_boolean(probability ,threshold=0.5):
    """
    Function to convert the probability to boolean.

    :param probability: Probability ranging from 0 to 1.
    :param threshold: Threshold to convert the probability to boolean.
    :return: Boolean
    """
    return np.apply_along_axis(lambda x: x>threshold, 0, probability).tolist()


def calculate_score(func):
    """
    Function to calculate the prediciton score. To be used as a decorator.
    """
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        bool_result = prob_to_boolean(result)
        y_true = self.test_data[1]
        if 'y_true' in kwargs.keys():
            print("Custom data for prediction")
            y_true = kwargs['y_true'].squeeze().tolist()
        print(f"Prediction Accuracy: {np.round(accuracy_score(y_true=y_true ,y_pred=bool_result),3)}")
        #print(f"Coefficient of determination of prediction: {self.molde.score()}")
    return wrapper


class PredictiveModel:

    def __init__(self, project_dir):
        self.project_dir = project_dir
        self.data_path = os.path.join(self.project_dir, 'data')
        self.model_path = os.path.join(self.project_dir, 'model')
        self.loan_df = pd.read_csv(os.path.join(self.data_path, 'loan_data.csv'))
        self.x = None
        self.y = None
        self.model = None
        self.train_data = []
        self.test_data = []

    def prepare_data(self, use_categorical=True):
        """
        Prepares the data used in the model
        """
        self.x, self.y = self.loan_df.drop(labels='loan_status', axis=1), self.loan_df[['loan_status']]
        # convert columns with string data to category
        categorical_columns = self.x.select_dtypes(exclude=np.number).columns.tolist()
        if use_categorical:
            for col in categorical_columns:
                self.x[col] = self.x[col].astype('category')
        else:
            print("using pandas's get_dummies function for categorical data")
            self.x = pd.get_dummies(self.x,columns=categorical_columns)

        # test vs train data
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.25, random_state=1)
        self.train_data = [x_train, y_train]
        self.test_data = [x_test, y_test]

    def remove_variable(self, variable_name):
        """
        Remove variable(s) used in the model.

        :param variable_name: Variable(s) to remove. Can be a string or a list of strings.
        """
        if isinstance(variable_name, list):
            # remove multiple variable
            [self.remove_variable(x) for x in variable_name]
        else:
            # remove one variable
            num_vars =self.x.shape[1]
            self.x.drop(variable_name, axis=1, inplace=True)
            print(f"Variable '{variable_name}' removed! ({num_vars} columns to {self.x.shape[1]} columns)")

    def fit_model(self):
        raise NotImplementedError("Subclass will need to implement this method")

    def predict(self):
        raise NotImplementedError("Subclass will need to implement this method")


class XgboostModel(PredictiveModel):

    def __init__(self, project_dir):
        super().__init__(project_dir)
        self.model_parms = {'objective': 'binary:logistic', 'max_depth': 5, 'eta': 0.3}

    def prepare_data(self):
        super().prepare_data(use_categorical=True)

    def fit_model(self, n_boost: int=10):
        """
        Splits the data into train and test data.
        Use the test data to fit the model

        :arg n_boost: Number of boosting iterations
        """
        x_train,y_train = self.train_data

        d_train = xgb.DMatrix(data=x_train, label=y_train, enable_categorical=True)
        # train model
        final_gb = xgb.train(self.model_parms, d_train, num_boost_round=n_boost)
        self.model = final_gb

    def get_config(self):
        """
        Retrieves the model config and the model dump in JSON format.

        """
        # Output internal parameter configuration of Booster as a JSON string
        config_string = self.model.save_config()
        config_json = json.loads(config_string)
        model_dump = self.model.get_dump(dump_format='json')
        tree_model_dump = [json.loads(x) for x in model_dump] # one for each boost round
        return config_json, tree_model_dump

    @calculate_score
    def predict(self, x_data: xgb.DMatrix, y_true) -> np.array:
        """
        Prediction based on the data given.

        :param x_data: Data used to perform the prediction.
        :param y_true: True value of the preiction. Used to calculate the accuracy score

        :return: prediction
        """
        return self.model.predict(x_data, output_margin=False, pred_leaf=False)

    def predict_with_test_data(self):
        """
        Prediction based on the test data.

        :return: prediction
        """
        x_test,y_test = self.test_data
        d_test = xgb.DMatrix(data=x_test, label=y_test, enable_categorical=True)
        return self.predict(x_data=d_test, y_true=y_test)


class LogisticRegressionModel(PredictiveModel):

    def __init__(self, project_dir):
        super().__init__(project_dir)

    def prepare_data(self):
        super().prepare_data(use_categorical=False)

    def fit_model(self):
        x_train,y_train = self.train_data
        # train logistic regression model
        self.model = LogisticRegression(random_state=1,solver='saga',max_iter=5000).fit(x_train, y_train)

    @calculate_score
    def predict(self, x_data, y_true):
        prediction_out = self.model.predict_proba(x_data).tolist()
        return [x[1] for x in prediction_out]

    def predict_with_test_data(self):
        """
        Prediction based on the test data.

        :return: prediction
        """
        x_test,y_test = self.test_data
        return self.predict(x_data=x_test, y_true=y_test)


if __name__ == "__main__":
    project_dir = r"D:\Pycharm_Projects\xgboost_loan_data"
    print("Use Xgboost model")
    M1 = XgboostModel(project_dir)
    M1.prepare_data()
    #M1.remove_variable(["previous_loan_defaults_on_file","credit_score"])
    M1.fit_model()
    prediction_model1 = M1.predict_with_test_data()

    print("Use Logistic model")
    M2 = LogisticRegressionModel(project_dir)
    M2.prepare_data()
    M2.fit_model()
    prediction_model2 = M2.predict_with_test_data()

