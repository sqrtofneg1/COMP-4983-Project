import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from create_submission_csv import create_submission
from data_preprocessing import load
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
from sklearn.linear_model import Lasso
import stat_functions as stats

boolean_answers = pd.read_csv("datasets/trainingset_boolean_claim_amount.csv").loc[:, "ClaimAmount"]
num_expected_claim_amounts = 3335
claim_amounts_answers = pd.read_csv("datasets/trainingset_claim_amounts_only.csv").loc[:, "ClaimAmount"]
all_data = pd.read_csv("datasets/trainingset.csv")
all_data_answers = all_data.loc[:, "ClaimAmount"]
all_data = pd.read_csv("datasets/trainingset.csv")
boolean_data = load('datasets/trainingset_boolean_claim_amount.csv')
continuous_data = load("datasets/trainingset_claim_amounts_only.csv")
true_boolean_data = pd.read_csv('datasets/trainingset_boolean_claim_amount.csv')
true_continuous_data = pd.read_csv("datasets/trainingset_claim_amounts_only.csv")
continuous_features = continuous_data.drop("ClaimAmount", axis=1, inplace=False)
continuous_labels = continuous_data.loc[:, "ClaimAmount"]
true_continuous_features = true_continuous_data.drop("ClaimAmount", axis=1, inplace=False)
true_continuous_labels = true_continuous_data.loc[:, "ClaimAmount"]
continuous_features.insert(continuous_features.columns.get_loc("feature13_1"), "feature13_0",
                           [0] * continuous_features.shape[0])
continuous_features.insert(continuous_features.columns.get_loc("feature14_1"), "feature14_0",
                           [0] * continuous_features.shape[0])


class TestModelZeroes:

    @staticmethod
    def predict(testdata):
        return [0] * len(testdata)


class CrossValidation:

    @staticmethod
    @ignore_warnings(category=ConvergenceWarning)
    def cv(x, y, p, k, method):
        """
            k-fold cross-validation for polynomial regression
            :param x: training x: array
            :param y: training y: array
            :param p: Lambda value
            :param K: number of folds: int
            :param method: method to be used Eg ridge or lasso
            :return:
                train_error: average MAE of the training set across all K folds
                cv_error: average MAE of the validation set across all K folds
            """
        number_of_entries = len(x)
        size_of_folds = number_of_entries // k
        train_error_array = []
        cv_error_array = []
        index = 0
        x = pd.DataFrame(x)

        for i in range(0, k):
            end_index = int(index + size_of_folds)
            cv_data = x.iloc[index: end_index]
            train_data = pd.concat([x.iloc[: index], x.iloc[end_index:]])
            cv_y = y.iloc[index: end_index]
            train_y = pd.concat([y.iloc[: index], y.iloc[end_index:]])
            index += size_of_folds

            c = method(alpha=p)
            c.fit(train_data, train_y)
            predicted_train = c.predict(train_data)
            predicted_cv = c.predict(cv_data)
            train_mae = np.mean(abs(train_y - predicted_train))
            cv_mae = np.mean(abs(cv_y - predicted_cv))
            train_error_array.append(train_mae)
            cv_error_array.append(cv_mae)

        cv_error = np.average(cv_error_array)
        train_error = np.average(train_error_array)

        return train_error, cv_error


class TestModelLasso2rf:

    def __init__(self, tolerance, X, y):
        self.tolerance = tolerance
        self.lambdas = [10 ** -2, 10 ** -1.75, 10 ** -1.5, 10 ** -1.25, 10 ** -1, 10 ** -.75, 10 ** -.5, 10 ** -.25,
                        10 ** 0, 10 ** .25, 10 ** .5, 10 ** .75, 10 ** 1, 10 ** 1.25, 10 ** 1.5, 10 ** 1.75, 10 ** 2,
                        10 ** 2.25, 10 ** 2.5, 10 ** 2.75, 10 ** 3]

        forest = RandomForestClassifier(n_estimators=300, random_state=100)
        # trees = DecisionTreeClassifier()
        self.boolean_claim_model = forest.fit(X, y)
        # self.boolean_claim_model = trees.fit(X, y)

        best_lambda = self.find_best_lambda(continuous_features, continuous_labels)
        print(best_lambda)
        lasso_continuous = Lasso(alpha=best_lambda)

        self.claim_amount_model = lasso_continuous.fit(continuous_features, continuous_labels)

        # For getting the stats for check_claim_amount_mae
        self.predict(continuous_features)
        # For getting the stats for check_claim_or_not
        self.predict(load("datasets/trainingset.csv").drop("ClaimAmount", axis=1, inplace=False))

    def find_best_lambda(self, train_data, train_data_y):
        train_error_values = []
        cv_error_values = []
        lowest_validation_error = float('inf')
        best_lambda = ''
        for i in self.lambdas:
            train_error, cv_error = CrossValidation.cv(train_data, train_data_y, i, 10, Lasso)
            train_error_values.append(train_error)
            cv_error_values.append(cv_error)
            if cv_error < lowest_validation_error:
                lowest_validation_error = cv_error
                best_lambda = i
        return best_lambda

    def predict(self, features):
        predictions_claim_or_not_raw = self.boolean_claim_model.predict(features)
        predictions_claim_or_not = [0] * len(features)
        for i in range(len(features)):
            if predictions_claim_or_not_raw[i] < self.tolerance:  # tune tolerance value to be closer to actual ratio
                predictions_claim_or_not[i] = 0
            else:
                predictions_claim_or_not[i] = 1
        stats.check_claim_or_not(predictions_claim_or_not)  # only fires on trainingset

        predictions_claim_amount_raw = self.claim_amount_model.predict(features)
        predictions_claim_amount = [0] * len(features)
        for i in range(len(features)):
            if predictions_claim_or_not[i] == 0:  # tune tolerance value to be closer to actual ratio
                predictions_claim_amount[i] = 0
            else:
                predictions_claim_amount[i] = predictions_claim_amount_raw[i]
        stats.check_claim_amount_mae(predictions_claim_amount)  # only fires on trainingset_claim_amounts_only
        stats.check_overall_mae(predictions_claim_amount)
        return predictions_claim_amount


class TestModelLasso2dt:

    def __init__(self, tolerance, X, y):
        self.tolerance = tolerance
        self.lambdas = [10 ** -2, 10 ** -1.75, 10 ** -1.5, 10 ** -1.25, 10 ** -1, 10 ** -.75, 10 ** -.5, 10 ** -.25,
                        10 ** 0, 10 ** .25, 10 ** .5, 10 ** .75, 10 ** 1, 10 ** 1.25, 10 ** 1.5, 10 ** 1.75, 10 ** 2,
                        10 ** 2.25, 10 ** 2.5, 10 ** 2.75, 10 ** 3]

        # forest = RandomForestClassifier(n_estimators=100, random_state=100)
        trees = DecisionTreeClassifier()
        # self.boolean_claim_model = forest.fit(X, y)
        self.boolean_claim_model = trees.fit(X, y)

        best_lambda = self.find_best_lambda(continuous_features, continuous_labels)
        print(best_lambda)
        lasso_continuous = Lasso(alpha=best_lambda)

        self.claim_amount_model = lasso_continuous.fit(continuous_features, continuous_labels)

        # For getting the stats for check_claim_amount_mae
        self.predict(continuous_features)
        # For getting the stats for check_claim_or_not
        self.predict(load("datasets/trainingset.csv").drop("ClaimAmount", axis=1, inplace=False))

    def find_best_lambda(self, train_data, train_data_y):
        train_error_values = []
        cv_error_values = []
        lowest_validation_error = float('inf')
        best_lambda = ''
        for i in self.lambdas:
            train_error, cv_error = CrossValidation.cv(train_data, train_data_y, i, 10, Lasso)
            train_error_values.append(train_error)
            cv_error_values.append(cv_error)
            if cv_error < lowest_validation_error:
                lowest_validation_error = cv_error
                best_lambda = i
        return best_lambda

    def predict(self, features):
        predictions_claim_or_not_raw = self.boolean_claim_model.predict(features)
        predictions_claim_or_not = [0] * len(features)
        for i in range(len(features)):
            if predictions_claim_or_not_raw[i] < self.tolerance:  # tune tolerance value to be closer to actual ratio
                predictions_claim_or_not[i] = 0
            else:
                predictions_claim_or_not[i] = 1
        stats.check_claim_or_not(predictions_claim_or_not)  # only fires on trainingset

        predictions_claim_amount_raw = self.claim_amount_model.predict(features)
        predictions_claim_amount = [0] * len(features)
        for i in range(len(features)):
            if predictions_claim_or_not[i] == 0:  # tune tolerance value to be closer to actual ratio
                predictions_claim_amount[i] = 0
            else:
                predictions_claim_amount[i] = predictions_claim_amount_raw[i]
        stats.check_claim_amount_mae(predictions_claim_amount)  # only fires on trainingset_claim_amounts_only
        stats.check_overall_mae(predictions_claim_amount)
        return predictions_claim_amount


class DecisionTree:
    def __init__(self, tolerance, X, y):
        self.tolerance = tolerance
        forest = RandomForestClassifier(n_estimators=300, random_state=100)
        # #trees = DecisionTreeClassifier()
        self.boolean_claim_model = forest.fit(X, y)
        # self.boolean_claim_model = trees.fit(X, y)
        treereg = DecisionTreeRegressor(max_features=14)
        self.claim_amount_model = treereg.fit(true_continuous_features, true_continuous_labels)

        # For getting the stats for check_claim_amount_mae
        self.predict(true_continuous_features)
        # For getting the stats for check_claim_or_not
        data = pd.read_csv("datasets/trainingset.csv").drop("ClaimAmount", axis=1, inplace=False)
        self.predict(data)

    def predict(self, features):
        predictions_claim_or_not_raw = self.boolean_claim_model.predict(features)
        predictions_claim_or_not = [0] * len(features)
        for i in range(len(features)):
            if predictions_claim_or_not_raw[i] < self.tolerance:  # tune tolerance value to be closer to actual ratio
                predictions_claim_or_not[i] = 0
            else:
                predictions_claim_or_not[i] = 1
        stats.check_claim_or_not(predictions_claim_or_not)  # only fires on trainingset

        predictions_claim_amount_raw = self.claim_amount_model.predict(features)
        predictions_claim_amount = [0] * len(features)
        for i in range(len(features)):
            if predictions_claim_or_not[i] == 0:  # tune tolerance value to be closer to actual ratio
                predictions_claim_amount[i] = 0
            else:
                predictions_claim_amount[i] = predictions_claim_amount_raw[i]
        stats.check_claim_amount_mae(predictions_claim_amount)  # only fires on trainingset_claim_amounts_only
        stats.check_overall_mae(predictions_claim_amount)
        return predictions_claim_amount


class UnderOverSamplerData:

    def __init__(self, strat1, strat2, bool):
        if bool:
            X = true_boolean_data.drop("ClaimAmount", axis=1, inplace=False)
            y = true_boolean_data.loc[:, "ClaimAmount"]
        else:
            X = boolean_data.drop("ClaimAmount", axis=1, inplace=False)
            y = boolean_data.loc[:, "ClaimAmount"]
        over = RandomOverSampler(sampling_strategy=strat1)
        under = RandomUnderSampler(sampling_strategy=strat2)
        X, y = over.fit_resample(X, y)
        self.X, self.y = under.fit_resample(X, y)

    def values(self):
        print(self.X.shape)
        return self.X, self.y


class SMOTESamplerData:

    def __init__(self, bool):
        if bool:
            X = true_boolean_data.drop("ClaimAmount", axis=1, inplace=False)
            y = true_boolean_data.loc[:, "ClaimAmount"]
        else:
            X = boolean_data.drop("ClaimAmount", axis=1, inplace=False)
            y = boolean_data.loc[:, "ClaimAmount"]
        smote = SMOTE()
        self.X, self.y = smote.fit_resample(X, y)

    def values(self):
        return self.X, self.y


def run():

    print("***** LASSO UNDER OVER *****")
    print("Tolerance 0.05")
    under_over = UnderOverSamplerData(0.1, 0.5, False)
    X_under_over, y_under_over = under_over.values()
    model = TestModelLasso2rf(0.05, X_under_over, y_under_over)
    create_submission(model, 3, 8, False)

    print("***** LASSO SMOTE *****")
    print("Tolerance 0.05")
    smote = SMOTESamplerData(False)
    X_smote, y_smote = smote.values()
    model = TestModelLasso2rf(0.05, X_smote, y_smote)
    create_submission(model, 3, 8, False)

    print("***** LASSO UNDER OVER *****")
    print("Tolerance 0.15")
    under_over = UnderOverSamplerData(0.1, 0.5, False)
    X_under_over, y_under_over = under_over.values()
    model = TestModelLasso2rf(0.15, X_under_over, y_under_over)
    create_submission(model, 3, 10, False)

    print("***** LASSO SMOTE *****")
    print("Tolerance 0.15")
    smote = SMOTESamplerData(False)
    X_smote, y_smote = smote.values()
    model = TestModelLasso2rf(0.15, X_smote, y_smote)
    create_submission(model, 3, 10, False)

    print("***** UNDER-OVERSAMPLE WITH DECISION TREE *****")
    under_over = UnderOverSamplerData(0.1, 0.5, True)
    X_under_over, y_under_over = under_over.values()
    model = DecisionTree(0.1, X_under_over, y_under_over)
    create_submission(model, 3, 4, True)

    print("***** SMOTE WITH DECISION TREE *****")
    smote = SMOTESamplerData(True)
    X_smote, y_smote = smote.values()
    model = DecisionTree(0.1, X_smote, y_smote)
    create_submission(model, 3, 4, True)
