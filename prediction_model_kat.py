import numpy as np
import pandas as pd
from create_submission_csv import create_submission
from data_preprocessing import load
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso

boolean_answers = pd.read_csv("datasets/trainingset_boolean_claim_amount.csv").loc[:, "ClaimAmount"]
num_expected_claim_amounts = 3335
claim_amounts_answers = pd.read_csv("datasets/trainingset_claim_amounts_only.csv").loc[:, "ClaimAmount"]
all_data = pd.read_csv("datasets/trainingset.csv")
all_data_answers = all_data.loc[:, "ClaimAmount"]


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


class TestModelRidgeRegression:

    def __init__(self, tolerance):
        self.tolerance = tolerance
        self.lambdas = [10 ** -3, 10 ** -2, 10 ** -1, 10 ** 0, 10 ** 1, 10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6,
                        10 ** 7,
                        10 ** 8, 10 ** 9, 10 ** 10]
        self.scaler = StandardScaler()

        boolean_data = load("datasets/trainingset_boolean_claim_amount.csv")
        boolean_features = boolean_data.drop("ClaimAmount", axis=1, inplace=False)
        boolean_labels = boolean_data.loc[:, "ClaimAmount"]

        scaled_boolean_data = self.scaler.fit_transform(boolean_features)
        scaled_boolean_data = pd.DataFrame(scaled_boolean_data)
        bool_best_lambda = self.find_best_lambda(scaled_boolean_data, boolean_labels)
        ridge_bool = Ridge(alpha=bool_best_lambda)

        self.boolean_claim_model = ridge_bool.fit(scaled_boolean_data, boolean_labels)

        continuous_data = load("datasets/trainingset_claim_amounts_only.csv")
        continuous_features = continuous_data.drop("ClaimAmount", axis=1, inplace=False)
        continuous_labels = continuous_data.loc[:, "ClaimAmount"]
        continuous_features.insert(continuous_features.columns.get_loc("feature13_1"), "feature13_0",
                                   [0] * continuous_features.shape[0])
        continuous_features.insert(continuous_features.columns.get_loc("feature14_1"), "feature14_0",
                                   [0] * continuous_features.shape[0])

        scaled_continuous_data = self.scaler.fit_transform(continuous_features)
        best_lambda = self.find_best_lambda(scaled_continuous_data, continuous_labels)
        ridge_continuous = Ridge(alpha=best_lambda)

        self.claim_amount_model = ridge_continuous.fit(scaled_boolean_data, boolean_labels)

        # For getting the stats for check_claim_amount_mae
        self.predict(scaled_continuous_data)
        # For getting the stats for check_claim_or_not
        data = load("datasets/trainingset.csv").drop("ClaimAmount", axis=1, inplace=False)
        scaled_data = self.scaler.fit_transform(data)
        self.predict(scaled_data)

    def find_best_lambda(self, train_data, train_data_y):
        train_error_values = []
        cv_error_values = []
        lowest_validation_error = float('inf')
        best_lambda = ''
        for i in self.lambdas:
            train_error, cv_error = CrossValidation.cv(train_data, train_data_y, i, 5, Ridge)
            train_error_values.append(train_error)
            cv_error_values.append(cv_error)
            if cv_error < lowest_validation_error:
                lowest_validation_error = cv_error
                best_lambda = i
        return best_lambda

    def predict(self, features):
        features = self.scaler.fit_transform(features)
        predictions_claim_or_not_raw = self.boolean_claim_model.predict(features)
        predictions_claim_or_not = [0] * len(features)
        for i in range(len(features)):
            if predictions_claim_or_not_raw[i] < self.tolerance:  # tune tolerance value to be closer to actual ratio
                predictions_claim_or_not[i] = 0
            else:
                predictions_claim_or_not[i] = 1
        check_claim_or_not(predictions_claim_or_not)  # only fires on trainingset

        predictions_claim_amount_raw = self.claim_amount_model.predict(features)
        predictions_claim_amount = [0] * len(features)
        for i in range(len(features)):
            if predictions_claim_or_not[i] == 0:  # tune tolerance value to be closer to actual ratio
                predictions_claim_amount[i] = 0
            else:
                predictions_claim_amount[i] = predictions_claim_amount_raw[i]
        check_claim_amount_mae(predictions_claim_amount)  # only fires on trainingset_claim_amounts_only
        check_overall_mae(predictions_claim_amount)
        return predictions_claim_amount


class TestModelLasso:

    def __init__(self, tolerance):
        self.tolerance = tolerance
        self.lambdas = [10 ** -2, 10 ** -1.75, 10 ** -1.5, 10 ** -1.25, 10 ** -1, 10 ** -.75, 10 ** -.5, 10 ** -.25,
                        10 ** 0, 10 ** .25, 10 ** .5, 10 ** .75, 10 ** 1, 10 ** 1.25, 10 ** 1.5, 10 ** 1.75, 10 ** 2]

        boolean_data = load("datasets/trainingset_boolean_claim_amount.csv")
        boolean_features = boolean_data.drop("ClaimAmount", axis=1, inplace=False)
        boolean_labels = boolean_data.loc[:, "ClaimAmount"]

        bool_best_lambda = self.find_best_lambda(boolean_features, boolean_labels)
        lasso_bool = Lasso(alpha=bool_best_lambda)

        self.boolean_claim_model = lasso_bool.fit(boolean_features, boolean_labels)

        continuous_data = load("datasets/trainingset_claim_amounts_only.csv")
        continuous_features = continuous_data.drop("ClaimAmount", axis=1, inplace=False)
        continuous_features.insert(continuous_features.columns.get_loc("feature13_1"), "feature13_0",
                                   [0] * continuous_features.shape[0])
        continuous_features.insert(continuous_features.columns.get_loc("feature14_1"), "feature14_0",
                                   [0] * continuous_features.shape[0])
        continuous_labels = continuous_data.loc[:, "ClaimAmount"]

        best_lambda = self.find_best_lambda(continuous_features, continuous_labels)
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
            train_error, cv_error = CrossValidation.cv(train_data, train_data_y, i, 5, Lasso)
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
        check_claim_or_not(predictions_claim_or_not)  # only fires on trainingset

        predictions_claim_amount_raw = self.claim_amount_model.predict(features)
        predictions_claim_amount = [0] * len(features)
        for i in range(len(features)):
            if predictions_claim_or_not[i] == 0:  # tune tolerance value to be closer to actual ratio
                predictions_claim_amount[i] = 0
            else:
                predictions_claim_amount[i] = predictions_claim_amount_raw[i]
        check_claim_amount_mae(predictions_claim_amount)  # only fires on trainingset_claim_amounts_only
        check_overall_mae(predictions_claim_amount)
        return predictions_claim_amount


def check_claim_or_not(preds):
    if len(preds) != len(boolean_answers):
        return
    false_positives = 0
    missed_positives = 0
    for i in range(len(preds)):
        if preds[i] == 0 and boolean_answers[i] == 1:
            missed_positives += 1
        if preds[i] == 1 and boolean_answers[i] == 0:
            false_positives += 1
    print(" *** ClaimAmount Identification Stats ***")
    print(f"  False positives: {false_positives}")
    print(f"  Percentage false positive: {false_positives / (len(preds) - num_expected_claim_amounts) * 100:.3f}%")
    print(f"  Missed positives: {missed_positives} out of {num_expected_claim_amounts}")
    print(
        f"  Percentage correctly identified: {(num_expected_claim_amounts - missed_positives) / num_expected_claim_amounts * 100:.3f}%")
    print(f"  Overall correct percentage: {(len(preds) - false_positives - missed_positives) / len(preds) * 100:.3f}%")
    print()


def check_claim_amount_mae(preds):
    if len(preds) != len(claim_amounts_answers):
        return
    print(" *** ClaimAmount Value Prediction Stats ***")
    mae = np.mean(abs(preds - claim_amounts_answers))
    print(f"  MAE: {mae}")
    print()


def check_overall_mae(preds):
    if len(preds) != len(all_data_answers):
        return
    print(" *** Overall Prediction Stats ***")
    mae = np.mean(abs(preds - all_data_answers))
    print(f"  Overall MAE: {mae}")
    print()


def run():
    print("***** RIDGE REGRESSION *****")
    model = TestModelRidgeRegression(0.1)
    create_submission(model, 1, 3, False)
    print("***** LASSO *****")
    model = TestModelLasso(0.1)
    create_submission(model, 1, 4, False)
