import numpy as np
import pandas as pd
from create_submission_csv import create_submission
from data_preprocessing import load
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import sklearn.linear_model as lin_mod
import prediction_model_sunmin_tensorflow as tensorflow_model
import stat_functions as stats


# TODO: Implement ridge and lasso


categorical = ["feature3", "feature4", "feature5", "feature7", "feature9", "feature11",
               "feature13", "feature14", "feature15", "feature16", "feature18"]
continuous = ["feature1", "feature2", "feature6", "feature8", "feature10", "feature12", "feature17"]


class TestModelZeroes:

    @staticmethod
    def predict(testdata):
        return [0] * len(testdata)


class TestModelLinearRegression:  # data normalization only on continuous data

    def __init__(self, tolerance=None):
        self.tolerance = tolerance
        lin_reg = LinearRegression()
        boolean_data = pd.read_csv("datasets/trainingset_boolean_claim_amount.csv")
        boolean_features = boolean_data.drop("ClaimAmount", axis=1, inplace=False)
        boolean_features.loc[:, continuous] = StandardScaler().fit_transform(boolean_features.loc[:, continuous])
        boolean_labels = boolean_data.loc[:, "ClaimAmount"]
        self.claim_or_not_model = lin_reg.fit(boolean_features, boolean_labels)

        lin_reg_2 = LinearRegression()
        claim_amount_data = pd.read_csv("datasets/trainingset_claim_amounts_only.csv")
        claim_amount_features = claim_amount_data.drop("ClaimAmount", axis=1, inplace=False)
        claim_amount_features.loc[:, continuous] = StandardScaler().fit_transform(
            claim_amount_features.loc[:, continuous])
        claim_amount_labels = claim_amount_data.loc[:, "ClaimAmount"]
        self.claim_amount_model = lin_reg_2.fit(claim_amount_features, claim_amount_labels)

        self.calculate_tolerance()

        # For getting the stats for check_claim_amount_mae
        self.predict(
            pd.read_csv("datasets/trainingset_claim_amounts_only.csv").drop("ClaimAmount", axis=1, inplace=False))
        # For getting the stats for check_claim_or_not
        self.predict(pd.read_csv("datasets/trainingset.csv").drop("ClaimAmount", axis=1, inplace=False))

    def calculate_tolerance(self):
        features = pd.read_csv("datasets/trainingset.csv").drop("ClaimAmount", axis=1, inplace=False)
        features.loc[:, continuous] = StandardScaler().fit_transform(features.loc[:, continuous])
        predictions_for_tolerance = self.claim_or_not_model.predict(features)
        if self.tolerance is None:
            sorted_arr = sorted(predictions_for_tolerance)
            print(np.percentile(sorted_arr, 95))
            self.tolerance = sorted_arr[-stats.num_expected_claim_amounts]

    def predict(self, features):
        features.loc[:, continuous] = StandardScaler().fit_transform(features.loc[:, continuous])
        predictions_claim_or_not_raw = self.claim_or_not_model.predict(features)
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


class TestModelTFLinearRegression:  # data normalization on all data

    def __init__(self, tolerance=None):
        self.tolerance = tolerance
        self.claim_or_not_model = tensorflow_model.get_tensorflow_model_boolean()

        lin_reg_2 = LinearRegression()
        claim_amount_data = pd.read_csv("datasets/trainingset_claim_amounts_only.csv")
        claim_amount_features = claim_amount_data.drop("ClaimAmount", axis=1, inplace=False)
        claim_amount_features = StandardScaler().fit_transform(claim_amount_features)
        claim_amount_labels = claim_amount_data.loc[:, "ClaimAmount"]
        self.claim_amount_model = lin_reg_2.fit(claim_amount_features, claim_amount_labels)

        self.calculate_tolerance()

        # For getting the stats for check_claim_amount_mae
        self.predict(
            pd.read_csv("datasets/trainingset_claim_amounts_only.csv").drop("ClaimAmount", axis=1, inplace=False))
        # For getting the stats for check_claim_or_not
        self.predict(pd.read_csv("datasets/trainingset.csv").drop("ClaimAmount", axis=1, inplace=False))

    def calculate_tolerance(self):
        features = pd.read_csv("datasets/trainingset.csv").drop("ClaimAmount", axis=1, inplace=False)
        features = StandardScaler().fit_transform(features)
        predictions_for_tolerance = self.claim_or_not_model.predict(features)[:, 1]
        if self.tolerance is None:
            sorted_arr = sorted(predictions_for_tolerance)
            self.tolerance = np.percentile(sorted_arr, 93)
        print(f"TOLERANCE: {self.tolerance}")

    def predict(self, features):
        features = StandardScaler().fit_transform(features)
        predictions_claim_or_not_raw = self.claim_or_not_model.predict(features)[:, 1]
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


class TestModelRandomForestLasso:

    def __init__(self, tolerance=None):
        self.tolerance = tolerance
        boolean_data = pd.read_csv("datasets/trainingset_boolean_claim_amount.csv")
        boolean_features = boolean_data.drop("ClaimAmount", axis=1, inplace=False)
        boolean_features.loc[:, continuous] = StandardScaler().fit_transform(boolean_features.loc[:, continuous])
        boolean_labels = boolean_data.loc[:, "ClaimAmount"]
        self.claim_or_not_model = RandomForestClassifier(class_weight="balanced").fit(boolean_features, boolean_labels)

        claim_amount_data = pd.read_csv("datasets/trainingset_claim_amounts_only.csv")
        claim_amount_features = claim_amount_data.drop("ClaimAmount", axis=1, inplace=False)
        claim_amount_features = StandardScaler().fit_transform(claim_amount_features)
        claim_amount_labels = claim_amount_data.loc[:, "ClaimAmount"]
        lambdas_lasso = [10 ** (x * .25) for x in (range(-8, 14))]
        lasso_results = [kfoldCV_lasso(claim_amount_features, claim_amount_labels, 5, lambda_value) for lambda_value in lambdas_lasso]
        lasso_cv_errors = [(lasso_results[i][1] + lasso_results[i][0]).tolist() for i in range(len(lasso_results))]
        lowest_MAE_lasso = min(lasso_cv_errors)
        selected_lambda = lambdas_lasso[lasso_cv_errors.index(lowest_MAE_lasso)]
        print(f"Lasso Selected Lambda: {selected_lambda}")
        self.claim_amount_model = lin_mod.Lasso(selected_lambda).fit(claim_amount_features, claim_amount_labels)

        self.calculate_tolerance()

        # For getting the stats for check_claim_amount_mae
        self.predict(
            pd.read_csv("datasets/trainingset_claim_amounts_only.csv").drop("ClaimAmount", axis=1, inplace=False))
        # For getting the stats for check_claim_or_not
        self.predict(pd.read_csv("datasets/trainingset.csv").drop("ClaimAmount", axis=1, inplace=False))

    def calculate_tolerance(self):
        features = pd.read_csv("datasets/trainingset.csv").drop("ClaimAmount", axis=1, inplace=False)
        features = StandardScaler().fit_transform(features)
        predictions_for_tolerance = self.claim_or_not_model.predict_proba(features)[:, 1]
        if self.tolerance is None:
            sorted_arr = sorted(predictions_for_tolerance)
            self.tolerance = np.percentile(sorted_arr, 95)

    def predict(self, features):
        features = StandardScaler().fit_transform(features)
        predictions_claim_or_not_raw = self.claim_or_not_model.predict_proba(features)[:, 1]
        predictions_claim_or_not = [0] * len(features)
        for i in range(len(features)):
            if predictions_claim_or_not_raw[i] < self.tolerance:
                predictions_claim_or_not[i] = 0
            else:
                predictions_claim_or_not[i] = 1
        stats.check_claim_or_not(predictions_claim_or_not)  # only fires on trainingset

        predictions_claim_amount_raw = self.claim_amount_model.predict(features)
        predictions_claim_amount = [0] * len(features)
        for i in range(len(features)):
            if predictions_claim_or_not[i] == 0:
                predictions_claim_amount[i] = 0
            else:
                predictions_claim_amount[i] = predictions_claim_amount_raw[i]
        stats.check_claim_amount_mae(predictions_claim_amount)  # only fires on trainingset_claim_amounts_only
        stats.check_overall_mae(predictions_claim_amount)
        return predictions_claim_amount


class TestModelTFRidge:  # data normalization on all data

    def __init__(self, tolerance=None):
        self.tolerance = tolerance
        self.claim_or_not_model = tensorflow_model.get_tensorflow_model_boolean()

        claim_amount_data = pd.read_csv("datasets/trainingset_claim_amounts_only.csv")
        claim_amount_features = claim_amount_data.drop("ClaimAmount", axis=1, inplace=False)
        claim_amount_features = StandardScaler().fit_transform(claim_amount_features)
        claim_amount_labels = claim_amount_data.loc[:, "ClaimAmount"]
        lambdas_ridge = [10 ** x for x in (range(-3, 11))]
        ridge_results = [kfoldCV_ridge(claim_amount_features, claim_amount_labels, 5, lambda_value) for lambda_value in
                         lambdas_ridge]
        ridge_cv_errors = [ridge_results[i][1].tolist() for i in range(len(ridge_results))]
        lowest_MAE_ridge = min(ridge_cv_errors)
        selected_lambda = lambdas_ridge[ridge_cv_errors.index(lowest_MAE_ridge)]
        print(f"Ridge Selected Lambda: {selected_lambda}")
        self.claim_amount_model = lin_mod.Ridge(selected_lambda).fit(claim_amount_features, claim_amount_labels)

        self.calculate_tolerance()

        # For getting the stats for check_claim_amount_mae
        self.predict(
            pd.read_csv("datasets/trainingset_claim_amounts_only.csv").drop("ClaimAmount", axis=1, inplace=False))
        # For getting the stats for check_claim_or_not
        self.predict(pd.read_csv("datasets/trainingset.csv").drop("ClaimAmount", axis=1, inplace=False))

    def calculate_tolerance(self):
        features = pd.read_csv("datasets/trainingset.csv").drop("ClaimAmount", axis=1, inplace=False)
        features = StandardScaler().fit_transform(features)
        predictions_for_tolerance = self.claim_or_not_model.predict(features)[:, 1]
        if self.tolerance is None:
            sorted_arr = sorted(predictions_for_tolerance)
            self.tolerance = np.percentile(sorted_arr, 95)
        print(f"TOLERANCE: {self.tolerance}")

    def predict(self, features):
        features = StandardScaler().fit_transform(features)
        predictions_claim_or_not_raw = self.claim_or_not_model.predict(features)[:, 1]
        predictions_claim_or_not = [0] * len(features)
        for i in range(len(features)):
            if predictions_claim_or_not_raw[i] < self.tolerance:
                predictions_claim_or_not[i] = 0
            else:
                predictions_claim_or_not[i] = 1
        stats.check_claim_or_not(predictions_claim_or_not)  # only fires on trainingset

        predictions_claim_amount_raw = self.claim_amount_model.predict(features)
        predictions_claim_amount = [0] * len(features)
        for i in range(len(features)):
            if predictions_claim_or_not[i] == 0:
                predictions_claim_amount[i] = 0
            else:
                predictions_claim_amount[i] = predictions_claim_amount_raw[i]
        stats.check_claim_amount_mae(predictions_claim_amount)  # only fires on trainingset_claim_amounts_only
        stats.check_overall_mae(predictions_claim_amount)
        return predictions_claim_amount


# ADAPTED FROM LAB 5
# k-fold cross-validation
# Inputs:
#  x: training input
#  y: training output
#  K: number of folds
#  lambda_value: constant that controls strength of regularization
# Outputs:
#  train_error: average MAE of training set across all K folds
#  cv_error: average MAE of validation set across all K folds
def kfoldCV_ridge(x, y, K, lambda_value):
    x = pd.DataFrame(x)
    y = pd.DataFrame(y)
    N = len(x)
    predictions_training = list(range(K))
    predictions_validation = list(range(K))
    MAEs_training = list(range(K))
    MAEs_validation = list(range(K))
    for i in range(K):
        starting_index = int(i*(N/K))
        ending_index = int((i+1)*(N/K))
        validation_set = x[starting_index:ending_index]
        validation_set = validation_set.reset_index(drop=True)
        validation_set_output = y[starting_index:ending_index]
        validation_set_output = validation_set_output.reset_index(drop=True)
        training_set = x.copy()
        training_set_output = y.copy()
        training_set.drop(training_set.index[starting_index:ending_index], inplace=True)
        training_set = training_set.reset_index(drop=True)
        training_set_output.drop(training_set_output.index[starting_index:ending_index], inplace=True)
        training_set_output = training_set_output.reset_index(drop=True)
        ridge = lin_mod.Ridge(lambda_value)
        ridge.fit(training_set, training_set_output)
        predictions_validation[i] = ridge.predict(validation_set)
        predictions_training[i] = ridge.predict(training_set)
        errors_validation = [abs(predictions_validation[i][j] - validation_set_output.iloc[j]) for j in range(len(validation_set))]
        errors_training = [abs(predictions_training[i][j] - training_set_output.iloc[j]) for j in range(len(training_set))]
        MAEs_validation[i] = sum(errors_validation) / len(validation_set)
        MAEs_training[i] = sum(errors_training) / len(training_set)
    train_error = sum(MAEs_training) / K
    cv_error = sum(MAEs_validation) / K
    return [train_error, cv_error]


# ADAPTED FROM LAB 5
# k-fold cross-validation
# Inputs:
#  x: training input
#  y: training output
#  K: number of folds
#  lambda_value: constant that controls strength of regularization
# Outputs:
#  train_error: average MAE of training set across all K folds
#  cv_error: average MAE of validation set across all K folds
def kfoldCV_lasso(x, y, K, lambda_value):
    x = pd.DataFrame(x)
    y = pd.DataFrame(y)
    N = len(x)
    predictions_training = list(range(K))
    predictions_validation = list(range(K))
    MAEs_training = list(range(K))
    MAEs_validation = list(range(K))
    for i in range(K):
        starting_index = int(i*(N/K))
        ending_index = int((i+1)*(N/K))
        validation_set = x[starting_index:ending_index]
        validation_set = validation_set.reset_index(drop=True)
        validation_set_output = y[starting_index:ending_index]
        validation_set_output = validation_set_output.reset_index(drop=True)
        training_set = x.copy()
        training_set_output = y.copy()
        training_set.drop(training_set.index[starting_index:ending_index], inplace=True)
        training_set = training_set.reset_index(drop=True)
        training_set_output.drop(training_set_output.index[starting_index:ending_index], inplace=True)
        training_set_output = training_set_output.reset_index(drop=True)
        lasso = lin_mod.Lasso(lambda_value)
        lasso.fit(training_set, training_set_output)
        predictions_validation[i] = lasso.predict(validation_set)
        predictions_training[i] = lasso.predict(training_set)
        errors_validation = [abs(predictions_validation[i][j] - validation_set_output.iloc[j]) for j in range(len(validation_set))]
        errors_training = [abs(predictions_training[i][j] - training_set_output.iloc[j]) for j in range(len(training_set))]
        MAEs_validation[i] = sum(errors_validation) / len(validation_set)
        MAEs_training[i] = sum(errors_training) / len(training_set)
    train_error = sum(MAEs_training) / K
    cv_error = sum(MAEs_validation) / K
    return [train_error, cv_error]


def run():
    # print("***** ZEROES *****")
    # stats.check_claim_amount_mae(np.zeros(len(stats.claim_amounts_answers)))
    # stats.check_claim_or_not(np.zeros(len(stats.boolean_answers)))
    # stats.check_overall_mae(np.zeros(len(stats.all_data_answers)))
    # create_submission(TestModelZeroes, 1, 1, False)
    #
    # print("***** LINEAR REGRESSION *****")
    # lin_reg_model = TestModelLinearRegression()
    # create_submission(lin_reg_model, 1, 2, True)
    #
    # print("***** TF THEN LINEAR REGRESSION *****")
    # tf_lin_reg_model = TestModelTFLinearRegression()
    # create_submission(tf_lin_reg_model, 1, 9, True)

    print("***** TF THEN RIDGE *****")
    tf_ridge_model = TestModelTFRidge()
    create_submission(tf_ridge_model, 2, 1, True)

    print("***** RANDOM FOREST THEN LASSO *****")
    random_forest_lasso_model = TestModelRandomForestLasso()
    create_submission(random_forest_lasso_model, 2, 2, True)


if __name__ == "__main__":
    run()
