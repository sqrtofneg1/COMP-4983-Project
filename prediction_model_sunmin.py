import numpy as np
import pandas as pd
from create_submission_csv import create_submission
from data_preprocessing import load
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import prediction_model_sunmin_tensorflow as tensorflow_model
import stat_functions as stats


boolean_answers = pd.read_csv("datasets/trainingset_boolean_claim_amount.csv").loc[:, "ClaimAmount"]
num_expected_claim_amounts = 3335
claim_amounts_answers = pd.read_csv("datasets/trainingset_claim_amounts_only.csv").loc[:, "ClaimAmount"]
all_data = pd.read_csv("datasets/trainingset.csv")
all_data_answers = all_data.loc[:, "ClaimAmount"]
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
            self.tolerance = sorted_arr[-num_expected_claim_amounts]
        print(f"TOLERANCE: {self.tolerance}")

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


def run():
    print("***** ZEROES *****")
    stats.check_claim_amount_mae(np.zeros(len(claim_amounts_answers)))
    stats.check_claim_or_not(np.zeros(len(boolean_answers)))
    stats.check_overall_mae(np.zeros(len(all_data_answers)))
    create_submission(TestModelZeroes, 1, 1, False)

    print("***** LINEAR REGRESSION *****")
    lin_reg_model = TestModelLinearRegression()
    create_submission(lin_reg_model, 1, 2, True)

    print("***** TF THEN LINEAR REGRESSION *****")
    tf_lin_reg_model = TestModelTFLinearRegression()
    create_submission(tf_lin_reg_model, 1, 9, True)

# Decision tree for the boolean part?


if __name__ == "__main__":
    run()
