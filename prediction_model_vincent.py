import numpy as np
import pandas as pd
from create_submission_csv import create_submission
from data_preprocessing import load
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import stat_functions as stats

boolean_answers = pd.read_csv("datasets/trainingset_boolean_claim_amount.csv").loc[:, "ClaimAmount"]
num_expected_claim_amounts = 3335
claim_amounts_answers = pd.read_csv("datasets/trainingset_claim_amounts_only.csv").loc[:, "ClaimAmount"]
all_data_answers = pd.read_csv("datasets/trainingset.csv").loc[:, "ClaimAmount"]
features_set1 = ["feature1", "feature2", "feature6", "feature8", "feature10", "feature17"]
features_set2 = ["feature2", "feature8", "feature10", "feature12", "feature17"]
all_features = ["feature1", "feature2", "feature3", "feature4", "feature5", "feature6", "feature7", "feature8",
                "feature9", "feature10", "feature11", "feature12", "feature13", "feature14", "feature15",
                "feature16", "feature17", "feature18"]
categorical = ["feature3", "feature4", "feature5", "feature7", "feature9", "feature11",
               "feature13", "feature14", "feature15", "feature16", "feature18"]


class TestModelLinearRegression:

    def __init__(self, tolerance, features):
        self.tolerance = tolerance
        self.feature_set = features
        lin_reg_1 = LinearRegression()
        boolean_data = pd.read_csv("datasets/trainingset_boolean_claim_amount.csv")
        boolean_features = boolean_data.drop("ClaimAmount", axis=1, inplace=False)
        boolean_features.loc[:, self.feature_set] = StandardScaler().fit_transform(boolean_features.loc[:,
                                                                                   self.feature_set])
        boolean_labels = boolean_data.loc[:, "ClaimAmount"]

        self.boolean_model = lin_reg_1.fit(boolean_features, boolean_labels)

        lin_reg_2 = LinearRegression()
        claim_amount_data = pd.read_csv("datasets/trainingset_claim_amounts_only.csv")
        claim_amount_features = claim_amount_data.drop("ClaimAmount", axis=1, inplace=False)
        claim_amount_features.loc[:, self.feature_set] = StandardScaler().fit_transform(claim_amount_features.loc[:,
                                                                                        self.feature_set])
        claim_amount_labels = claim_amount_data.loc[:, "ClaimAmount"]

        self.claim_model = lin_reg_2.fit(claim_amount_features, claim_amount_labels)

        self.predict(pd.read_csv("datasets/trainingset_claim_amounts_only.csv").drop("ClaimAmount", axis=1, inplace=False))
        self.predict(pd.read_csv("datasets/trainingset.csv").drop("ClaimAmount", axis=1, inplace=False))

    def predict(self, features):
        features.loc[:, self.feature_set] = StandardScaler().fit_transform(features.loc[:, self.feature_set])
        predictions_claim_or_not_raw = self.boolean_model.predict(features)
        predictions_claim_or_not = [0] * len(features)
        for i in range(len(features)):
            if predictions_claim_or_not_raw[i] < self.tolerance:  # tune tolerance value to be closer to actual ratio
                predictions_claim_or_not[i] = 0
            else:
                predictions_claim_or_not[i] = 1
        stats.check_claim_or_not(predictions_claim_or_not)  # only fires on trainingset

        predictions_claim_amount_raw = self.claim_model.predict(features)
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
    # print("***** LINEAR REGRESSION #7 *****")
    # model = TestModelLinearRegression(0.1, features_set1)
    # create_submission(model, 1, 7, True)
    # print("***** LINEAR REGRESSION #8 *****")
    # model = TestModelLinearRegression(0.1, features_set2)
    # create_submission(model, 1, 8, True)

    print("***** LINEAR REGRESSION #7 *****")
    model = TestModelLinearRegression(0.1, features_set1)
    create_submission(model, 2, 7, True)
    print("***** LINEAR REGRESSION #8 *****")
    model = TestModelLinearRegression(0.1, features_set2)
    create_submission(model, 2, 8, True)


def main():
    run()


if __name__ == '__main__':
    main()
