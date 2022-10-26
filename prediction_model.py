import numpy as np
import pandas as pd
from create_submission_csv import create_submission
from data_preprocessing import load
from sklearn.linear_model import LinearRegression


# SUNMIN CODE START
class TestModelZeroes:

    @staticmethod
    def predict(testdata):
        return [0] * len(testdata)


class TestModelLinearRegression:

    def __init__(self, tolerance):
        self.tolerance = tolerance
        lin_reg = LinearRegression()
        boolean_data = pd.read_csv("datasets/trainingset_boolean_claim_amount.csv")
        boolean_features = boolean_data.drop("ClaimAmount", axis=1, inplace=False)
        boolean_labels = boolean_data.loc[:, "ClaimAmount"]
        self.claim_or_not_model = lin_reg.fit(boolean_features, boolean_labels)

        lin_reg_2 = LinearRegression()
        continuous_data = pd.read_csv("datasets/trainingset_claim_amounts_only.csv")
        continuous_features = continuous_data.drop("ClaimAmount", axis=1, inplace=False)
        continuous_labels = continuous_data.loc[:, "ClaimAmount"]
        self.claim_amount_model = lin_reg_2.fit(continuous_features, continuous_labels)

    def predict(self, features):
        predictions_claim_or_not = self.claim_or_not_model.predict(features)
        predictions_claim_amount = self.claim_amount_model.predict(features)
        for i in range(len(features)):
            if predictions_claim_or_not[i] < self.tolerance:  # tune tolerance value to be closer to actual ratio
                predictions_claim_amount[i] = 0
        return predictions_claim_amount


create_submission(TestModelZeroes, 1, 1, False)
model = TestModelLinearRegression(0.1)
create_submission(model, 1, 2, True)

# SUNMIN CODE END
