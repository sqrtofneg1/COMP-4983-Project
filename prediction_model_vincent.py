import numpy as np
import pandas as pd
from create_submission_csv import create_submission
from data_preprocessing import load
from sklearn.linear_model import LinearRegression

boolean_answers = pd.read_csv("datasets/trainingset_boolean_claim_amount.csv").loc[:, "ClaimAmount"]
num_expected_claim_amounts = 3335
claim_amounts_answers = pd.read_csv("datasets/trainingset_claim_amounts_only.csv").loc[:, "ClaimAmount"]
all_data_answers = pd.read_csv("datasets/trainingset.csv").loc[:, "ClaimAmount"]

# copied base code. Next Step: Choose features and modify code to get prediction


class TestModelLinearRegression:

    def __init__(self, tolerance):
        self.tolerance = tolerance
        lin_reg_1 = LinearRegression()
        boolean_data = pd.read_csv("datasets/trainingset_boolean_claim_amount.csv")
        boolean_features = boolean_data.drop("ClaimAmount", axis=1, inplace=False)
        boolean_labels = boolean_data.loc[:, "ClaimAmount"]

        self.boolean_model = lin_reg_1.fit(boolean_features, boolean_labels)

        lin_reg_2 = LinearRegression()
        claim_amount_data = pd.read_csv("datasets/trainingset_claim_amounts_only.csv")
        claim_amount_features = claim_amount_data.drop("ClaimAmount", axis=1, inplace=False)
        claim_amount_labels = claim_amount_data.loc[:, "ClaimAmount"]

        self.claim_model = lin_reg_2.fit(claim_amount_features, claim_amount_labels)

        self.predict(pd.read_csv("datasets/trainingset_claim_amounts_only.csv").drop("ClaimAmount", axis=1, inplace=False))
        self.predict(pd.read_csv("datasets/trainingset.csv").drop("ClaimAmount", axis=1, inplace=False))

    def predict(self, features):
        features = [features]                   # placeholder
        predictions_claim_or_not_raw = self.claim_or_not_model.predict(features)
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


# execute
lin_reg = TestModelLinearRegression(0.1)
# create_submission(lin_reg, 1, 7, False)

