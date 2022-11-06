from numpy import mean
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from create_submission_csv import create_submission
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from imblearn.pipeline import Pipeline
import stat_functions as stats
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd

boolean_answers = pd.read_csv("datasets/trainingset_boolean_claim_amount.csv").loc[:, "ClaimAmount"]
num_expected_claim_amounts = 3335
claim_amounts_answers = pd.read_csv("datasets/trainingset_claim_amounts_only.csv").loc[:, "ClaimAmount"]
all_data = pd.read_csv("datasets/trainingset.csv")
all_data_answers = all_data.loc[:, "ClaimAmount"]
boolean_data = pd.read_csv('datasets/trainingset_boolean_claim_amount.csv')
X = boolean_data.drop("ClaimAmount", axis=1, inplace=False)
y = boolean_data.loc[:, "ClaimAmount"]
continuous_data = pd.read_csv("datasets/trainingset_claim_amounts_only.csv")
continuous_features = continuous_data.drop("ClaimAmount", axis=1, inplace=False)
continuous_labels = continuous_data.loc[:, "ClaimAmount"]


undersample = RandomUnderSampler(sampling_strategy=0.5)
oversample = RandomOverSampler(sampling_strategy=0.5)

X_over, y_over = oversample.fit_resample(X, y)
X_under, y_under = undersample.fit_resample(X, y)

class UnderSamplerDecisionTree:

    def __init__(self, tolerance):
        self.tolerance = tolerance
        treeclass = DecisionTreeClassifier()
        self.boolean_claim_model = treeclass.fit(X_under, y_under)
        treereg = DecisionTreeRegressor()
        self.claim_amount_model = treereg.fit(continuous_features, continuous_labels)

        # For getting the stats for check_claim_amount_mae
        self.predict(continuous_features)
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

# def run():
print("***** tree *****")
model = UnderSamplerDecisionTree(0.1)
create_submission(model, 2, 3, True)

