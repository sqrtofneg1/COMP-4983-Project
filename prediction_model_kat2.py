from numpy import mean
from sklearn.ensemble import RandomForestClassifier
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
continuous_data = pd.read_csv("datasets/trainingset_claim_amounts_only.csv")
continuous_features = continuous_data.drop("ClaimAmount", axis=1, inplace=False)
continuous_labels = continuous_data.loc[:, "ClaimAmount"]


class DecisionTree:
    def __init__(self, tolerance, X, y):
        self.tolerance = tolerance
        forest = RandomForestClassifier(n_estimators=100, random_state=100)
        self.boolean_claim_model = forest.fit(X, y)
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


class OverSamplerData:

    def __init__(self):
        X = boolean_data.drop("ClaimAmount", axis=1, inplace=False)
        y = boolean_data.loc[:, "ClaimAmount"]
        oversample = RandomOverSampler(sampling_strategy=0.5)
        self.X, self.y = oversample.fit_resample(X, y)

    def values(self):
        return self.X, self.y


class UnderSamplerData:

    def __init__(self):
        X = boolean_data.drop("ClaimAmount", axis=1, inplace=False)
        y = boolean_data.loc[:, "ClaimAmount"]
        undersample = RandomUnderSampler(sampling_strategy=0.5)
        self.X, self.y = undersample.fit_resample(X, y)

    def values(self):
        return self.X, self.y


class UnderOverSamplerData:

    def __init__(self):
        X = boolean_data.drop("ClaimAmount", axis=1, inplace=False)
        y = boolean_data.loc[:, "ClaimAmount"]
        over = RandomOverSampler(sampling_strategy=0.1)
        under = RandomUnderSampler(sampling_strategy=0.5)
        X, y = over.fit_resample(X, y)
        self.X, self.y = under.fit_resample(X, y)

    def values(self):
        return self.X, self.y


def run():
    print("***** UNDERSAMPLE WITH DECISON TREE *****")
    under = UnderSamplerData()
    X_under, y_under = under.values()
    model = DecisionTree(0.1, X_under, y_under)
    create_submission(model, 2, 3, True)
    print("***** OVERSAMPLE WITH DECISION TREE *****")
    over = OverSamplerData()
    X_over, y_over = over.values()
    model = DecisionTree(0.1, X_over, y_over)
    create_submission(model, 2, 4, True)
    print("***** UNDER-OVERSAMPLE WITH DECISION TREE *****")
    under_over = UnderOverSamplerData()
    X_under_over, y_under_over = under_over.values()
    model = DecisionTree(0.1, X_under_over, y_under_over)
    create_submission(model, 2, 5, True)
