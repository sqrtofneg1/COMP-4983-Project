import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from create_submission_csv import create_submission
import stat_functions as stats
import pickle

boolean_answers = pd.read_csv("datasets/trainingset_boolean_claim_amount.csv").loc[:, "ClaimAmount"]
num_expected_claim_amounts = 3335
claim_amounts_answers = pd.read_csv("datasets/trainingset_claim_amounts_only.csv").loc[:, "ClaimAmount"]
all_data = pd.read_csv("datasets/trainingset.csv")
all_data_answers = all_data.loc[:, "ClaimAmount"]
all_data = pd.read_csv("datasets/trainingset.csv")

boolean_data = pd.read_csv('datasets/trainingset_boolean_claim_amount.csv')
boolean_data = boolean_data.iloc[np.random.permutation(boolean_data.index)].reset_index(drop=True)
continuous_data = pd.read_csv("datasets/trainingset_claim_amounts_only.csv")
continuous_data = continuous_data.iloc[np.random.permutation(continuous_data.index)].reset_index(
    drop=True)

true_continuous_features = continuous_data.drop("ClaimAmount", axis=1, inplace=False)
true_continuous_labels = continuous_data.loc[:, "ClaimAmount"]


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

    def __init__(self):
        X = boolean_data.drop("ClaimAmount", axis=1, inplace=False)
        y = boolean_data.loc[:, "ClaimAmount"]
        over = RandomOverSampler(sampling_strategy=0.1)
        under = RandomUnderSampler(sampling_strategy=0.5)
        X, y = over.fit_resample(X, y)
        self.X, self.y = under.fit_resample(X, y)

    def values(self):
        return self.X, self.y


if __name__ == "__main__":
    userinput = input("please enter filepath to competitionset.csv\n")
    # under_over = UnderOverSamplerData()
    # X_under_over, y_under_over = under_over.values()
    # model = DecisionTree(0.1, X_under_over, y_under_over)
    # with open("final_model.pkl", "wb") as outp:
    #     pickle.dump(model, outp)
    with open("final_model.pkl", "rb") as inp:
        model = pickle.load(inp)
        create_submission(model, True, userinput)
    create_submission(model, True, userinput)
