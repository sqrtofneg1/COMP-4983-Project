import numpy as np
import pandas as pd
from create_submission_csv import create_submission
from data_preprocessing import load
from sklearn.linear_model import LinearRegression, Ridge
import stat_functions as stats


class TestModelPoly:

    def __init__(self, degree):
        self.degree = degree

        claim_amount_data = pd.read_csv("datasets/trainingset_claim_amounts_only.csv")
        claim_amount_features = claim_amount_data.drop("ClaimAmount", axis=1, inplace=False)
        claim_amount_labels = claim_amount_data.loc[:, "ClaimAmount"]

        best_feature = self.find_best_feature(claim_amount_features, claim_amount_labels)

        self.best_feature_col = claim_amount_data.loc[:, f"feature{best_feature}"]

        self.reg = np.polyfit(x=self.best_feature_col, y=claim_amount_labels, deg=self.degree)

        self.predict(pd.read_csv("datasets/trainingset_claim_amounts_only.csv"))

    def predict(self, dataset):
        test_feature = dataset.loc[:, "feature1"]
        prediction = np.polyval(self.reg, test_feature)
        df = pd.DataFrame()
        df['ClaimAmount']=pd.Series(prediction)
        return df

    def find_best_feature(self, feats, labes):
        lg = LinearRegression()
        lg.fit(feats, labes)
        importance = lg.coef_
        scores = []
        for i, v in enumerate(importance):
            # print('Feature: %0d, Score: %5f' % (i, np.abs(v)))
            scores.append(np.abs(v))
        return scores.index(max(scores))

def run():
    deg5 = TestModelPoly(5)
    deg15 = TestModelPoly(15)

    create_submission(deg5, 1, 5, True)
    create_submission(deg15, 1, 6, True)

run()

