import numpy as np
import pandas as pd
from create_submission_csv import create_submission
from data_preprocessing import load
from sklearn.linear_model import LinearRegression, Ridge


#do later: find the most important features and implememnt polyval onto them, arbitrary degree?
#solve for best degree if have time
class TestModel:

    def __init__(self, degree):
        self.degree = degree

        claim_amount_data = pd.read_csv("datasets/trainingset_claim_amounts_only.csv")
        claim_amount_features = claim_amount_data.drop("ClaimAmount", axis=1, inplace=False)
        claim_amount_labels = claim_amount_data.loc[:, "ClaimAmount"]

        self.reg = np.polyfit(x=claim_amount_features, y=claim_amount_labels, deg=self.degree)

        self.predict(pd.read_csv("datasets/trainingset_claim_amounts_only.csv").drop("ClaimAmount", axis=1, inplace=False))
        self.predict(pd.read_csv("datasets/trainingset.csv").drop("ClaimAmount", axis=1, inplace=False))

    def predict(self, features):
        prediction = self.reg.polyval(features)
        return prediction


deg5 = TestModel(5)
deg25 = TestModel(25)

create_submission(deg5, 1, 5, True)
create_submission(deg25, 1, 6, True)