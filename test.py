
import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.feature_selection import VarianceThreshold

df = pd.read_csv("datasets/trainingset.csv")
X = df.drop("ClaimAmount", axis=1, inplace=False)
y = df.loc[:, "ClaimAmount"]
print(X.shape)
norm = Normalizer().fit(X)
norm_X_train = norm.transform(X)
print(norm_X_train.var(axis=0))


selector = VarianceThreshold(threshold=1e-7)
selected_features = selector.fit_transform(norm_X_train)
print(selected_features.shape)


# from sklearn.feature_selection import RFE
#
# RFE_selector = RFE(estimator=dt, n_features_to_select=4, step=1) # how do we get the estimator here?
# RFE_selector.fit(X, y)
# print(X.columns[RFE_selector.support_])
