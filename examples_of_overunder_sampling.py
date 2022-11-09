from numpy import mean
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
data = pd.read_csv('datasets/trainingset_boolean_claim_amount.csv')
X = data.drop("ClaimAmount", axis=1, inplace=False)
y = data.loc[:, "ClaimAmount"]

# Random oversampling of boolean dataset
# define pipeline
over = RandomOverSampler(sampling_strategy=0.1)
steps = [('over', over), ('model', DecisionTreeClassifier())]
pipeline = Pipeline(steps=steps)
# evaluate pipeline
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
score = mean(scores)
print('Over with decision tree F1 Score: %.3f' % score)

# Random undersampling of boolean dataset
# define pipeline
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('under', under), ('model', DecisionTreeClassifier())]
pipeline = Pipeline(steps=steps)
# evaluate pipeline
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
score = mean(scores)
print('Under with decision tree F1 Score: %.3f' % score)

# Random oversampling and undersampling combined
# define pipeline
over = RandomOverSampler(sampling_strategy=0.1)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('o', over), ('u', under), ('m', DecisionTreeClassifier())]
pipeline = Pipeline(steps=steps)
# evaluate pipeline
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
score = mean(scores)
print('Combined with decision tree F1 Score: %.3f' % score)

# Random oversampling of boolean dataset
# define pipeline
over = RandomOverSampler(sampling_strategy=0.1)
steps = [('over', over), ('model', RandomForestClassifier(n_estimators=100, random_state=100))]
pipeline = Pipeline(steps=steps)
# evaluate pipeline
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
score = mean(scores)
print('Over with random forest F1 Score: %.3f' % score)

# Random undersampling of boolean dataset
# define pipeline
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('under', under), ('model', RandomForestClassifier(n_estimators=100, random_state=100))]
pipeline = Pipeline(steps=steps)
# evaluate pipeline
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
score = mean(scores)
print('Under with random forest F1 Score: %.3f' % score)

# Random oversampling and undersampling combined
# define pipeline
over = RandomOverSampler(sampling_strategy=0.1)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('o', over), ('u', under), ('m', RandomForestClassifier(n_estimators=100, random_state=100))]
pipeline = Pipeline(steps=steps)
# evaluate pipeline
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
score = mean(scores)
print('Combined with random forest F1 Score: %.3f' % score)
