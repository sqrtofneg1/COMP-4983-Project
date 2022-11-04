import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD


def get_tensorflow_model_boolean():
    boolean_data = pd.read_csv("datasets/trainingset_boolean_claim_amount.csv")
    boolean_features = boolean_data.drop("ClaimAmount", axis=1, inplace=False)
    boolean_features = StandardScaler().fit_transform(boolean_features)
    boolean_labels = boolean_data.loc[:, "ClaimAmount"]

    num_features = boolean_features.shape[1]

    class_weights = {0: 1, 1: 19}

    model = Sequential()
    model.add(Input(shape=(num_features,)))
    model.add(Dense(800, activation="tanh"))
    model.add(Dense(100, activation="tanh"))
    model.add(Dense(2, activation="softmax"))

    model.summary()

    optimizer = SGD(learning_rate=0.1)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics="accuracy")

    model.fit(boolean_features, boolean_labels, epochs=10, verbose=1, class_weight=class_weights)

    return model
