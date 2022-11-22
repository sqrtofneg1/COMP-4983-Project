import pandas as pd
from data_preprocessing import load


def create_submission(model, raw_data, userinput):
    """
    Creates a .csv file for submission.
     param model: a model that has a 'predict' function
     param checkpoint_num: checkpoint number from 1 to 4.
     param submission_num: submission number from 1 to 10.
     param raw_data: true if you want to use the raw dataset to test, false if you want to use one hot encoded data
    """
    if raw_data:
        testdata = pd.read_csv("datasets/competitionset.csv")
    else:
        testdata = load("datasets/competitionset.csv")
    predictions = model.predict(testdata)
    predictions_data_frame = pd.DataFrame(predictions, columns=["ClaimAmount"])
    predictions_data_frame.to_csv(f"submissions/predictedclaimamount.csv", index_label="rowIndex")
