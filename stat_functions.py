import pandas as pd
import numpy as np

boolean_answers = pd.read_csv("datasets/trainingset_boolean_claim_amount.csv").loc[:, "ClaimAmount"]
num_expected_claim_amounts = 3335
claim_amounts_answers = pd.read_csv("datasets/trainingset_claim_amounts_only.csv").loc[:, "ClaimAmount"]
all_data = pd.read_csv("datasets/trainingset.csv")
all_data_answers = all_data.loc[:, "ClaimAmount"]


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
    print(f"  Percentage correctly identified: "
          f"{(num_expected_claim_amounts - missed_positives) / num_expected_claim_amounts * 100:.3f}%")
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
