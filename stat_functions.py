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
    false_negatives = 0
    true_positives = 0
    true_negatives = 0
    for i in range(len(preds)):
        if preds[i] == 0:
            if boolean_answers[i] == 1:
                false_negatives += 1
            else:
                true_negatives += 1
        if preds[i] == 1:
            if boolean_answers[i] == 0:
                false_positives += 1
            else:
                true_positives += 1
    f1_score = true_positives / (true_positives + 0.5 * (false_positives + false_negatives))
    print(" *** ClaimAmount Identification Stats ***")
    print(f" *F1 Score: {f1_score:.6f}")
    print(f"  True positives: {true_positives}")
    print(f"  True negatives: {true_negatives}")
    print(f"  False positives: {false_positives}")
    print(f"  False negatives: {false_negatives}")
    print(f"  Accuracy: {(len(preds) - false_positives - false_negatives) / len(preds) * 100:.3f}%")
    if true_positives + false_positives == 0:
        print(f"  Precision: NaN")
    else:
        print(f"  Precision: {true_positives / (true_positives + false_positives) * 100:.3f}%")
    print(f"  Sensitivity/Recall: {true_positives / num_expected_claim_amounts * 100:.3f}%")
    print(f"  Specificity: {true_negatives / (len(preds) - num_expected_claim_amounts) * 100:.3f}%")
    print(f"  False Positive Rate: {false_positives / (len(preds) - num_expected_claim_amounts) * 100:.3f}%")
    print(f"  False Negative Rate/MissRate: {false_negatives / num_expected_claim_amounts * 100:.3f}%")

    print()


def check_claim_amount_mae(preds):
    if len(preds) != len(claim_amounts_answers):
        return
    print(" *** ClaimAmount Value Prediction Stats ***")
    mae = np.mean(abs(preds - claim_amounts_answers))
    print(f"  MAE: {mae}")
    print()
    return mae


def check_overall_mae(preds):
    if len(preds) != len(all_data_answers):
        return
    print(" *** Overall Prediction Stats ***")
    mae = np.mean(abs(preds - all_data_answers))
    print(f"  Overall MAE: {mae:.4f}")
    print()
    return mae
