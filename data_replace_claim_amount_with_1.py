import pandas as pd

data = pd.read_csv("datasets/trainingset.csv")

data["ClaimAmount"] = (data["ClaimAmount"] > 0).astype(int)

data.to_csv("datasets/trainingset_boolean_claim_amount.csv")
