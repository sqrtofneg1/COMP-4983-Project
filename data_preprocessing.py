import pandas as pd


def load(filepath):
    """
    load and encode categorical data
    Inputs:
    filepath: path to the csv file
    Outputs:
    data: a DataFrame object containing encoded data
    """
    unencoded = pd.read_csv(filepath, low_memory=False)
    categorical = []
    for column in unencoded:
        if unencoded[column].nunique() <= 20 or unencoded[column].dtypes == object:
            categorical.append(column)
    try:
        categorical.remove("ClaimAmount")
    except:
        pass

    trans_data = pd.get_dummies(unencoded, columns=categorical)
    print(trans_data)
    sorted_columns = []
    column_order = {}

    for column in unencoded:
        index = unencoded.columns.get_loc(column)
        if column in trans_data:
            column_order.update({column: index})
        else:
            for c in trans_data:
                if column in c:
                    column_order.update({c: index})

    sorted_dict = sorted(column_order.items(), key=lambda x: x[1])
    # print(sorted_dict)
    for item in sorted_dict:
        sorted_columns.append(item[0])

    data = trans_data.reindex(sorted_columns, axis=1)
    # print(data.columns)

    return data