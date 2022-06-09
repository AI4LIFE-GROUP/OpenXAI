import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

from data_cleaning_utils import convert_categorical_cols

# German credit feature types
# https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
german_feature_types = ['c', 'd', 'c', 'c', 'd', 'c', 'c', 'd', 'c', 
                 'c', 'd', 'c', 'd', 'c', 'c', 'd', 'c', 'd', 'c']

label_col = 'credit-risk'

def main():
    filePath = '../Data_Sets/German_Credit_Data/'
    data_name = 'german.csv'

    # Read Data from csv
    all = pd.read_csv(f'{filePath}{data_name}')
    
    # drop the label column, append to the end
    y_col = all[label_col]
    all = all.drop(label_col, axis=1)

    X_train, X_test = train_test_split(all, stratify=y_col, test_size=0.20, random_state = 0)

    train_df = pd.DataFrame(X_train)
    test_df = pd.DataFrame(X_test)

    train_ind = train_df.shape[0]

    # merge them
    data_df = pd.concat([train_df.reset_index(drop=True), test_df.reset_index(drop=True)], axis=0)

    # get dummies for the categorical cols
    # instead of pd.get_dummies, make own dummy-creator fn to store information 

    cols = np.array(data_df.columns)
    cat_cols = cols[np.argwhere(np.array(german_feature_types) == 'c').flatten()]

    one_hot, feature_metadata = convert_categorical_cols(data = data_df, feature_types = german_feature_types, original_columns = cols, columns = cat_cols)
    
    # append the label column to the end of the dataframe
    one_hot[label_col] = y_col

    # split back into train and test, save
    X_train = one_hot.iloc[:train_ind]
    X_test = one_hot.iloc[train_ind:]

    X_train.to_csv(f'{filePath}german-train.csv', index=False)
    X_test.to_csv(f'{filePath}german-test.csv', index=False)
    pickle.dump(feature_metadata, open(f'{filePath}german-feature-metadata.p', 'wb'))
    
    print(X_train.columns)
    
if __name__ == "__main__":

    main()