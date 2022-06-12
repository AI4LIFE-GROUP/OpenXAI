import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    
    filePath = 'Data_Sets/Adult/'
    testFile = 'adult_val.csv'
    trainFile = 'adult.csv'
    
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                    'income']
    categorical_features = ['sex', 'workclass', 'marital-status', 'occupation', 'relationship', 'race',
                            'native-country']
    
    # Read Data from csv
    test_df = pd.read_csv(filePath + testFile, skipinitialspace=True, header=None)
    train_df = pd.read_csv(filePath + trainFile, skipinitialspace=True, header=None)
    # if data is already cleansed, don't do anything
    
    all = pd.read_csv(filePath + 'adult_full.csv', skipinitialspace=True)
    all = pd.get_dummies(all, columns=categorical_features, drop_first=True)
    y = all['income']
    all = all.drop(columns=['income'])
    all['income'] = y
    X_train, X_test = train_test_split(all, stratify=all['income'], test_size=0.20)

    X_train.to_csv('Data_Sets/Adult/adult-train.csv', index=False)
    X_test.to_csv('Data_Sets/Adult/adult-test.csv', index=False)

    if test_df.shape[1] <= len(column_names):
        test_df.columns = column_names
        train_df.columns = column_names
    
        # Concatenate test and train set to one dataframe to remain consistency
        df = pd.concat([test_df, train_df])
        # convert all '?' to NaN and drop those rows
        df = df[df != '?']
        df.dropna(axis=0, inplace=True)
    
        # convert binary features in 0 and 1
        # <=50K: 0; >50K: 1
        df['income'] = df['income'].str.rstrip('.')  # Original test data contains '.' as last element in column income
        #process.categorize_binary(df, 'income', '>50K')
    
        # simplify categories from workclass
        workclass_map = {'Private': 'Private',
                          'Self-emp-not-inc': 'Non-Private',
                          'Local-gov': 'Non-Private',
                          'State-gov': 'Non-Private',
                          'Self-emp-inc': 'Non-Private',
                          'Federal-gov': 'Non-Private',
                          'Without-pay': 'Non-Private'}
        df['workclass'] = df['workclass'].map(workclass_map)
    
        # change marital status column
        df['marital-status'] = df['marital-status'].replace(['Divorced', 'Married-spouse-absent',
                                                             'Separated', 'Widowed', 'Never-married'], 'Non-Married')
        df['marital-status'] = df['marital-status'].replace(['Married-AF-spouse', 'Married-civ-spouse'], 'Married')
    
        # simplify occupation column (roughly high skilled vs. low-skilled job)
        occ_map = {'Adm-clerical': 'Managerial-Specialist', 'Armed-Forces': 'Other', 'Craft-repair': 'Other',
                   'Exec-managerial': 'Managerial-Specialist', 'Farming-fishing': 'Other', 'Handlers-cleaners': 'Other',
                   'Machine-op-inspct': 'Managerial-Specialist', 'Other-service': 'Other', 'Priv-house-serv': 'Other',
                   'Prof-specialty': 'Managerial-Specialist', 'Protective-serv': 'Other', 'Sales': 'Other',
                   'Tech-support': 'Other', 'Transport-moving': 'Other'}
    
        df['occupation'] = df['occupation'].map(occ_map)
    
    
        # drop education column
        df = df.drop(columns=['education'], axis=1)
        
        # change race column
        race_map = {'White': 'White', 'Amer-Indian-Eskimo': 'Non-White', 'Asian-Pac-Islander': 'Non-White',
                    'Black': 'Non-White', 'Other': 'Non-White'}
        df['race'] = df['race'].map(race_map)
        
        # change relationship column
        rel_map = {'Unmarried': 'Non-Husband', 'Wife': 'Non-Husband', 'Husband': 'Husband',
                   'Not-in-family': 'Non-Husband', 'Own-child': 'Non-Husband', 'Other-relative': 'Non-Husband'}
        df['relationship'] = df['relationship'].map(rel_map)
    
        # change 'native-country' to US vs Non-US
        data = [df]
        for d in data:
            d.loc[d['native-country'] != 'United-States', 'native-country'] = 'Non-US'
            d.loc[d['native-country'] == 'United-States', 'native-country'] = 'US'
    
        # One hot encode all categorical features
        df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    
        # save cleansed data in edited folder
        df.to_csv(filePath + 'adult_full.csv', index=False)
    
        # save column names of categorical data
        with open(filePath + 'cat_names.txt', 'wb') as f:
            pickle.dump(categorical_features, f)
            

if __name__ == "__main__":

    main()
