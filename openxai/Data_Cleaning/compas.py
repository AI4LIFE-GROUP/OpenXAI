import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

class Params():
    """Parameters object taken from: https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/nlp/utils.py

    Parameters
    ----------
    json_path : string

    Returns
    ----------
    Parameters object
    """
    
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)
    
    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
    
    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)
    
    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


def get_and_preprocess_compas_data(params):
    """Handle processing of COMPAS according to: https://github.com/propublica/compas-analysis

    Parameters
    ----------
    params : Params

    Returns
    ----------
    Pandas data frame X of processed data, np.ndarray y, and list of column names
    """
    PROTECTED_CLASS = params.protected_class
    UNPROTECTED_CLASS = params.unprotected_class
    POSITIVE_OUTCOME = params.positive_outcome
    NEGATIVE_OUTCOME = params.negative_outcome
    
    compas_df = pd.read_csv("Data_Sets/COMPAS/compas-scores-2years.csv", index_col=0)
    compas_df = compas_df.loc[(compas_df['days_b_screening_arrest'] <= 30) &
                              (compas_df['days_b_screening_arrest'] >= -30) &
                              (compas_df['is_recid'] != -1) &
                              (compas_df['c_charge_degree'] != "O") &
                              (compas_df['score_text'] != "NA")]
    
    compas_df['length_of_stay'] = (
                pd.to_datetime(compas_df['c_jail_out']) - pd.to_datetime(compas_df['c_jail_in'])).dt.days
    X = compas_df[['age', 'two_year_recid', 'c_charge_degree', 'race', 'sex', 'priors_count', 'length_of_stay']]
    
    # if person has high score give them the _negative_ model outcome
    y = np.array([NEGATIVE_OUTCOME if score == 'High' else POSITIVE_OUTCOME for score in compas_df['score_text']])
    sens = X.pop('race')
    
    # assign African-American as the protected class
    X = pd.get_dummies(X)
    sensitive_attr = np.array(pd.get_dummies(sens).pop('African-American'))
    X['race'] = sensitive_attr
    
    # make sure everything is lining up
    assert all((sens == 'African-American') == (X['race'] == PROTECTED_CLASS))
    cols = [col for col in X]
    
    return X, y, cols



def main():
    
    params = Params("Data_Sets/COMPAS/model_configurations/experiment_params.json")
    np.random.seed(params.seed)
    X, y, cols = get_and_preprocess_compas_data(params)
    
    features = [c for c in X]
    
    X = X.drop(columns=['sex_Male', 'c_charge_degree_M'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.20)

    X_train['risk'] = y_train
    X_test['risk'] = y_test

    X_train.to_csv('Data_Sets/COMPAS/compas-scores-train.csv', index=False)
    X_test.to_csv('Data_Sets/COMPAS/compas-scores-test.csv', index=False)

    
if __name__ == "__main__":
    # execute data preprocessing
    main()