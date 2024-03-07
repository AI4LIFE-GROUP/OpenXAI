import os
from io import StringIO
import requests
import torch
import pandas as pd
from errno import EEXIST
import torch.utils.data as data
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler

dataverse_prefix = 'https://dataverse.harvard.edu/api/access/datafile/'
dataverse_ids = {
    'train': {
        'adult': '8550940', 'compas': '8550936', 'gaussian': '8550929', 'german': '8550931',
        'gmsc': '8550934', 'heart': '8550932', 'heloc': '8550942', 'pima': '8550937',
    },
    'test': {
        'adult': '8550933', 'compas': '8550944', 'gaussian': '8550941', 'german': '8550930',
        'gmsc': '8550939', 'heart': '8550935', 'heloc': '8550943', 'pima': '8550938',
    }
}

feature_types = {
    'adult': ['c'] * 6 + ['d'] * 7, 'german': ['c'] * 8 + ['d'] * 12,
    'compas': ['c', 'd', 'c', 'c', 'd', 'd', 'd'], 'gaussian': ['c'] * 20,
    'gmsc': ['c'] * 10, 'heloc': ['c'] * 23, 'pima': ['c'] * 8,
    'heart': ['d', 'c', 'c', 'd', 'c'] + ['d'] * 4 + ['c'] * 6,
}
labels = {'adult': 'income', 'compas': 'risk', 'gaussian': 'target', 'german': 'credit-risk',
          'gmsc': 'SeriousDlqin2yrs', 'heart': 'TenYearCHD', 'heloc': 'RiskPerformance', 'pima': 'Outcome'}

class TabularDataLoader(data.Dataset):
    def __init__(self, path, filename, label, download=False, scale='minmax'):
            
        """
        Load training dataset
        :param path: string with path to training set
        :param filename: string with name of file
        :param label: string, column name for label
        :param scale: string; 'minmax', 'standard', or 'none'
        :return: tensor with training data
        """
        self.data_name, self.split = filename.split('.')[0].split('-')
        if download or not os.path.isfile(path + filename):
            self.mkdir_p(path)
            r = requests.get(dataverse_prefix + dataverse_ids[self.split][self.data_name], allow_redirects=True)
            df = pd.read_csv(StringIO(r.text), sep='\t')
            df.to_csv(path + filename, index=False)

        if not os.path.isfile(path + filename):
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self.dataset = pd.read_csv(path + filename)
        self.target = label

        # Split data into features and target
        self.X = self.dataset.drop(self.target, axis=1)
        self.feature_names = self.X.columns.to_list()
        self.feature_types = feature_types[self.data_name]
        if self.data_name == 'german':
            self.feature_metadata = {
                'feature_types': self.feature_types,
                'feature_n_cols': [1, 1, 1, 1, 1, 1, 1, 1, 4, 5, 10, 5, 5, 4, 3, 4, 3, 3, 4, 2]
            }
        else:
            self.feature_metadata = self.feature_types
        self.target_name = label

        # Transform data
        if scale == 'minmax':
            self.scaler = MinMaxScaler()
        elif scale == 'standard':
            self.scaler = StandardScaler()
        elif scale == 'none':
            self.scaler = None
        else:
            raise NotImplementedError('The current version of DataLoader class only provides the following transformations: {minmax, standard, none}')
            
        if self.scaler is not None:
            self.scaler.fit_transform(self.X)
            self.data = self.scaler.transform(self.X)
        else:
            self.data = self.X.values

        self.targets = self.dataset[self.target]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        idx = idx.tolist() if isinstance(idx, torch.Tensor) else idx
        return (self.data[idx], self.targets.values[idx])

    def get_number_of_features(self):
        return self.data.shape[1]
    
    def get_number_of_instances(self):
        return self.data.shape[0]

    def mkdir_p(self, mypath):
        """Creates a directory. equivalent to using mkdir -p on the command line"""
        try:
            os.makedirs(mypath)
        except OSError as exc:  # Python >2.5
            if exc.errno == EEXIST and os.path.isdir(mypath):
                pass
            else:
                raise 


def ReturnLoaders(data_name, download=False, batch_size=32, scaler='minmax'):
    """
    Load training and test datasets as DataLoader objects
    :param data_name: string with name of dataset
    :param download: boolean, whether to download the dataset
    :param batch_size: int, batch size
    :param scaler: string; 'minmax', 'standard', or 'none'
    :return: tuple with training and test dataloaders
    """
    
    prefix = f'./data/{data_name}/'
    file_train, file_test = data_name + '-train.csv', data_name + '-test.csv'

    dataset_train = TabularDataLoader(path=prefix, filename=file_train, label=labels[data_name],
                                      scale=scaler, download=download)

    dataset_test = TabularDataLoader(path=prefix, filename=file_test, label=labels[data_name],
                                     scale=scaler, download=download)

    trainloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    
    return trainloader, testloader

def ReturnTrainTestX(data_name, n_test=None, n_train=None, download=False,
                             float_tensor=False, return_feature_metadata=False):
    """
    Load training and test datasets as DataLoader objects
    :param data_name: string with name of dataset
    :param n_test_samples: int, number of test samples
    :param n_train_samples: int, number of train samples
    :param download: boolean, whether to download the dataset
    :param float_tensor: boolean, whether to convert to FloatTensor
    :param return_feature_metadata: boolean, whether to return feature metadata
    :return: tuple with training and test inputs (optionally feature metadata)
    """
    trainloader, testloader = ReturnLoaders(data_name, download=download)
    X_test = testloader.dataset.data[:n_test] if n_test is not None else testloader.dataset.data
    X_train = trainloader.dataset.data[:n_train] if n_train is not None else trainloader.dataset.data
    if float_tensor:
        X_test = torch.FloatTensor(X_test)
        X_train = torch.FloatTensor(X_train)
    if return_feature_metadata:
        return X_train, X_test, trainloader.dataset.feature_metadata
    return X_train, X_test