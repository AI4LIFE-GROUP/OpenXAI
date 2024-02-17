import csv
import io
import os
import subprocess
import torch
import pandas as pd
from errno import EEXIST
import torch.utils.data as data
from torch.utils.data import DataLoader
from urllib.request import urlretrieve
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def download_file(url, filename):
    # Download the file from the URL
    subprocess.call(["wget", "-O", filename, url])

    with open(filename, "r") as f:
        data = f.read()

    # Detect the file format
    if '\t' in data:  # if the file is tab delimited
        # Convert the file to CSV format
        data = io.StringIO(data)
        reader = csv.reader(data, delimiter='\t')
        output = io.StringIO()
        writer = csv.writer(output)
        for row in reader:
            writer.writerow(row)
        data = output.getvalue()

        # Save the file to disk
        with open(filename, 'w', newline='') as f:
            f.write(data)


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

        self.path = path

        if download:
            self.mkdir_p(path)
            # ToDo: Port to dataverse
            url = 'https://raw.githubusercontent.com/chirag126/data/main/'
            file_download = url + filename
            urlretrieve(file_download, path + filename)

        if not os.path.isfile(path + filename):
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self.dataset = pd.read_csv(path + filename)
        self.target = label

        # Save target and predictors
        self.X = self.dataset.drop(self.target, axis=1)
        
        # Save feature names
        self.feature_names = self.X.columns.to_list()
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


def return_loaders(data_name, download=False, batch_size=32, scaler='minmax'):
    """
    Load training and test datasets as DataLoader objects
    :param data_name: string with name of dataset
    :param download: boolean, whether to download the dataset
    :param batch_size: int, batch size
    :param scaler: string; 'minmax', 'standard', or 'none'
    :return: tuple with training and test dataloaders
    """
    labels = {'adult': 'income', 'compas': 'risk', 'gaussian': 'target', 'german': 'credit-risk',
              'gmsc': 'SeriousDlqin2yrs', 'heart': 'TenYearCHD', 'heloc': 'RiskPerformance', 'pima': 'Outcome'}
    
    prefix = f'./data/{data_name}/'
    file_train, file_test = data_name + '-train.csv', data_name + '-test.csv'

    dataset_train = TabularDataLoader(path=prefix, filename=file_train, label=labels[data_name],
                                      scale=scaler, download=download)

    dataset_test = TabularDataLoader(path=prefix, filename=file_test, label=labels[data_name],
                                     scale=scaler, download=download)

    trainloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    
    return trainloader, testloader
