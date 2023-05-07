import csv
import io
import os
import re
import subprocess
import torch
import requests
import pandas as pd
from openxai import dgp_synthetic
from errno import EEXIST
from typing import Any, List
import torch.utils.data as data
from torchvision import transforms
from torch.utils.data import DataLoader
from urllib.request import urlopen, urlretrieve
# from xai_benchmark.dataset.Synthetic import dgp_synthetic
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
    def __init__(self, path, filename, label, download=False, scale='minmax', gauss_params=None, file_url=None):
            
        """
        Load training dataset
        :param path: string with path to training set
        :param label: string, column name for label
        :param scale: string; 'minmax', 'standard', or 'none'
        :param dict: standard params of gaussian dgp
        :return: tensor with training data
        """

        self.path = path

        # Load Synthetic dataset
        if 'Synthetic' in self.path:
            
            '''
            if download:
                url = 'https://raw.githubusercontent.com/chirag126/data/main/'
                self.mkdir_p(path)
                file_download = url + 'dgp_synthetic.py'
                # import ipdb; ipdb.set_trace()
                urlretrieve(file_download, path + 'dgp_synthetic.py')

            if not os.path.isdir(path + 'dgp_synthetic.py'):
                raise RuntimeError("Dataset not found. You can use download=True to download it")           

            from openxai import dgp_synthetic
            
            '''

            if gauss_params is None:
                gauss_params = {
                    'n_samples': 2500,
                    'dim': 20,
                    'n_clusters': 10,
                    'distance_to_center': 5,
                    'test_size': 0.25,
                    'upper_weight': 1,
                    'lower_weight': -1,
                    'seed': 564,
                    'sigma': None,
                    'sparsity': 0.25
                }
            
            data_dict, data_dict_train, data_dict_test = dgp_synthetic.generate_gaussians(gauss_params['n_samples'],
                                                        gauss_params['dim'],
                                                        gauss_params['n_clusters'],
                                                        gauss_params['distance_to_center'],
                                                        gauss_params['test_size'],
                                                        gauss_params['upper_weight'],
                                                        gauss_params['lower_weight'],
                                                        gauss_params['seed'],
                                                        gauss_params['sigma'],
                                                        gauss_params['sparsity']).dgp_vars()
            
            self.ground_truth_dict = data_dict
            self.target = label
            
            if 'train' in filename:
                data_dict = data_dict_train
            elif 'test' in filename:
                data_dict = data_dict_test
            else:
                raise NotImplementedError('The current version of DataLoader class only provides training and testing splits')
                   
            self.dataset = pd.DataFrame(data_dict['data'])
            data_y = pd.DataFrame(data_dict['target'])
            
            names = []
            for i in range(gauss_params['dim']):
                name = 'x' + str(i)
                names.append(name)
                
            self.dataset.columns = names
            self.dataset['y'] = data_y
            
            # add additional Gaussian related aspects
            self.probs = data_dict['probs']
            self.masks = data_dict['masks']
            self.weights = data_dict['weights']
            self.masked_weights = data_dict['masked_weights']
            self.cluster_idx = data_dict['cluster_idx']
            
        else:
            if download:
                self.mkdir_p(path)
                if file_url is None:
                    url = 'https://raw.githubusercontent.com/chirag126/data/main/'
                    file_download = url + filename
                    urlretrieve(file_download, path + filename)
                else:
                    download_file(file_url, path + filename)

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

        # select correct row with idx
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        
        if 'Synthetic' in self.path:
            return (self.data[idx], self.targets.values[idx], self.weights[idx], self.masks[idx],
                    self.masked_weights[idx], self.probs[idx], self.cluster_idx[idx])
        else:
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


def return_loaders(data_name, download=False, batch_size=32, transform=None, scaler='minmax', gauss_params=None):
                
    # Create a dictionary with all available dataset names
    dict = {
            'adult': ('Adult', transform, 'income'),
            'compas': ('COMPAS', transform, 'risk'),
            'german': ('German_Credit_Data', transform, 'credit-risk'),
            'heloc': ('Heloc', transform, 'RiskPerformance'),
            'credit': ('Credit', transform, 'SeriousDlqin2yrs'),
            'synthetic': ('Synthetic', transform, 'y'),
            'rcdv': ('rcdv1980', transform, 'recid'),
            'lending-club': ('lending-club', transform, 'loan_repaid'),
            'student': ('student', transform, 'decision'),
            }

    urls = {
            'rcdv-train': 'https://dataverse.harvard.edu/api/access/datafile/7093737',
            'rcdv-test': 'https://dataverse.harvard.edu/api/access/datafile/7093739',
            'lending-club-train': 'https://dataverse.harvard.edu/api/access/datafile/6767839',
            'lending-club-test': 'https://dataverse.harvard.edu/api/access/datafile/6767838',
            'student-train': 'https://dataverse.harvard.edu/api/access/datafile/7093733',
            'student-test': 'https://dataverse.harvard.edu/api/access/datafile/7093734',
            }
    
    if dict[data_name][0] == 'synthetic':
        prefix = './data/' + dict[data_name][0] + '/'
        file_train = 'train'
        file_test = 'test'
    else:
        prefix = './data/' + dict[data_name][0] + '/'
        file_train = data_name + '-train.csv'
        file_test = data_name + '-test.csv'

    dataset_train = TabularDataLoader(path=prefix, filename=file_train,
                                      label=dict[data_name][2], scale=scaler,
                                      gauss_params=gauss_params, download=download,
                                      file_url=urls.get(file_train[:-4], None))

    dataset_test = TabularDataLoader(path=prefix, filename=file_test,
                                     label=dict[data_name][2], scale=scaler,
                                     gauss_params=gauss_params, download=download,
                                     file_url=urls.get(file_test[:-4], None))

    trainloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    
    return trainloader, testloader
