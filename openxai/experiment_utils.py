import os
import torch
import numpy as np
import joblib
import contextlib
import json
from tqdm import tqdm
from functools import partialmethod

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

def convert_k_to_int(k, n_feat):
    """
    Return the range of k values to evaluate
    :param k: int, float, or str
        :setting k to -1 will evaluate all features
        :setting k to a float will evaluate the top k% of features (rounded up)
    :param n_feat: int, number of features
    :return: int
    """
    if k == -1:
        return n_feat
    if not isinstance(k, int):
        if isinstance(k, float):
            if 0 < k < 1:
                return np.ceil(k * n_feat).astype(int)
            else:
                raise ValueError(f'Float value for k {k} must be between 0 and 1')
        else:
            raise ValueError(f'Invalid type for k: {type(k)}')

def load_parameterized_file(prefix, params, extension='.npy'):
    """
    Load a file with parameters in the file name
    :param prefix: str, file path barring parameters and extension
    :param params: dict, parameters
    """
    param_str = construct_param_string(params)
    results = np.load(prefix + param_str + extension)
    return results

def make_directory(path):
    """
    Create a directory if it does not exist
    :param path: str, path to the directory
    """
    if not os.path.exists(path):
        os.makedirs(path)

def construct_param_string(params):
    """
    Construct a string from a dictionary of parameters
    :param params: dict
    :return: str
    """
    param_str = '_' + '_'.join([f'{k}_{v}' for k, v in params.items()]) if params else ''
    return param_str

def invalid_model_metric_combination(model_name, metric):
    """
    Check if the model-metric combination is invalid
    :param model_name: str, name of the model
    :param metric: str, name of the metric
    :return: bool
    """
    invalid_combinations = {
        'ann': ['PRA', 'RC', 'FA', 'RA', 'SA', 'SRA'],
        'lr': ['RRS']  # RRS == ROS for logistic regression
    }
    return metric in invalid_combinations[model_name]

def load_config(config_path):
    """
    Loads the configuration file
    :param config_path: str, path to the configuration file
    :return: dict, configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def fill_param_dict(method, param_dict, dataset_tensor):
    """
    Fills in the dataset specific parameters for ig and lime
    :param method: str, name of the method
    :param param_dict: dict, parameter dictionary
    :param dataset_tensor: torch.FloatTensor, dataset tensor
    :return: dict, filled parameter dictionary
    """
    # Parameters requiring variables from IG and LIME 
    if method == 'ig':
        param_dict['baseline'] = torch.mean(dataset_tensor, dim=0).reshape(1, -1).float()
    elif method == 'lime':
        param_dict['data'] = dataset_tensor
        
    return param_dict

def convert_to_numpy(x):
    """Converts input to numpy array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()  # in case of GPU
    elif not isinstance(x, np.ndarray):
        x = np.array(x)
    return x

def convert_to_tensor(x):
    """Converts input to torch tensor."""
    if not isinstance(x, torch.Tensor):
        x = torch.Tensor(x)
    return x

def compute_Lp_norm_diff(vec1, vec2, p_norm, normalize_to_relative_change = True):
    """
    Computes the L-p norm difference between two vectors
    :param vec1: np.array, torch.Tensor, list
    :param vec2: np.array, torch.Tensor, list
    :param p_norm: int, float
    :param normalize_to_relative_change: bool
    :return: float
    """
    vec1, vec2 = convert_to_numpy(vec1).flatten(), convert_to_numpy(vec2).flatten()
    diff = vec1 - vec2
    norm_diff = np.linalg.norm(diff, ord=p_norm)
    if normalize_to_relative_change:
        vec1_norm = np.linalg.norm(vec1, ord=p_norm)
        norm_diff = np.nan if vec1_norm == 0 else norm_diff/vec1_norm
    return norm_diff

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

def print_summary(model, trainloader, testloader):
    # Get data
    X_train, y_train = torch.FloatTensor(trainloader.dataset.data), trainloader.dataset.targets.to_numpy()
    X_test, y_test = torch.FloatTensor(testloader.dataset.data), testloader.dataset.targets.astype(int)

    # Get predictions
    model.eval()
    preds = (model(X_test)[:, 1] >= 0.5).to(int).detach().numpy()
    preds_tr = (model(X_train)[:, 1] >= 0.5).to(int).detach().numpy()
    test_acc, train_acc = (preds == y_test).mean(), (preds_tr == y_train).mean()

    # Print summary
    print(f'Proportion of Class 1:\n\tTest Preds:\t{preds.mean():.4f}\n\tTest Set:\t{y_test.mean():.4f}')
    print(f'Test Accuracy: {test_acc:.4f}')
    print(f'Train Accuracy: {train_acc:.4f}')

def generate_mask(explanation, top_k):
    if not isinstance(explanation, torch.Tensor):
        explanation = torch.Tensor(explanation)
    mask_indices = torch.topk(explanation.abs(), top_k).indices
    mask = torch.ones(explanation.shape, dtype=bool)
    for i in mask_indices:
        mask[i] = False
    return mask
