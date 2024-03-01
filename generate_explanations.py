# Utils
import os
import time
import torch
import numpy as np
import warnings; warnings.filterwarnings("ignore")
from openxai.experiment_utils import fill_param_dict

# Models, Data, and Explainers
from openxai.model import LoadModel
from openxai.dataloader import return_loaders
from openxai.explainer import Explainer

# Default parameters for each explainer's __init__ method
default_param_dicts = {
    'control': {
        'seed': 0
    },
    'grad': {
        'absolute_value': False
    },
    'ig': {
        'method': 'gausslegendre', 
        'multiply_by_inputs': False
    },
    'itg': {},
    'sg': {
        'n_samples': 100,
        'standard_deviation': 0.1,
        'seed': 0
    },
    'shap': {
        'n_samples': 500,
        'model_impl': 'torch',
        'seed': 0,
    },
    'lime': {
        'n_samples': 1000,
        'kernel_width': 0.75,
        'std': 0.1,
        'mode': 'tabular',
        'sample_around_instance': True,
        'discretize_continuous': False,
        'seed': 0,
    }
}

# Compute saved file names
param_strs = {method: '_'.join([f'{k}_{v}' for k, v in default_param_dicts[method].items()])\
              for method in default_param_dicts}

def GenerateExplanations(methods, data_name, model_name, n_test_samples):
    # Get data
    trainloader, testloader = return_loaders(data_name=data_name, download=False)
    dataset_tensor = torch.FloatTensor(trainloader.dataset.data)
    inputs = torch.FloatTensor(testloader.dataset.data[:n_test_samples])

    # Load model and make predictions
    model = LoadModel(data_name, model_name, pretrained=True)
    preds = model(inputs).argmax(dim=-1)
    
    # Compute explanations for each method
    for method in methods:
        print(f'Computing explanations for {method} (elapsed time: {time.time() - start_time:.2f}s)')
        param_dict = fill_param_dict(method, default_param_dicts[method], dataset_tensor)
        explainer = Explainer(method, model, param_dict)
        explanations = explainer.get_explanation(inputs, preds).detach().numpy()
        param_str = '_' + param_strs[method] if param_strs[method] else ''
        filename = f'explanations/{data_name}_{model_name}_{method}_{n_test_samples}{param_str}.npy'
        np.save(filename.format(filename), explanations)
        del explanations  # free up memory

if __name__ == '__main__':
    # Parameters
    methods = ['control', 'grad', 'ig', 'itg', 'sg', 'shap', 'lime']
    data_names = ['adult', 'compas', 'gaussian', 'german', 'gmsc', 'heart', 'heloc', 'pima']
    model_names = ['lr', 'ann']
    n_test_samples = 1000

    # Make directory for explanations
    if not os.path.exists('explanations'):
        os.makedirs('explanations')

    # Generate explanations
    start_time = time.time()
    for data_name in data_names:
        for model_name in model_names:
            print(f"Data: {data_name}, Model: {model_name}")
            GenerateExplanations(methods, data_name, model_name, n_test_samples)