# Utils
import torch
import numpy as np

# Explanation Models
from openxai.explainers import Gradient, IntegratedGradients,\
    InputTimesGradient, SmoothGrad, LIME, SHAPExplainerC, RandomBaseline

explainers_dict = {
    'grad': Gradient,
    'sg': SmoothGrad,
    'itg': InputTimesGradient,
    'ig': IntegratedGradients,
    'shap': SHAPExplainerC,
    'lime': LIME,
    'control': RandomBaseline
}

default_param_dicts = {
    'grad': {
        'absolute_value': False
    },
    'sg': {
        'n_samples': 500,
        'standard_deviation': 0.1,
        'seed': 0
    },
    'ig': {
        'method': 'gausslegendre', 
        'multiply_by_inputs': False
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
    },
    'itg': {},
    'control': {}
}

def fill_param_dict(method, param_dict, dataset_tensor):
    """
    Fills in the missing parameters for the given method with the default parameters
    :param method: str, name of the method
    :param param_dict: dict, parameter dictionary
    :return: dict, filled parameter dictionary
    """
    # None case
    param_dict = {} if param_dict is None else param_dict
    
    # Parameters requiring variables from IG and LIME 
    if (method == 'ig') and ('baseline' not in param_dict):
        param_dict['baseline'] = torch.mean(dataset_tensor, dim=0).reshape(1, -1).float()
    elif (method == 'lime') and ('dataset_tensor' not in param_dict):
        param_dict['data'] = dataset_tensor

    # Fill in missing parameters
    default_param_dict = default_param_dicts[method]
    for key in default_param_dict.keys():
        if key not in param_dict:
            param_dict[key] = default_param_dict[key]
    return param_dict

def Explainer(method, model, dataset_tensor, param_dict=None):
    """
    Returns an explainer object for the given method
    :param method: str, name of the method
    :param model: PyTorch model or function
    :param dataset_tensor: torch.tensor, dataset tensor
    :param param_dict: dict, parameter dictionary
    :return: explainer object
    """
    # Verify method
    if method not in explainers_dict.keys():
        raise NotImplementedError("This method has not been implemented, yet.")

    # Configure parameters
    model = model.predict if method == 'lime' else model
    param_dict = fill_param_dict(method, param_dict, dataset_tensor)

    # Return explainer
    explainer = explainers_dict[method](model, **param_dict)
    return explainer
