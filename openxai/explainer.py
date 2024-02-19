# Utils
import torch
import numpy as np

# Explanation Models
from openxai.explainers import Gradient
from openxai.explainers import IntegratedGradients
from openxai.explainers import InputTimesGradient
from openxai.explainers import SmoothGrad
from openxai.explainers import LIME
from openxai.explainers import SHAPExplainerC
from openxai.explainers import RandomBaseline

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
        'absolute_value': True
    },
    'sg': {
        'num_samples': 100,
        'standard_deviation': 0.005
    },
    'itg': {},
    'ig': {
        'method': 'gausslegendre',
        'multiply_by_inputs': False
    },
    'shap': {
        'model_impl': 'torch',
        'n_samples': 500
    },
    'lime': {
        'kernel_width': 0.75,
        'std': float(np.sqrt(0.03)),
        'mode': 'tabular',
        'sample_around_instance': True,
        'n_samples': 1000,
        'discretize_continuous': False
    },
    'control': {}
}

def fill_param_dict(method: str, param_dict: dict, dataset_tensor: torch.tensor):
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

def Explainer(method: str,
              model,
              dataset_tensor: torch.tensor,
              param_dict=None):
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
