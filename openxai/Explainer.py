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


def Explainer(method: str,
              model,
              dataset_tensor: torch.tensor,
              param_dict_grad=None,
              param_dict_sg=None,
              param_dict_ig=None,
              param_dict_lime=None,
              param_dict_shap=None):
    
    if method == 'grad':
        if param_dict_grad is None:
            param_dict_grad = dict()
            param_dict_grad['absolute_value'] = True
        explainer = Gradient(model,
                             absolute_value=param_dict_grad['absolute_value'])
    
    elif method == 'sg':
        if param_dict_sg is None:
            param_dict_sg = dict()
            param_dict_sg['n_samples'] = 100
            param_dict_sg['standard_deviation'] = 0.005
        explainer = SmoothGrad(model,
                               num_samples=param_dict_sg['n_samples'],
                               standard_deviation=param_dict_sg['standard_deviation'])
    
    elif method == 'itg':
        explainer = InputTimesGradient(model)
    
    elif method == 'ig':
        if param_dict_ig is None:
            param_dict_ig = dict()
            param_dict_ig['method'] = 'gausslegendre'
            param_dict_ig['multiply_by_inputs'] = False
            param_dict_ig['baseline'] = torch.mean(dataset_tensor, dim=0).reshape(1, -1).float()
        explainer = IntegratedGradients(model,
                                        method=param_dict_ig['method'],
                                        multiply_by_inputs=param_dict_ig['multiply_by_inputs'],
                                        baseline=param_dict_ig['baseline'])
    
    elif method == 'shap':
        if param_dict_shap is None:
            param_dict_shap = dict()
            param_dict_shap['subset_size'] = 500
        explainer = SHAPExplainerC(model,
                                   model_impl='torch',
                                   n_samples=param_dict_shap['subset_size'])

    elif method == 'lime':
        if param_dict_lime is None:
            param_dict_lime = dict()
            param_dict_lime['dataset_tensor'] = dataset_tensor
            param_dict_lime['kernel_width'] = 0.75
            param_dict_lime['std'] = float(np.sqrt(0.05))
            param_dict_lime['mode'] = 'tabular'
            param_dict_lime['sample_around_instance'] = True
            param_dict_lime['n_samples'] = 1000
            param_dict_lime['discretize_continuous'] = False

        explainer = LIME(model.predict,
                         param_dict_lime['dataset_tensor'],
                         std=param_dict_lime['std'],
                         mode=param_dict_lime['mode'],
                         sample_around_instance=param_dict_lime['sample_around_instance'],
                         kernel_width=param_dict_lime['kernel_width'],
                         n_samples=param_dict_lime['n_samples'],
                         discretize_continuous=param_dict_lime['discretize_continuous'])

    elif method == 'control':
        explainer = RandomBaseline(model)
    
    else:
        raise NotImplementedError("This method has not been implemented, yet.")
    
    return explainer
