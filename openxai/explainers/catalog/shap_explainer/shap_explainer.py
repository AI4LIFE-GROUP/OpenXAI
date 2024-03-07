import xgboost
import shap
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from ...api import BaseExplainer
from shap import KernelExplainer
from shap import DeepExplainer

class SHAPExplainer(BaseExplainer):
    
    '''
    param: model: model object
    param: data: pandas data frame or numpy array
    param: link: str, 'identity' or 'logit'
    param: feature_perturbation: str, 'tree_path_dependent' or 'interventional'
    '''
    
    def __init__(self, model, data: torch.FloatTensor, domain: str = 'non_deep', link: str = 'identity',
                 function_class: str = 'non_tree', feature_perturbation: str = 'interventional'):
        super().__init__(model)
        self.data = data.numpy()
        self.domain = domain
        if self.domain == 'non_deep':
            if function_class == 'non_tree':
                self.explainer = shap.KernelExplainer(self.model, self.data, link=link)
            else:
                self.explainer = shap.TreeExplainer(self.model, self.data, model_output='raw',
                                           feature_perturbation=feature_perturbation)
        elif self.domain == 'deep':
            self.explainer = shap.DeepExplainer(self.model[0], self.data)

    def get_explanations(self, x: torch.FloatTensor, label=None):
        '''
        Returns SHAP values as the explaination of the decision made for the input data (data_x)
        :param x: data samples to explain decision for
        :return: SHAP values [dim (shap_vals) == dim (x)]
        '''
        
        x = x.numpy()
        
        if self.domain == 'non_deep':
            # we are explaining the the prob of 1; choosing [0] would explain P(y=0|x)
            shap_vals = self.explainer.shap_values(x)[1]
        elif self.domain == 'deep':
            shap_vals = self.explainer.shap_values(x)
        return torch.FloatTensor(shap_vals)



