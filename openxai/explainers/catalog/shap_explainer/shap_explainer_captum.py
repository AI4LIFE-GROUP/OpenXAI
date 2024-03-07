import torch
import numpy as np
from ...api import BaseExplainer
from captum.attr import KernelShap


class SHAPExplainerC(BaseExplainer):
    '''
    param: model: model object
    param: data: pandas data frame or numpy array
    param: link: str, 'identity' or 'logit'
    param: feature_perturbation: str, 'tree_path_dependent' or 'interventional'
    '''

    def __init__(self, model, n_samples=500, model_impl: str = 'torch', seed=None) -> None:
        super().__init__(model)
        self.n_samples = n_samples
        if model_impl == 'torch':
            self.explainer = KernelShap(model)
        elif model_impl == 'sklearn':
            self.explainer = KernelShap(self.forward_func_sklearn)
        self.seed = seed

    def forward_func_sklearn(self, input):
        return torch.tensor(self.model.predict_proba(input))

    def forward_func_torch(self, input):
        return self.model(input)

    def get_explanations(self, x: torch.FloatTensor, label=None) -> torch.FloatTensor:
        '''
        Returns SHAP values as the explaination of the decision made for the input data (x)
        :param x: data samples to explain decision for
        :return: SHAP values [dim (shap_vals) == dim (x)]
        '''
        label = self.model(x.float()).argmax(dim=-1) if label is None else label
        if self.seed is not None:
            torch.manual_seed(self.seed); np.random.seed(self.seed)
        shap_vals = self.explainer.attribute(x.float(), target=label, n_samples=self.n_samples)
        return torch.FloatTensor(shap_vals)



