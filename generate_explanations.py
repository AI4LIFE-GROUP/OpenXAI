# Utils
import os
import torch
import numpy as np

# ML models
from openxai.model import LoadModel

# Data loaders
from openxai.dataloader import return_loaders

# Explanation models
from openxai.explainer import Explainer

def GenerateExplanations(methods, data_name, model_name, n_test_samples=100):
    # Get data
    loader_train, loader_test = return_loaders(data_name=data_name, download=False)
    dataset_tensor = torch.FloatTensor(loader_train.dataset.data)
    inputs = torch.FloatTensor(loader_test.dataset.data[:n_test_samples])

    # Load model and make predictions
    model = LoadModel(data_name, model_name, pretrained=True)
    preds = model(inputs).argmax(dim=-1)
    
    # Compute explanations for each method
    explanations = {method: None for method in methods}
    for method in methods:
        explainer = Explainer(method, model, dataset_tensor, param_dict=None)  # use default hyperparameters in openxai.explainer.py
        explanations[method] = explainer.get_explanation(inputs, preds)
    return explanations

if __name__ == '__main__':
    if not os.path.exists('explanations'):
        os.makedirs('explanations')
    methods = ['control', 'grad', 'ig', 'itg', 'sg', 'shap', 'lime']
    data_names = ['adult', 'compas', 'gaussian', 'german', 'gmsc', 'heart', 'heloc', 'pima']
    model_names = ['lr', 'ann']
    n_test_samples = 1000
    for data_name in data_names:
        for model_name in model_names:
            print(f"Data: {data_name}, Model: {model_name}")
            explanations = GenerateExplanations(methods, data_name, model_name, n_test_samples)
            for method, explanation in explanations.items():
                filename = f'explanations/{data_name}_{model_name}_{method}_{n_test_samples}.npy'
                np.save(filename.format(filename), explanation.detach().numpy())