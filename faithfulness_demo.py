# Utils
import torch
import numpy as np

# Models, Data, Explainers, and Evaluators
from openxai.model import LoadModel
from openxai.dataloader import ReturnLoaders
from openxai.explainer import Explainer
from openxai.evaluator import Evaluator, stability_metrics
from openxai.explainers.perturbation_methods import get_perturb_method

# Choose the model and the data set you wish to generate explanations for
data_name = 'heloc'  # one of ['adult', 'compas', 'gaussian', 'german', 'gmsc', 'heart', 'heloc', 'pima']
model_name = 'lr'  # one of ['lr', 'ann']
n_test = 10

"""### (1) Data Loaders"""
trainloader, testloader = ReturnLoaders(data_name=data_name, download=True, batch_size=n_test)
inputs, labels = next(iter(testloader))

"""### (2) Load a pretrained ML model"""
model = LoadModel(data_name=data_name, ml_model=model_name, pretrained=True)

"""### (3) Choose an explanation method"""
method = 'grad'  # one of ['grad', 'sg', 'itg', 'ig', 'shap', 'lime', 'control']
explainer = Explainer(method=method, model=model,
                      dataset_tensor=torch.FloatTensor(trainloader.dataset.data),
                      param_dict=None) # None uses default hyperparams
explanations = explainer.get_explanations(inputs.float(), label=labels.type(torch.int64))

"""### (4) Choose an evaluation metric (see evaluate_metrics._construct_kwargs for more details)"""
metric = 'PGI'  # or PGU
kwargs = {
    'explanations': explanations,  # update kwargs per explanation method
    'inputs': inputs,
    'k': 3,
    'perturb_method': get_perturb_method(std=0.1, data_name=data_name),
    'feature_metadata': trainloader.dataset.feature_metadata,
    'num_samples': 100,
    'seed': -1,
    'n_jobs': None  # Number of parallel jobs, -1 to use all available cores, None to disable parallelism
}

"""### (5) Evaluate the explanation method"""
evaluator = Evaluator(model, metric)
score, mean_score = evaluator.evaluate(**kwargs)
std_err = np.std(score) / np.sqrt(len(score))
print(f"{metric}: {mean_score:.2f}\u00B1{std_err:.2f}")
if metric in stability_metrics:
    print(f"log({metric}): {np.log(mean_score):.2f}\u00B1{np.log(std_err):.2f}")