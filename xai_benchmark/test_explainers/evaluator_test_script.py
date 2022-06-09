import pandas as pd

# ANN models
import torch
import ML_Models.ANN.model as model_ann

# (Train & Test) Loaders
import ML_Models.data_loader as loaders

# Explanation Models
from explainers.lime import LIME
from explainers.shap_explainer import SHAPExplainer
from explainers.grad import Gradient
from explainers.ebp import EBP
from explainers.integrated_gradients import IntegratedGradients
from explainers.guided_backprop import GuidedBackprop
from explainers.smoothgrad import SmoothGrad
from explainers.gradcam import GradCAM
from explainers.guided_gradcam import GuidedGradCAM
from explainers.lrp import LRP
from explainers.input_x_gradient import InputTimesGradient

# Perturbation Methods
import torch.distributions as tdist
from explainers.perturbation_methods import RandomPerturbation
from explainers.perturbation_methods import UniformPerturbation
from explainers.perturbation_methods import BootstrapPerturbation
from explainers.perturbation_methods import MarginalPerturbation
from explainers.perturbation_methods import AdversarialPerturbation

# Import Evaluation Methods
from evaluator import Evaluator

# lime utils
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
import os, json

# torch utils
import torch
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F


def main():

    '''
    Loading Data Loaders
    '''
    
    # this has to be fixed for the moment as the ANN model is trained on this data
    
    params = {
        'n_samples': 1000,
        'dim': 20,
        'n_clusters': 10,
        'distance_to_center': 6,
        'test_size': 0.25,
        'upper_weight': 1,
        'lower_weight': -1,
        'seed': 564,
        'sigma': None,
        'sparsity': 0.25
    }
    
    #loader_train_cifar, loader_test_mnist = loaders.return_loaders(data_name='cifar10', is_tabular=False, batch_size=1)
    loader_train_gauss, loader_test_gauss = loaders.return_loaders(data_name='gaussian', is_tabular=True,
                                                                   batch_size=20, gauss_params=params)
    
    
    '''
    Gaussian Data DGP
    '''
    gauss_train_input = loader_train_gauss.dataset.ground_truth_dict
    data_iter = iter(loader_train_gauss)
    inputs, labels, weights, masks, masked_weights, probs, cluster_idx = data_iter.next()
    gaussian_all = torch.FloatTensor(loader_train_gauss.dataset.data)
    
    
    '''
    Load ML Models
    '''
    model_path = 'ML_Models/Saved_Models/ANN/gaussian_lr_0.002_acc_0.91.pt'
    ann = model_ann.ANN_softmax(20, hidden_layer_1=100, num_of_classes=2)
    ann.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    # Testing whether the outputs work
    a = ann(inputs.float())
    b = ann.predict(inputs.float())
    c = ann.predict_proba(inputs.float())
    
    
    inp = inputs.detach().float()[0, :].reshape(1, -1)
    lab = labels.type(torch.int64)[0]
    
    # this is my sample of interest [1, 20], therefore we apply reshape(-1) when we use the perturbation methods below
    print('single input:', inp)
    
    # those will be my data samples for the bootstrapped method below
    data_samples = inputs.detach().float()
    
    
    '''
    Testing Perturbation Methods
    '''
    
    feature_mask = masks[0]
    inp = inputs.detach().float()[0, :].reshape(1, -1)
    data_samples = inputs.detach().float()
    
    print("Generating Random Perturbations....")
    perturbation_method = RandomPerturbation("tabular")
    num_samples = 10
    max_distance = 4
    perturbed_inp = perturbation_method.get_perturbed_inputs(inp.reshape(-1), feature_mask, num_samples, max_distance)
    print(perturbed_inp.shape)
    
    print("Generating Uniform Perturbations....")
    perturbation_method = UniformPerturbation("tabular")
    perturbed_inp = perturbation_method.get_perturbed_inputs(inp.reshape(-1), feature_mask, num_samples, max_distance)
    print(perturbed_inp.shape)
    
    print("Generating Marginal Perturbations....")
    perturbation_method = MarginalPerturbation("tabular",
                                               [tdist.Normal(0,5), tdist.Normal(1,6),
                                                tdist.Normal(0,5), tdist.Normal(1,6),
                                                tdist.Normal(0,5), tdist.Normal(1,6),
                                                tdist.Normal(0,5), tdist.Normal(1,6),
                                                tdist.Normal(0,5), tdist.Normal(1,6),
                                                tdist.Normal(0,5), tdist.Normal(1,6),
                                                tdist.Normal(0,5), tdist.Normal(1,6),
                                                tdist.Normal(0,5), tdist.Normal(1,6),
                                                tdist.Normal(0,5), tdist.Normal(1,6),
                                                tdist.Normal(0,5), tdist.Normal(1,6)])
    num_samples = 100
    max_distance = 30
    perturbed_inp_m = perturbation_method.get_perturbed_inputs(inp.reshape(-1), feature_mask, num_samples, max_distance)
    print(perturbed_inp_m.shape)
    
    print("Generating Bootstrap Perturbations....")
    perturbation_method = BootstrapPerturbation("tabular")
    num_samples = 3
    max_distance = 100
    perturbed_inp = perturbation_method.get_perturbed_inputs(inp.reshape(-1), feature_mask,
                                                             num_samples, max_distance, data_samples)
    print(perturbed_inp.shape)

    '''
    Testing Grad on Gaussian Data
    Note that this is Explaining f(x), i.e., the logit score
    All methods below explain the logit score
    '''
    
    exps_grad = []
    for i in range(20):
        grad = Gradient(ann)
        inp = inputs.detach().float()[0, :].reshape(1, -1)
        lab = labels.type(torch.int64)[0]
        exp_grad = grad.get_explanation(inp, lab)
        exps_grad.append(exp_grad)



    '''
    Testing Evaluator
    '''

    # perturbation = UniformPerturbation("tabular")
    perturbation = RandomPerturbation("tabular")

    def generate_mask(explanations, top_k):
        mask_indices = torch.topk(explanations[0], top_k).indices
        mask = torch.zeros(explanations[0].shape) > 10
        for i in mask_indices:
            mask[i] = True
        return mask

    'Evaluate Relative Stability'

    def dict_collector(index, Lmap, perturbation, labels, masks, inputs, ann,
                       explanation_method, explanations, max_distance=4):
        
        input_dict = {}

        input_dict['x'] = inputs[index].reshape(-1)
        input_dict['explainer'] = explanation_method
        input_dict['explanation_x'] = explanations[index]
        input_dict['input_data'] = inputs.float()
        input_dict['perturbation'] = perturbation
        input_dict['L_map'] = Lmap
        input_dict['p_norm'] = 2
        input_dict['top_k'] = 3
        input_dict['eval_metric'] = 'eval_relative_stability'
        input_dict['perturb_max_distance'] = max_distance
        input_dict['perturb_method'] = perturbation
        input_dict['y'] = labels[index].detach().item()
        input_dict['y_pred'] = torch.max(ann(inputs[index].unsqueeze(0).float()), 1).indices.detach().item()
        input_dict['mask'] = generate_mask(input_dict['explanation_x'], input_dict['top_k'])
        
        return input_dict

    stab_measure2 = []
    
    dist_measures = []
    for j in [1, 4, 7]:
        stab_measure1 = []
        for i in range(15):
            input_dict = dict_collector(i, ann, perturbation, labels, masks, inputs, ann,
                                        explanation_method=grad, explanations=exps_grad, max_distance=j)
            evaluator = Evaluator(input_dict)
            stability, stability_ratios, rep_diffs, x_diffs, exp_diffs = evaluator.eval_relative_stability(use_treshold=False)
            print(stability)
            stab_measure1.append(stability)
            dist_measures.append(stability)
    
    plt.scatter(np.r_[np.repeat(1, 15), np.repeat(4, 15), np.repeat(7, 15)], np.array(dist_measures))
    plt.xlabel('Max Distance')
    plt.title('Using Output Difference in Denominator')
    plt.ylabel('Stability')
    plt.tight_layout()
    plt.savefig('Stability_output_diff.png')
    plt.show()
    plt.close()

    dist_measures = []
    for j in [1, 4, 7]:
        stab_measure1 = []
        for i in range(15):
            input_dict = dict_collector(i, ann, perturbation, labels, masks, inputs, ann,
                                        explanation_method=grad, explanations=exps_grad, max_distance=j)
            evaluator = Evaluator(input_dict)
            stability, stability_ratios, rep_diffs, x_diffs, exp_diffs = evaluator.eval_relative_stability(
                use_treshold=False, rep_denominator_flag=True)
            print(stability)
            stab_measure1.append(stability)
            dist_measures.append(stability)

    plt.scatter(np.r_[np.repeat(1, 15), np.repeat(4, 15), np.repeat(7, 15)], np.array(dist_measures))
    plt.xlabel('Max Distance')
    plt.ylabel('Stability')
    plt.title('Using Output Feature Difference in Denominator')
    plt.tight_layout()
    plt.savefig('Stability_feature_diff.png')
    plt.show()
    plt.close()

    dist_measures = []
    for j in [1, 4, 7]:
        stab_measure1 = []
        for i in range(15):
            input_dict = dict_collector(i, ann.L_relu, perturbation, labels, masks, inputs, ann,
                                        explanation_method=grad, explanations=exps_grad, max_distance=j)
            evaluator = Evaluator(input_dict)
            stability, stability_ratios, rep_diffs, x_diffs, exp_diffs = evaluator.eval_relative_stability(
                use_treshold=False)
            print(stability)
            stab_measure1.append(stability)
            dist_measures.append(stability)

    plt.scatter(np.r_[np.repeat(1, 15), np.repeat(4, 15), np.repeat(7, 15)], np.array(dist_measures))
    plt.xlabel('Max Distance')
    plt.ylabel('Stability')
    plt.title('Using Intermediate Difference in Denominator')
    plt.tight_layout()
    plt.savefig('Stability_intermediate_diff.png')
    plt.show()
    plt.close()
    
    
    for i in range(15):
        input_dict = dict_collector(i, ann.L_relu, perturbation, labels, masks, inputs, ann,
                                    explanation_method=grad, explanations=exps_grad, max_distance=4)
        evaluator = Evaluator(input_dict)
        stability, stability_ratios, rep_diffs, x_diffs, exp_diffs = evaluator.eval_relative_stability(use_treshold=True)
        print(stability)
        stab_measure2.append(stability)
        
        
        
    '''
    'Evaluate Faithfulness'
    
    input_dict['input_data'] = inputs.float()
    input_dict['perturbation'] = perturbation
    input_dict['L_map'] = ann.L_relu
    input_dict['p_norm'] = 2
    input_dict['top_k'] = 1  # this argument doesnt do anything yet
    input_dict['eval_metric'] = 'eval_pred_faithfulness'
    input_dict['perturb_max_distance'] = 4
    input_dict['perturb_method'] = perturbation
    input_dict['y'] = labels[0].detach().item()
    input_dict['y_pred'] = torch.max(ann(inputs[0].unsqueeze(0).float()), 1).indices.detach().item()
    input_dict['mask'] = feature_mask
    input_dict['model'] = ann

    evaluator = Evaluator(input_dict)
    faith = evaluator.eval_pred_faithfulness()
    
    a = 2
    
    '''
    
if __name__ == "__main__":
    # execute training
    main()
    