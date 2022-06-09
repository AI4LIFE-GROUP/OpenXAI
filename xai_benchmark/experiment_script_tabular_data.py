# models
import xai_benchmark.ML_Models.ANN.model as model_ann
from xai_benchmark.ML_Models.SVM.model import SVM, SVM_Loss
from xai_benchmark.ML_Models.LR.model import LogisticRegression

# Experiment
from xai_benchmark.experiment_utils import ExperimentRunner

# (Train & Test) Loaders
import xai_benchmark.dataloader as loaders

# Perturbation Methods
# import torch.distributions as tdist
# from xai_benchmark.explainers.perturbation_methods import RandomPerturbation
# from xai_benchmark.explainers.perturbation_methods import UniformPerturbation
# from xai_benchmark.explainers.perturbation_methods import BootstrapPerturbation
# from xai_benchmark.explainers.perturbation_methods import MarginalPerturbation
# from xai_benchmark.explainers.perturbation_methods import AdversarialPerturbation
from xai_benchmark.explainers.catalog.perturbation_methods import NormalPerturbation
from xai_benchmark.explainers.catalog.perturbation_methods import NewDiscrete_NormalPerturbation

import pickle
import warnings
warnings.filterwarnings("ignore")

# torch utils
import torch

# experiment parameter defaults
from xai_benchmark.experiment_config import *
from tqdm import tqdm
from functools import partialmethod

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


def main():
    data_names = ['compas', 'adult']
    model_names = ['lr', 'ann']  #'ann
    
    """
    LOOP OVER ALL TABULAR DATA SETS
    """
    
    for model_name in model_names:
        
        print('Classifier:', model_name)
        
        for data_name in data_names:
        
            print(' Dataset:', data_name)
        
            # COMPAS feature types
            if data_name == 'compas':
                feature_types = ['c', 'd', 'c', 'c', 'd', 'd', 'd']
            
            # Adult feature types
            elif data_name == 'adult':
                feature_types = ['c', 'c', 'c', 'c', 'c', 'c', 'd', 'd', 'd', 'd', 'd', 'd', 'd']
            
            # Gaussian feature types
            elif data_name == 'gaussian':
                feature_types = ['c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c',
                                 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c']
            
            # this has to be fixed for the moment as the ANN model is trained on this data
            gauss_params = {
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
            
            '''
            GET DATA LOADERS
            '''
            
            if data_name == 'gaussian':
                loader_train, loader_test = loaders.return_loaders(data_name=data_name, download=False,
                                                                   batch_size=data_loader_batch_size,
                                                                   gauss_params=gauss_params)
                data_iter = iter(loader_test)
                inputs, labels, weights, masks, masked_weights, probs, cluster_idx = data_iter.next()
            else:
                loader_train, loader_test = loaders.return_loaders(data_name=data_name, download=True,
                                                                   batch_size=data_loader_batch_size)
                data_iter = iter(loader_test)
                inputs, labels = data_iter.next()
            
            top_k = int(percentage_most_important * inputs.shape[1])
            
            '''
            INFER IG BASLINE
            '''
            
            data_all = torch.FloatTensor(loader_train.dataset.data)
            ig_baseline = torch.mean(data_all, axis=0).reshape(1, -1)
            
            '''
            LOAD ML MODELS
            '''
            
            if data_name == 'gaussian':
                if model_name == 'ann':
                    model_path = './ML_Models/Saved_Models/ANN/gaussian_lr_0.002_acc_0.91.pt'
                    ann = model_ann.ANN_softmax(input_layer=inputs.shape[1], hidden_layer_1=100, num_of_classes=2)
                    ann.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                    
                    model = ann
                    L_map = ann.L_relu
                    
                elif model_name == 'lr':
                    model_path = './ML_Models/Saved_Models/LR/gaussian_lr_0.002_acc_0.73.pt'
                    lr = LogisticRegression(input_dim=inputs.shape[1])
                    lr.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                    
                    model = lr
                    L_map = lr.linear
                    
                elif model_name == 'svm':
                    model_path = './xai_benchmark/ML_Models/Saved_Models/SVM/gaussian_lr_0.002_acc_0.72.pt'
                    svm = SVM(inputs.shape[1], num_of_classes=1)
                    svm.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            
            elif data_name == 'adult':
                if model_name == 'ann':
                    model_path = './xai_benchmark/Saved_Models/ANN/adult_lr_0.002_acc_0.83.pt'
                    ann = model_ann.ANN_softmax(input_layer=inputs.shape[1], hidden_layer_1=100, num_of_classes=2)
                    ann.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
                    model = ann
                    L_map = ann.L_relu
    
                elif model_name == 'lr':
                    model_path = '/xai_benchmark/ML_Models/Saved_Models/LR/adult_lr_0.002_acc_0.84.pt'
                    lr = LogisticRegression(input_dim=inputs.shape[1])
                    lr.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        
                    model = lr
                    L_map = lr.linear
                    
            elif data_name == 'compas':
                if model_name == 'ann':
                    model_path = './xai_benchmark/ML_Models/Saved_Models/ANN/compas_lr_0.002_acc_0.85.pt'
                    ann = model_ann.ANN_softmax(input_layer=inputs.shape[1], hidden_layer_1=100, num_of_classes=2)
                    ann.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                    
                    model = ann
                    L_map = ann.L_relu
    
                elif model_name == 'lr':
                    model_path = './xai_benchmark/ML_Models/Saved_Models/LR/compas_lr_0.002_acc_0.85.pt'
                    lr = LogisticRegression(input_dim=inputs.shape[1])
                    lr.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        
                    model = lr
                    L_map = lr.linear
            
            elif data_name == 'german':
                if model_name == 'ann':
                    model_path = './ML_Models/Saved_Models/ANN/german_lr_0.002_acc_0.71.pt'
                    ann = model_ann.ANN_softmax(input_layer=inputs.shape[1], hidden_layer_1=100, num_of_classes=2)
                    ann.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                    
                    model = ann
                    L_map = ann.L_relu
                
                elif model_name == 'svm':
                    dataset_train = loaders.DataLoader_Tabular(
                        path='./Data_Sets/German_Credit_Data/',
                        filename='german-train.csv', label='credit-risk')
                    
                    model_path = './ML_Models/Saved_Models/SVM/german_svm_0.002_acc_0.73.pt'
                    svm = SVM(dataset_train.get_number_of_features(), num_of_classes=2)
                    svm.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                    
                    model = svm
                    L_map = svm.input1
                    
                elif model_name == 'lr':
                    model_path = './ML_Models/Saved_Models/LR/german_lr_0.002_acc_0.72.pt'
                    lr = LogisticRegression(input_dim=inputs.shape[1])
                    lr.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                    model = lr
                    L_map = lr.linear
            
            """
            LOOP OVER EXPERIMENT SETUPS
            """
            
            if data_name == 'german':
                # use special perturbation class
                perturbation = NewDiscrete_NormalPerturbation("tabular", mean=perturbation_mean,
                                                              std_dev=perturbation_std,
                                                              flip_percentage=perturbation_flip_percentage)

                # load feature metadata for perturbation class
                feature_types = pickle.load(open('./Data_Sets/German_Credit_Data/german-feature-metadata.p', 'rb'))

            else:
                perturbation = NormalPerturbation("tabular", mean=perturbation_mean,
                                                  std_dev=perturbation_std,
                                                  flip_percentage=perturbation_flip_percentage)

            er = ExperimentRunner(model=model,
                                  Lmap=L_map,
                                  dataset_tensor=data_all,
                                  perturbation=perturbation,
                                  experiment_name_str=f'{data_name}_{model_name}_bernp={perturbation_flip_percentage}_std={perturbation_std}',
                                  feature_metadata=feature_types,
                                  random_seed=612)

            er.get_explanations(inputs, labels)

            # Relative Stability
            stability_versions = ['diff_input', 'diff_output', 'diff_representation']
            
            for stability_version in stability_versions:
                er.run_stability_experiments(version_str=stability_version)


if __name__ =="__main__":
    main()

