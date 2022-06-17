# models
from openxai.LoadModel import LoadModel

# Experiment
from openxai.experiment_utils import ExperimentRunner

# (Train & Test) Loaders
import openxai.dataloader as loaders

# Perturbation Methods
from openxai.explainers.catalog.perturbation_methods import NormalPerturbation
from openxai.explainers.catalog.perturbation_methods import NewDiscrete_NormalPerturbation

import pickle
import warnings
warnings.filterwarnings("ignore")

# torch utils
import torch

# experiment parameter defaults
from openxai.experiment_config import *
from tqdm import tqdm
from functools import partialmethod

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


def main():
    data_names = ['german']
    model_names = ['lr', 'ann']
    
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
                feature_types = ['c'] * 6 + ['d'] * 7
            # Gaussian feature types
            elif data_name == 'synthetic':
                feature_types = ['c'] * 20
            # Heloc feature types
            elif data_name == 'heloc':
                feature_types = ['c'] * 23
            elif data_name == 'german':
                feature_types = pickle.load(open('./data/German_Credit_Data/german-feature-metadata.p', 'rb'))
            else:
                raise ValueError("Additional data sets will be included soon.")
            
            '''
            GET DATA LOADERS
            '''
            
            if data_name == 'synthetic':
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
    
                loader_train, loader_test = loaders.return_loaders(data_name=data_name,
                                                                   download=True,
                                                                   batch_size=data_loader_batch_size,
                                                                   gauss_params=gauss_params)
                data_iter = iter(loader_test)
                inputs, labels, weights, masks, masked_weights, probs, cluster_idx = data_iter.next()
            else:
                loader_train, loader_test = loaders.return_loaders(data_name=data_name,
                                                                   download=True,
                                                                   batch_size=data_loader_batch_size)
                data_iter = iter(loader_test)
                inputs, labels = data_iter.next()
                
            data_all = torch.FloatTensor(loader_train.dataset.data)
            
            '''
            LOAD ML MODELS
            '''
            
            if model_name == 'ann':
                model = LoadModel(data_name=data_name,
                                  ml_model=model_name,
                                  pretrained=True)
                L_map = model.L_relu
                
            elif model_name == 'lr':
                model = LoadModel(data_name=data_name,
                                  ml_model=model_name,
                                  pretrained=True)
                L_map = model.linear
            else:
                raise ValueError("Additional ML models will be supported soon.")

            """
            LOOP OVER EXPERIMENT SETUPS
            """
            
            if data_name == 'german':
                # use special perturbation class
                perturbation = NewDiscrete_NormalPerturbation("tabular",
                                                              mean=perturbation_mean,
                                                              std_dev=perturbation_std,
                                                              flip_percentage=perturbation_flip_percentage)

                # load feature metadata for perturbation class
                feature_types = pickle.load(open('./data/German_Credit_Data/german-feature-metadata.p', 'rb'))

            else:
                perturbation = NormalPerturbation("tabular",
                                                  mean=perturbation_mean,
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

