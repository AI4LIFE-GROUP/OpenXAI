import pandas as pd

# Explanation Models
from openxai.Explainer import Explainer

# Perturbation Methods
from openxai.explainers.perturbation_methods import *

# Import Evaluation Methods
from openxai.evaluator import Evaluator

# lime utils
import os, json

# torch utils
import torch

# experiment parameter defaults
from openxai.experiment_config import *

from tqdm import tqdm
from functools import partialmethod

from datetime import timedelta
import time
import pickle

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


def generate_mask(explanation, top_k):
    if not isinstance(explanation, torch.Tensor):
        explanation = torch.Tensor(explanation)
    mask_indices = torch.topk(explanation.abs(), top_k).indices
    mask = torch.ones(explanation.shape, dtype=bool)
    for i in mask_indices:
        mask[i] = False
    return mask


def dict_collector(index,
                   Lmap,
                   perturbation,
                   labels,
                   top_k,
                   inputs,
                   model,
                   explanation_method,
                   explanation,
                   feature_metadata,
                   eval_metric,
                   max_distance=0.4,
                   norm=2,
                   masks=None):

    input_dict = dict()
    input_dict['x'] = inputs[index].reshape(-1)
    input_dict['explainer'] = explanation_method
    input_dict['explanation_x'] = explanation
    input_dict['input_data'] = inputs
    input_dict['perturbation'] = perturbation
    input_dict['L_map'] = Lmap
    input_dict['p_norm'] = norm
    input_dict['top_k'] = top_k
    input_dict['eval_metric'] = eval_metric
    input_dict['perturb_max_distance'] = max_distance
    input_dict['perturb_method'] = perturbation
    input_dict['y'] = labels[index].detach().item()
    input_dict['y_pred'] = torch.max(model(inputs[index].unsqueeze(0).float()), 1).indices.detach().item()
    input_dict['mask'] = generate_mask(input_dict['explanation_x'].reshape(-1), input_dict['top_k'])
    input_dict['model'] = model
    input_dict['feature_metadata'] = feature_metadata

    if masks is not None:
        input_dict['gt_mask'] = masks[index].unsqueeze(0)

    return input_dict


class ExperimentRunner():
    def __init__(self,
                 model,
                 Lmap,
                 dataset_tensor: torch.Tensor,
                 perturbation: BasePerturbation,
                 experiment_name_str: str,
                 feature_metadata,
                 random_seed: int = 0,
                 ig_baseline = None):
        
        '''
        Class to evaluate explanation methods given a dataset and model.
        '''

        np.random.seed(seed=random_seed)

        self.model = model
        self.Lmap = Lmap

        self.experiment_name_str = experiment_name_str
        self.feature_metadata = feature_metadata

        # initialize explainers using parameters specified by config
        self.perturbation = perturbation
        
        # Vanilla Gradients
        param_dict_grad = dict()
        param_dict_grad['absolute_value'] = grad_absolute_value
        grad = Explainer(method='grad',
                         model=model,
                         dataset_tensor=dataset_tensor,
                         param_dict_grad=param_dict_grad)
        
        # Integrated Gradients
        if ig_baseline is None:
            ig_baseline = torch.mean(dataset_tensor, dim=0).reshape(1, -1).float()
        param_dict_ig = dict()
        param_dict_ig['method'] = ig_method
        param_dict_ig['multiply_by_inputs'] = ig_multiply_by_inputs
        param_dict_ig['baseline'] = ig_baseline
        ig = Explainer(method='ig',
                       model=model,
                       dataset_tensor=dataset_tensor,
                       param_dict_ig=param_dict_ig)
        
        # Input x Gradient
        itg = Explainer(method='itg',
                        model=model,
                        dataset_tensor=dataset_tensor)
        
        # Smoothgrad
        param_dict_sg = dict()
        param_dict_sg['n_samples'] = sg_n_samples
        param_dict_sg['standard_deviation'] = sg_standard_deviation_005
        sg005 = Explainer(method='sg',
                          model=model,
                          dataset_tensor=dataset_tensor,
                          param_dict_sg=param_dict_sg)
        param_dict_sg['standard_deviation'] = sg_standard_deviation_003
        sg003 = Explainer(method='sg',
                          model=model,
                          dataset_tensor=dataset_tensor,
                          param_dict_sg=param_dict_sg)
        param_dict_sg['standard_deviation'] = sg_standard_deviation_01
        sg01 = Explainer(method='sg',
                         model=model,
                         dataset_tensor=dataset_tensor,
                         param_dict_sg=param_dict_sg)
        
        # Shap
        param_dict_shap = dict()
        param_dict_shap['subset_size'] = shap_subset_size
        shap = Explainer(method='shap',
                         model=model,
                         dataset_tensor=dataset_tensor,
                         param_dict_shap=param_dict_shap)
        
        # lime
        param_dict_lime = dict()
        param_dict_lime['dataset_tensor'] = dataset_tensor
        param_dict_lime['std'] = lime_standard_deviation_003
        param_dict_lime['mode'] = lime_mode
        param_dict_lime['sample_around_instance'] = lime_sample_around_instance
        param_dict_lime['kernel_width'] = lime_kernel_width
        param_dict_lime['n_samples'] = lime_n_samples
        param_dict_lime['discretize_continuous'] = lime_discretize_continuous
        lime003 = Explainer(method='lime',
                            model=model,
                            dataset_tensor=dataset_tensor,
                            param_dict_lime=param_dict_lime)
        param_dict_lime['std'] = lime_standard_deviation_005
        lime005 = Explainer(method='lime',
                            model=model,
                            dataset_tensor=dataset_tensor,
                            param_dict_lime=param_dict_lime)
        param_dict_lime['std'] = lime_standard_deviation_01
        lime01 = Explainer(method='lime',
                           model=model,
                           dataset_tensor=dataset_tensor,
                           param_dict_lime=param_dict_lime)
        
        # control
        control = Explainer(method='control',
                            model=model,
                            dataset_tensor=dataset_tensor)
        
        self.explainers_dict = {
            'grad': grad,
            'ig': ig,
            'itg': itg,
            'sg005': sg005,
            'shap': shap,
            'lime005': lime005,
            'control': control
        }
        
        self.experiment_directory = f'./Experiments/{self.experiment_name_str}'
        # make the directory if it does not yet exist
        os.makedirs(f'{self.experiment_directory}/', exist_ok=True)
        
    def _get_predicted_class(self, x):
        """ Returns the predicted class of self.model(x).
        
        Args:
            x: single input of shape (0, d) with d features.
        """
        y_prbs = self.model(x.float())
        return torch.argmax(y_prbs, dim = 1)
    
    def _compute_explanations_for_point(self, x, predicted_label) -> dict:
        """ Stores the explanations on point x in a dictionary object. """
        
        point_metadata = {}
        point_metadata['x'] = x
        point_metadata['y_pred'] = predicted_label
        
        for key, exp_method in tqdm(self.explainers_dict.items()):
            inp = x.detach().float().reshape(1, -1).float()
            lab = predicted_label.type(torch.int64)
            exp = exp_method.get_explanation(inp, label=lab)
            point_metadata[key] = exp
            
        return point_metadata
        
    def get_explanations(self,
                         inputs,
                         labels,
                         num_perturbations = 50):
        """ Computes explanations for all methods in self.explainers_dict for all points in inputs.
            If the explanations have previously been computed, this method loads them from a file.
        """
        # check to see if the explanations already exist
        self.explanation_dict_path = f'{self.experiment_directory}/explanations.p'
        
        if os.path.isfile(self.explanation_dict_path):
            self.explanation_dict = pickle.load(open(self.explanation_dict_path, 'rb'))
            print(f'Loaded explanations from: {self.explanation_dict_path}')
            
        else:
            print(f'Computing explanations for {num_perturbations} perturbations for {inputs.shape[0]} points...')
            start = time.time()
            
            # For each input in inputs, generate num_perturbations perturbations
            
            # store original points, perturbations, and explanations for each perturbation
            
            data_dict = {}
            data_dict['original_points'] = []
            data_dict['perturbations'] = []
            
            for point_idx, x in enumerate(inputs):
                y_pred = self._get_predicted_class(x.unsqueeze(0))
                x_metadata = self._compute_explanations_for_point(x, y_pred)
                data_dict['original_points'].append(x_metadata)
                
                perturbation_metadata = []
                
                # Perturb all inputs
                x_prime_samples = self.perturbation.get_perturbed_inputs(original_sample=x,
                                                                         feature_mask=torch.zeros(x.shape,
                                                                                                  dtype=torch.bool),
                                                                         num_samples=1000,
                                                                         max_distance=perturbation_max_distance,
                                                                         feature_metadata=self.feature_metadata)
                
                # Take the first num_perturbations points that have the same predicted class label
                y_prime_preds = self._get_predicted_class(x_prime_samples)
                
                ind_same_class = (y_prime_preds == y_pred).nonzero()[: num_perturbations].squeeze()
                x_prime_samples = torch.index_select(input=x_prime_samples,
                                                     dim=0,
                                                     index=ind_same_class)
                
                
                # For each perturbation, calculate the explanation
                for x_prime in x_prime_samples:
                    x_prime_metadata = self._compute_explanations_for_point(x_prime, y_pred)
                    x_prime_metadata['original_point_idx'] = point_idx
                    perturbation_metadata.append(x_prime_metadata)

                data_dict['perturbations'].append(perturbation_metadata)

            # Pickle and dump in file.
            pickle.dump(data_dict, open(self.explanation_dict_path, 'wb'))
            print(f'Explanations dumped to {self.explanation_dict_path}')
            print(f'Time elapsed: {str(timedelta(seconds=time.time() - start))}')
            self.explanation_dict = data_dict
            
        self.inputs = [self.explanation_dict['original_points'][ind]['x'] for ind in range(len(self.explanation_dict['original_points']))]
        self.labels = [self.explanation_dict['original_points'][ind]['y_pred'] for ind in range(len(self.explanation_dict['original_points']))]
        self.num_points = len(self.inputs)

    def run_stability_experiments(self,
                                  version_str: str,
                                  use_stability_threshold: bool = False
                                  ):
        '''
        Stores the relative stability (version specified by version_str) on each point in inputs.
        '''

        explanation_distances = []
        stab_measures = []
        denominator_distances = []
        methods = []
        
        inputs = [self.explanation_dict['original_points'][ind]['x'] for ind in range(self.num_points)]

        # Make sure that the output difference is computed correctly!
        if version_str == 'diff_output':
            self.Lmap = self.model

        for key, exp_method in tqdm(self.explainers_dict.items()):
            print('  Explanation Method:', key)
            
            # For each point, run the stability method
            
            for iter in range(self.num_points):
                
                # Get perturbations for the point
                perturbation_metadata = self.explanation_dict['perturbations'][iter]
                
                exp_at_input = self.explanation_dict['original_points'][iter][key]
                
                # Group by explanation type
                x_primes = []
                exp_primes = []
                
                for sample in perturbation_metadata:
                    x_primes.append(sample['x'])
                    exp_primes.append(sample[key])
                
                input_dict = dict_collector(iter,
                                            self.Lmap,
                                            self.perturbation,
                                            self.labels,
                                            3,
                                            self.inputs,
                                            self.model,
                                            explanation_method=exp_method,
                                            explanation = exp_at_input,
                                            feature_metadata=self.feature_metadata,
                                            eval_metric='eval_relative_stability')

                evaluator = Evaluator(input_dict)

                if version_str == 'diff_input':
                    stability, stability_ratios, rep_diffs, x_diffs, exp_diffs, ind_max = evaluator.eval_relative_stability(
                        x_prime_samples = x_primes, exp_prime_samples = exp_primes, exp_at_input = exp_at_input,
                        use_threshold=use_stability_threshold,
                        rep_denominator_flag=False)

                elif version_str == 'diff_output':
                    stability, stability_ratios, rep_diffs, x_diffs, exp_diffs, ind_max = evaluator.eval_relative_stability(
                        x_prime_samples = x_primes, exp_prime_samples = exp_primes, exp_at_input = exp_at_input,
                        use_threshold=use_stability_threshold,
                        rep_denominator_flag=True)

                elif version_str == 'diff_representation':
                    stability, stability_ratios, rep_diffs, x_diffs, exp_diffs, ind_max = evaluator.eval_relative_stability(
                        x_prime_samples = x_primes, exp_prime_samples = exp_primes, exp_at_input = exp_at_input,
                        use_threshold=use_stability_threshold,
                        rep_denominator_flag=True)
                
                stab_measures.append(stability)
                explanation_distances.append(exp_diffs[ind_max])
                methods.append(key)
                if version_str == 'diff_input':
                    denominator_distances.append(x_diffs[ind_max])
                else:
                    denominator_distances.append(rep_diffs[ind_max])
                
        results = np.c_[np.array(stab_measures), np.array(methods),
                        np.array(denominator_distances), np.array(explanation_distances)]
        results = pd.DataFrame(results)
        results.columns = ['Stability',
                           'Method',
                           'Denominator Distance',
                           'Explanation Distance']

        if use_stability_threshold:
            version_str += '_thresholded'

        os.makedirs(self.experiment_directory, exist_ok=True)
        results.to_csv(f'{self.experiment_directory}/stability2_{version_str}.csv')
