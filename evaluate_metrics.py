
# Utils
import os
import numpy as np
import torch
import time
import warnings; warnings.filterwarnings("ignore")
from openxai.experiment_utils import fill_param_dict

# Default explainer parameters
from generate_explanations import default_param_dicts
explainer_param_strs = {method: '_'.join([f'{k}_{v}' for k, v in default_param_dicts[method].items()])\
                        for method in default_param_dicts}

# Models, Data, Explainers, and Evaluators
from openxai.model import LoadModel
from openxai.dataloader import return_loaders
from openxai.explainer import Explainer
from openxai.evaluator import Evaluator, ground_truth_metrics, prediction_metrics, stability_metrics
from openxai.explainers.perturbation_methods import get_perturb_method


def _construct_kwargs(metric):
    # Compute necessary parameters
    feature_metadata = trainloader.dataset.feature_metadata
    perturb_method = get_perturb_method(std, data_name)

    # Load explanations if necessary
    if metric not in stability_metrics:
        param_str = '_' + explainer_param_strs[method] if explainer_param_strs[method] else ''
        file_name = f'explanations/{data_name}_{model_name}_{method}_{n_test_samples}{param_str}.npy'
        explanations = np.load(file_name)

    # Ground Truth Metrics
    if metric in ground_truth_metrics:
        if metric in ['FA', 'RA', 'SA', 'SRA']:
            kwargs = {
                'explanations': explanations,
                'predictions': predictions,
                'k': k
            }
        elif metric in ['PRA', 'RC']:
            kwargs = {
                'explanations': explanations,
            }
        
    # Prediction metrics
    elif metric in prediction_metrics:
        kwargs = {
            'explanations': explanations,
            'inputs': inputs,
            'k': k,
            'perturb_method': perturb_method,
            'feature_metadata': feature_metadata,
            'num_samples': prediction_num_samples,
            'seed': seed,
            'n_jobs': n_jobs
        }

    # Stability metrics
    elif metric in stability_metrics:
        dataset_tensor = torch.FloatTensor(trainloader.dataset.data)
        param_dict = fill_param_dict(method, default_param_dicts[method], dataset_tensor)
        kwargs = {
             'explainer': Explainer(method, model, param_dict),
             'inputs': inputs,
             'perturb_method': perturb_method,
             'feature_metadata': feature_metadata,
             'p_norm': p_norm,
             'num_samples': stability_num_samples,
             'num_perturbations': num_perturbations,
             'seed': seed,
             'n_jobs': n_jobs
        }

    # Exception
    else:
        raise ValueError(f"Metric {metric} not recognized")
    return kwargs

if __name__ == '__main__':
    # Configuration
    model_names = ['lr', 'ann']
    data_names = ['adult', 'compas', 'gaussian', 'german', 'gmsc', 'heart', 'heloc', 'pima']
    methods = ['control', 'grad', 'ig', 'itg', 'sg', 'shap', 'lime']
    n_test_samples = 1000
    metrics = stability_metrics # + ground_truth_metrics + prediction_metrics
    k = 3 # Number of top features to consider for ground truth/prediction metrics
    seed = -1  # -1 to use instance index as seed for stability/prediction metrics
    n_jobs = -1  # Number of parallel jobs for stability/prediction metrics, -1 to use all available cores, None to disable parallelism

    # Ground Truth Parameters
    ground_truth_str = f'_k_{k}'

    # Prediction Parameters
    prediction_num_samples = 100
    prediction_std = 0.1
    prediction_str = f'_std_{prediction_std}_n_samp_{prediction_num_samples}_k_{k}_seed_{seed}'

    # Stability Parameters
    p_norm = 2
    stability_num_samples = 1000
    num_perturbations = 100
    stability_std = 1e-5
    stability_str = f'_std_{stability_std}_n_samp_{stability_num_samples}_n_pert_{num_perturbations}_p_{p_norm}_seed_{seed}'

    # Evaluation Loop
    exp_num, num_exps = 1, len(model_names) * len(data_names) * len(methods)
    start_time = time.time()
    for model_name in model_names:
        start_time_model = time.time()

        # Load model and data
        for data_name in data_names:
            start_time_data = time.time()
            model = LoadModel(data_name, model_name)
            trainloader, testloader = return_loaders(data_name)
            inputs = torch.FloatTensor(testloader.dataset.data[:n_test_samples])
            predictions = model(inputs).argmax(-1)
            metrics_folder_name = f'metric_evals/{model_name}_{data_name}/'
            if not os.path.exists(metrics_folder_name):
                os.makedirs(metrics_folder_name)

            # Load explanations
            for method in methods:
                now = time.time()
                print(f"Model: {model_name}, Data: {data_name}, Explainer: {method} ({exp_num}/{num_exps}, {int(now - start_time)}s total, {int(now - start_time_model)}s on model, {int(now - start_time_data)}s on dataset)")
                exp_num += 1

                # Evaluate metrics
                for metric in metrics:
                    if model_name == 'ann' and metric in ground_truth_metrics:
                        continue
                    elif model_name == 'lr' and metric == 'RRS':
                        continue
                    # Set kwargs
                    std = stability_std if metric in stability_metrics else prediction_std
                    num_samples = stability_num_samples if metric in stability_metrics else prediction_num_samples
                    kwargs = _construct_kwargs(metric)

                    # Evaluate metric
                    evaluator = Evaluator(model, metric=metric)
                    score, mean_score = evaluator.evaluate(**kwargs)
                    std_err = np.std(score) / np.sqrt(len(score))
                    print(f"{metric}: {mean_score:.2f}\u00B1{std_err:.2f}")
                    if metric in stability_metrics:
                        print(f"log({metric}): {np.log(mean_score):.2f}\u00B1{np.log(std_err):.2f}")

                    # Save results
                    if metric in stability_metrics:
                        param_str = stability_str
                    elif metric in prediction_metrics:
                        param_str = prediction_str
                    elif metric in ground_truth_metrics:
                        if metric in ['PRA', 'RC']:
                            param_str = ''
                        else:
                            param_str = ground_truth_str
                    np.save(metrics_folder_name + f'{metric}_{method}_{n_test_samples}{param_str}.npy', score)
                    print()
