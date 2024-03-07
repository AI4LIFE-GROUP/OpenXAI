
# Utils
import numpy as np
import time
import copy
import warnings; warnings.filterwarnings("ignore")
import openxai.experiment_utils as utils

# Models, Data, Explainers, and Evaluators
from openxai.model import LoadModel
from openxai.dataloader import ReturnTrainTestX
from openxai.explainer import Explainer
from openxai.evaluator import Evaluator, ground_truth_metrics, prediction_metrics, stability_metrics
from openxai.explainers.perturbation_methods import get_perturb_method

def _construct_param_dict(config, metric):
    # Ground truth metrics PRA, RC, FA, SA, SRA
    if metric in ground_truth_metrics:
        p_dict = copy.deepcopy(config['evaluators']['ground_truth_metrics'])
        p_str = utils.construct_param_string(p_dict)
        p_dict['explanations'] = utils.load_parameterized_file(\
            f'explanations/{model_name}_{data_name}/{method}_{n_test_samples}', config['explainers'][method])
        if metric in ['FA', 'RA', 'SA', 'SRA']:
            p_dict['predictions'] = predictions  # flips ground truth according to prediction
        elif metric in ['PRA', 'RC']:
            del p_dict['k'], p_dict['AUC']
            p_str = ''

    # Prediction metrics PGU, PGI
    elif metric in prediction_metrics:
        p_dict = copy.deepcopy(config['evaluators']['prediction_metrics'])
        p_str = utils.construct_param_string(p_dict)
        p_dict['inputs'] = X_test
        p_dict['explanations'] = utils.load_parameterized_file(\
            f'explanations/{model_name}_{data_name}/{method}_{n_test_samples}', config['explainers'][method])
        p_dict['perturb_method'] = get_perturb_method(p_dict['std'], data_name)
        p_dict['feature_metadata'] = feature_metadata
        del p_dict['std']

    # Stability metrics RIS, RRS, ROS
    elif metric in stability_metrics:
        exp_p_dict = utils.fill_param_dict(method, copy.deepcopy(config['explainers'][method]), X_train)
        p_dict = copy.deepcopy(config['evaluators']['stability_metrics'])
        p_str = utils.construct_param_string(p_dict)
        p_dict['inputs'] = X_test
        p_dict['explainer'] = Explainer(method, model, exp_p_dict)
        p_dict['perturb_method'] = get_perturb_method(p_dict['std'], data_name)
        p_dict['feature_metadata'] = feature_metadata
        del p_dict['std']

    # Exception
    else:
        raise ValueError(f"Metric {metric} not recognized")
    
    return p_dict, p_str

if __name__ == '__main__':
    # Configuration
    config = utils.load_config('experiment_config.json')
    model_names, data_names = config['model_names'], config['data_names']
    methods, metrics = config['methods'], config['metrics']
    n_test_samples = config['n_test_samples']
    
    # Initialize trackers
    exp_num, num_exps = 1, len(model_names) * len(data_names) * len(methods)
    start_time = time.time()

    # Loop over models
    for model_name in model_names:
        start_time_model = time.time()

        # Loop over datasets
        for data_name in data_names:
            start_time_data = time.time()
            metrics_folder_name = f'metric_evals/{model_name}_{data_name}/'
            utils.make_directory(metrics_folder_name)

            # Load data and model
            X_train, X_test, feature_metadata =\
                ReturnTrainTestX(data_name, n_test=n_test_samples, float_tensor=True,
                                 return_feature_metadata=True)
            model = LoadModel(data_name, model_name)
            predictions = model(X_test).argmax(-1)

            # Loop over explanation methods
            for method in methods:
                # Initialize trackers
                now = time.time()
                print(f"Model: {model_name}, Data: {data_name}, Explainer: {method} ({exp_num}/{num_exps})"+\
                      f"{int(now - start_time)}s total, {int(now - start_time_model)}s on model, {int(now - start_time_data)}s on dataset)")
                exp_num += 1

                # Loop over metrics
                for metric in metrics:
                    # Skip invalid combinations
                    if utils.invalid_model_metric_combination(model_name, metric):
                        print(f"Skipping {metric} for {model_name}")
                        continue

                    # Evaluate metric
                    evaluator = Evaluator(model, metric=metric)
                    param_dict, param_str = _construct_param_dict(config, metric)
                    score, mean_score = evaluator.evaluate(**param_dict)

                    # Print results
                    std_err = np.std(score) / np.sqrt(len(score))
                    print(f"{metric}: {mean_score:.3f} \u00B1 {std_err:.3f}")
                    if metric in stability_metrics:
                        log_mu, log_std = np.log(mean_score), np.log(std_err)
                        print(f"log({metric}): {log_mu:.3f} \u00B1 {log_std:.3f}")

                    # Save results
                    np.save(metrics_folder_name + f'{metric}_{method}_{n_test_samples}{param_str}.npy', score)
                    print()