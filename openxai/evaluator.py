from openxai.metrics import pairwise_comp, rankcorr, eval_ground_truth_faithfulness
from openxai.metrics import eval_pred_faithfulness, eval_relative_stability

metrics_dict = {
    'PRA': pairwise_comp,
    'RC':  rankcorr,
    'FA':  eval_ground_truth_faithfulness,
    'RA':  eval_ground_truth_faithfulness,
    'SA':  eval_ground_truth_faithfulness,
    'SRA': eval_ground_truth_faithfulness,
    'PGI': eval_pred_faithfulness,
    'PGU': eval_pred_faithfulness,
    'RIS': eval_relative_stability,
    'RRS': eval_relative_stability,
    'ROS': eval_relative_stability
}

stability_params = {'p_norm': 2, 'num_samples': 1000, 'num_perturbations': 100, 'seed': -1, 'n_jobs': -1}
prediction_params = {'num_samples': 100, 'seed': -1}
metrics_params = {
    'PRA': {},
    'RC': {},
    'FA': {'metric': 'overlap'},
    'RA': {'metric': 'rank'},
    'SA': {'metric': 'sign'},
    'SRA': {'metric': 'ranksign'},
    'PGU': {**{'invert': True}, **prediction_params},
    'PGI': {**{'invert': False}, **prediction_params},
    'RIS': {**{'metric': 'RIS'}, **stability_params},
    'RRS': {**{'metric': 'RRS'}, **stability_params},
    'ROS': {**{'metric': 'ROS'}, **stability_params}
}

ground_truth_metrics = ['PRA', 'RC', 'FA', 'RA', 'SA', 'SRA']
prediction_metrics = ['PGU', 'PGI']
stability_metrics = ['RIS', 'RRS', 'ROS']

class Evaluator():
    """Evaluator object for a given model and metric."""
    def __init__(self, model, metric):
        # Set model and metric
        if metric not in metrics_dict:
            raise NotImplementedError(f"The metric {metric} is not implemented in the current OpenXAI version.")
        self.model = model
        self.metric = metric
        self.metric_fn = metrics_dict[metric]
        self.metrics_params = metrics_params[metric]

        # Set ground truth metric parameters
        if metric in ground_truth_metrics:
            if hasattr(model, 'return_ground_truth_importance'):
                self.metrics_params['ground_truth'] = self.model.return_ground_truth_importance()
            else:
                raise ValueError(f"The metric {metric} is incompatible with non-linear models.")
            
        # Set stability/prediction metric parameters
        if metric in stability_metrics + prediction_metrics:
            self.metrics_params['model'] = self.model
            if metric in stability_metrics:
                self.metrics_params['metric'] = metric

    def evaluate(self, **kwargs):
        """Explanation evaluation of a given metric"""
        self.metrics_params.update(kwargs)  # update metric_params with args
        return self.metric_fn(**self.metrics_params)