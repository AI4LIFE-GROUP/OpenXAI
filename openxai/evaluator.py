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

metrics_params = {
    'PRA': {},
    'RC': {},
    'FA': {'metric': 'overlap'},
    'RA': {'metric': 'rank'},
    'SA': {'metric': 'sign'},
    'SRA': {'metric': 'ranksign'},
    'PGU': {'num_samples': 100, 'invert': True},
    'PGI': {'num_samples': 100, 'invert': False},
    'RIS': {'num_samples': 100, 'rep_denominator_flag': False},
    'RRS': {'num_samples': 100, 'rep_denominator_flag': True},
    'ROS': {'num_samples': 100, 'rep_denominator_flag': True}
}

ground_truth_metrics = ['PRA', 'RC', 'FA', 'RA', 'SA', 'SRA']
prediction_metrics = ['PGU', 'PGI']
stability_metrics = ['RIS', 'RRS', 'ROS']

class Evaluator():
    """Evaluator object for a given model and metric."""
    def __init__(self, model, metric):
        self.model = model
        if metric not in metrics_dict:
            raise NotImplementedError("This metric is not implemented in the current OpenXAI version.")
        self.metric = metrics_dict[metric]
        self.metrics_params = metrics_params[metric]
        if metric in ground_truth_metrics:
            if hasattr(model, 'return_ground_truth_importance'):
                self.metrics_params['ground_truth'] = self.model.return_ground_truth_importance()
            else:
                raise ValueError("The chosen metric is incompatible with non-linear models.")
        elif metric in prediction_metrics + stability_metrics:
            self.metrics_params['model'] = self.model

    def evaluate(self, explanations, **kwargs):
        """Explanation evaluation of a given metric"""
        self.metrics_params.update(kwargs)  # update metric_params with args
        return self.metric(explanations, **self.metrics_params)