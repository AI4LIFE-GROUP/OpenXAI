# from explainers.perturbation_methods import BasePerturbation
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from scipy import stats


class Evaluator():
    """ Metrics to evaluate an explanation method.
    """

    def __init__(self, input_dict: dict):
        self.input_dict = input_dict

    def _arr(self, x) -> np.ndarray:
        """ Converts x to a numpy array.
        """
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.array(x)

    def _compute_top_k_mask_from_explanation(self, explanation: torch.Tensor, top_k: int) -> torch.BoolTensor:
        """ Returns a mask for the top k features with the largest explanation magnitudes.
        """
        # Sort the explanation magnitudes in descending order.
        top_k_vals, top_k_ind_flat = torch.abs(explanation).flatten().topk(top_k)

        # use the smallest magnitude still in the top K to threshold.
        top_k_magnitude_threshold = top_k_vals[-1]
        mask = torch.abs(explanation) > top_k_magnitude_threshold
        return mask
    
    def _compute_flattened_explanation_for_predicted_label(self) -> np.ndarray:
        """ Returns a np.ndarray containing the explanation at x with respect to label y_pred.
        """
        return self.explainer.get_explanation(self.x.float().reshape(1, -1), label=self.y_pred).flatten()

    def _parse_and_check_input(self, eval_metric: str):
        # explanation explanation_x
        # this is not needed
        '''
        if eval_metric in ['eval_pred_faithfulness', 'eval_relative_stability']:
            if not 'explanation_x' in self.input_dict:
                raise ValueError('Missing key of explanation_x')
            self.explanation_x = self.input_dict['explanation_x']
        '''
        # input x
        if eval_metric in ['eval_pred_faithfulness', 'eval_relative_stability',
                           'eval_counterfactual_fairness', 'eval_group_fairness',
                           'eval_gt_similarity', 'eval_gt_rank', 'eval_gt_f1k']:
            if not 'x' in self.input_dict:
                raise ValueError('Missing key of x')
            self.x = self.input_dict['x']

        # input y
        if eval_metric in ['eval_relative_stability']:
            if not 'y' in self.input_dict:
                raise ValueError('Missing key of y')
            self.y = self.input_dict['y']

        # input y_pred
        if eval_metric in ['eval_relative_stability', 'eval_gt_similarity', 'eval_gt_rank', 'eval_gt_f1k']:
            if not 'y_pred' in self.input_dict:
                raise ValueError('Missing key of y_pred')
            self.y_pred = self.input_dict['y_pred']

        # predictive model
        if eval_metric in ['eval_pred_faithfulness', 'eval_relative_stability', 'eval_group_fairness', 'eval_gt_similarity','eval_gt_rank', 'eval_gt_f1k']:
            if not 'model' in self.input_dict:
                raise ValueError('Missing key of model')
            self.model = self.input_dict['model']

        # callable explainer class
        if eval_metric in ['eval_relative_stability', 'eval_group_fairness', 'eval_gt_similarity', 'eval_gt_rank', 'eval_gt_f1k']:
            if not 'explainer' in self.input_dict:
                raise ValueError('Missing key of explainer')
            self.explainer = self.input_dict['explainer']

        # top-K parameter K
        if eval_metric in ['eval_pred_faithfulness', 'eval_relative_stability', 'eval_group_fairness', 'eval_gt_f1k']:
            if not 'top_k' in self.input_dict:
                raise ValueError('Missing key of top_k')
            self.top_k = self.input_dict['top_k']

        # representation map L
        if eval_metric in ['eval_relative_stability', 'eval_counterfactual_fairness']:
            if not 'L_map' in self.input_dict:
                raise ValueError('Missing key L_map')
            self.L_map = self.input_dict['L_map']

        # p-norm
        if eval_metric in ['eval_relative_stability', 'eval_counterfactual_fairness']:
            if not 'p_norm' in self.input_dict:
                raise ValueError('Missing key p_norm')
            self.p_norm = self.input_dict['p_norm']

        # counterfactual input x
        if eval_metric in ['eval_counterfactual_fairness']:
            if not 'x_cf' in self.input_dict:
                raise ValueError('Missing key x_cf')
            self.x_cf = self.input_dict['x_cf']

            if not 'explanation_x_cf' in self.input_dict:
                raise ValueError('Missing key explanation_x_cf')
            self.explanation_x_cf = self.input_dict['explanation_x_cf']

        # initialized perturbation method object BasePerturbation pertub_method
        if eval_metric in ['eval_pred_faithfulness', 'eval_relative_stability', 'eval_group_fairness']:
            if not 'perturb_method' in self.input_dict:
                raise ValueError('Missing key of perturbation method BasePerturbation perturb_method')
            # initialize the perturbation method, which extends from BasePerturbation
            self.perturb_method = self.input_dict['perturb_method']

        # initialized perturbation method object BasePerturbation pertub_method
        if eval_metric in ['eval_pred_faithfulness', 'eval_relative_stability', 'eval_group_fairness']:
            if not 'feature_metadata' in self.input_dict:
                raise ValueError('Missing key of feature metadata feature_metadata')
            # initialize the perturbation method, which extends from BasePerturbation
            self.feature_metadata = self.input_dict['feature_metadata']

        # initialized perturbation method object BasePerturbation pertub_method
        if eval_metric in ['eval_relative_stability', 'eval_pred_faithfulness']:
            if not 'input_data' in self.input_dict:
                raise ValueError('Missing key of input_data')
            # initialize the perturbation method, which extends from BasePerturbation
            self.input_data = self.input_dict['input_data']

            # self.top_k_mask = self._compute_top_k_mask_from_explanation(self.explanation_x, self.top_k)
            self.top_k_mask = self.input_dict['mask']

        # perturbation maximum distance perturb_max_distance
        if eval_metric in ['eval_pred_faithfulness', 'eval_relative_stability', 'eval_group_fairness']:
            if not 'perturb_max_distance' in self.input_dict:
                raise ValueError('Missing key of perturbation maximum distance perturb_max_distance')
            self.perturb_max_distance = self.input_dict['perturb_max_distance']

        # sensitive class assignment function get_sens_class_labels
        # boolean function that returns 1 iff input x is in the sensitive class
        if eval_metric in ['eval_group_fairness']:
            if not 'get_sens_class_labels' in self.input_dict:
                raise ValueError('Missing key of sensitive class assignment function get_sens_class_labels')
            self.get_sens_class_labels = self.input_dict['get_sens_class_labels']

        if eval_metric in ['eval_gt_similarity', 'eval_gt_rank', 'eval_gt_f1k']:
            # use the model's return_ground_truth_importance function to get the ground truth
            self.gt_feature_importances = self.model.return_ground_truth_importance(self.x)
            
    def eval_gt_similarity(self):
        """ Measures closeness between explanations generated by explanation method
            and feature coefficients of the model.
        """
        self._parse_and_check_input(eval_metric='eval_gt_similarity')

        explanation_x_f = self._compute_flattened_explanation_for_predicted_label()
        
        # Reshape for a single sample
        cos_sim_scores = cosine_similarity(self._arr(self.gt_feature_importances).reshape(1, -1), 
                                 self._arr(explanation_x_f).reshape(1, -1))

        return cos_sim_scores.item()

    def eval_gt_rank(self):
        """ Assesses correlation between ground truth underlying feature ranking
            and generated feature ranking from explanation.
        """
        self._parse_and_check_input(eval_metric='eval_gt_rank')

        explanation_x_f = self._compute_flattened_explanation_for_predicted_label()

        r_f = stats.rankdata(self._arr(self.gt_feature_importances)) - 1
        explanation_r_e = stats.rankdata(self._arr(explanation_x_f)) - 1

        rho, pval = stats.spearmanr(r_f, explanation_r_e)
        return rho

    def eval_gt_f1k(self):
        """ Calculates F1-score for top-k features using ground-truth and
            predicted mask vectors
        """
        self._parse_and_check_input(eval_metric='eval_gt_f1k')

        m_x_f = self._arr(self._compute_top_k_mask_from_explanation(self.gt_feature_importances, self.top_k).long().squeeze())

        explanation_x_f = self._compute_flattened_explanation_for_predicted_label()
        m_x_f_hat_arr = self._arr(self._compute_top_k_mask_from_explanation(explanation_x_f, self.top_k).long().squeeze())

        return f1_score(m_x_f, m_x_f_hat_arr), accuracy_score(m_x_f, m_x_f_hat_arr), precision_score(m_x_f, m_x_f_hat_arr), recall_score(m_x_f, m_x_f_hat_arr)

    def eval_pred_faithfulness(self, num_samples: int = 100):
        """ Approximates the expected local faithfulness of the explanation
            in a neighborhood around input x.

        Args:
            num_perturbations: number of perturbations used for Monte Carlo expectation estimate

        """
        self._parse_and_check_input(eval_metric='eval_pred_faithfulness')

        x_perturbed = self.perturb_method.get_perturbed_inputs(original_sample=self.x,
                                                               feature_mask=self.top_k_mask,
                                                               num_samples=num_samples,
                                                               max_distance=self.perturb_max_distance,
                                                               feature_metadata=self.feature_metadata)

        # Average the expected absolute difference.
        y = self._arr(self.model(self.x.reshape(1, -1).float()))
        y_perturbed = self._arr(self.model(x_perturbed.float()))

        return np.mean(np.abs(y - y_perturbed), axis=0)

    def _compute_Lp_norm_diff(self, vec1, vec2, normalize_to_relative_change: bool = True, eps: np.float = 0.001):
        """ Returns the Lp norm of the difference between vec1 and vec2.
        Args:
            normalize_by_vec1: when true, normalizes the difference between vec1 and vec2 by vec1
        """

        # arrays can be flattened, so long as ordering is preserved
        flat_diff = self._arr(vec1).flatten() - self._arr(vec2).flatten()

        if normalize_to_relative_change:
            vec1_arr = self._arr(vec1.flatten())
            vec1_arr = np.clip(vec1_arr, eps, None)
            flat_diff = np.divide(flat_diff, vec1_arr, where=vec1_arr != 0)

        return np.linalg.norm(flat_diff, ord=self.p_norm)

    def _compute_threshold(self, num_samples):

        data = self.input_data
        n_samples = data.shape[0]
        mean_rep_diffs = []
        for i in range(n_samples):
            x = data[i, :]
            x_primes = self.perturb_method.get_perturbed_inputs(original_sample=x,
                                                                feature_mask=self.top_k_mask,
                                                                num_samples=num_samples,
                                                                max_distance=self.perturb_max_distance)

            diffs_for_fixed_x = []
            for j in range(num_samples):
                x_prime = x_primes[j, :]
                rep_diff = self._compute_Lp_norm_diff(self.L_map(self.x.float().reshape(1, -1)),
                                                      self.L_map(x_prime.float().reshape(1, -1)),
                                                      normalize_to_relative_change=False)

                diffs_for_fixed_x.append(rep_diff)
            mean_for_fixed_x = (1 / num_samples) * np.sum(diffs_for_fixed_x)
            mean_rep_diffs.append(mean_for_fixed_x)

        threshold = (1 / n_samples) * np.sum(mean_rep_diffs)

        return threshold
    
    def _get_predicted_class(self, x):
        """ Returns the predicted class of self.model(x).
        
        Args:
            x: single input of shape (0, d) with d features.
        """
        y_prbs = self.model(x.float())
        return torch.argmax(y_prbs, dim = 1)

    def eval_relative_stability(self, x_prime_samples, exp_prime_samples, exp_at_input, rep_denominator_flag: bool = False,
                                   use_threshold: bool = False):
        """ Approximates the maximum L-p distance between explanations in a neighborhood around
            input x.

        Args:
            rep_denominator_flag: when true, normalizes the stability metric by the L-p distance
                between representations (instead of features).
            delta: representation distance threshold, excludes points greater than delta distance
                away in representation space

        """
        self._parse_and_check_input(eval_metric='eval_relative_stability')

        stability_ratios = []
        rep_diffs = []
        x_diffs = []
        exp_diffs = []

        for sample_ind, x_prime in enumerate(x_prime_samples):
            x_prime = x_prime.unsqueeze(0)

            rep_diff = self._compute_Lp_norm_diff(self.L_map(self.x.float().reshape(1, -1)),
                                                  self.L_map(x_prime.float().reshape(1, -1)),
                                                  normalize_to_relative_change=True)
            rep_diffs.append(rep_diff)

            exp_at_perturbation = exp_prime_samples[sample_ind]

            # for predictions per perturbation
            explanation_diff = self._compute_Lp_norm_diff(exp_at_input, exp_at_perturbation,
                                                          normalize_to_relative_change=True)

            exp_diffs.append(explanation_diff)

            if use_threshold:
                threshold = self._compute_threshold(num_samples)

                # only evaluate the stability metric for points x_prime with representations close to x
                if rep_diff < threshold:
                    stability_measure = explanation_diff
                    stability_ratios.append(stability_measure)
            else:
                if rep_denominator_flag:
                    # compute norm between representations
                    stability_measure = np.divide(explanation_diff, rep_diff)
                else:
                    feature_difference = self._compute_Lp_norm_diff(self.x, x_prime)
                    stability_measure = np.divide(explanation_diff, feature_difference)
                    x_diffs.append(self._compute_Lp_norm_diff(self.x, x_prime))

                stability_ratios.append(stability_measure)
                
        ind_max = np.argmax(stability_ratios)

        return stability_ratios[ind_max], stability_ratios, rep_diffs, x_diffs, exp_diffs, ind_max
