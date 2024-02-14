import numpy as np
import torch
from scipy.stats import pearsonr, rankdata
import itertools
from scipy.special import comb
import pandas as pd


class Evaluator():
    """ Metrics to evaluate an explanation method.
    """

    def __init__(self, input_dict: dict, inputs, labels, model, explainer):
        self.input_dict = input_dict
        self.inputs = inputs
        self.labels = labels
        self.model = model
        self.explainer = explainer
        if hasattr(model, 'return_ground_truth_importance'):
            self.gt_feature_importances = self.model.return_ground_truth_importance(self.inputs)
        self.explanation_x_f = self.input_dict['explanation_x']
        self.y_pred = self.input_dict['y_pred']

    def _compute_flattened_explanation_for_predicted_label(self) -> np.ndarray:
        """ Returns a np.ndarray containing the explanation at x with respect to label y_pred.
        """
        return self.explainer.get_explanation(self.inputs.float().reshape(1, -1),
                                              label=self.y_pred).flatten()

    def evaluate(self, metric: str):
        """Explanation evaluation of a given metric
        """
        if not hasattr(self.model, 'return_ground_truth_importance') and metric in ['PRA', 'RC', 'FA', 'RA', 'SA', 'SRA']:
            raise ValueError("This chosen metric is incompatible with non-linear models.")

        # Pairwise rank agreement
        if metric == 'PRA':
            scores, average_score = self.pairwise_comp()
            return scores, average_score
        # Rank correlation
        elif metric == 'RC':
            scores, average_score = self.rankcorr()
            return scores, average_score
        # Feature Agreement
        elif metric == 'FA':
            scores, average_score = self.agreement_fraction(metric='overlap')
            return scores, average_score
        # Rank Agreement
        elif metric == 'RA':
            scores, average_score = self.agreement_fraction(metric='rank')
            return scores, average_score
        # Sign Agreement
        elif metric == 'SA':
            scores, average_score = self.agreement_fraction(metric='sign')
            return scores, average_score
        # Signed Rank Agreement
        elif metric == 'SRA':
            scores, average_score = self.agreement_fraction(metric='ranksign')
            return scores, average_score
        # Prediction Gap on Important Features
        elif metric == 'PGI':
            scores = self.eval_pred_faithfulness(num_samples=100,
                                                 invert=False)
            return scores
        # Prediction Gap on Unimportant Features
        elif metric == 'PGU':
            scores = self.eval_pred_faithfulness(num_samples=100,
                                                 invert=True)
            return scores
        # Relative Input Stability
        elif metric == 'RIS':
            scores = self.eval_relative_stability(num_samples=100,
                                                  rep_denominator_flag=False)
            return scores
        # Relative Representation Stability
        elif metric == 'RRS':
            scores = self.eval_relative_stability(num_samples=100,
                                                  rep_denominator_flag=True)
            return scores
        # Relative Output Stability
        elif metric == 'ROS':
            scores = self.eval_relative_stability(num_samples=100,
                                                  rep_denominator_flag=True)
            return scores
        else:
            raise NotImplementedError("This metric is not implemented in this OpenXAI version.")

    def rankcorr(self):
        '''
        attrA: np.array, n x p
        attrB: np.array, n x p
        '''
        attrA = self.gt_feature_importances.detach().numpy().reshape(1, -1)
        attrB = self.explanation_x_f.detach().numpy().reshape(1, -1)

        corrs = []
        # rank features (accounting for ties)
        # rankdata gives rank1 for smallest # --> we want rank1 for largest # (aka # with largest magnitude)
        all_feat_ranksA = rankdata(-np.abs(attrA), method='dense', axis=1)
        all_feat_ranksB = rankdata(-np.abs(attrB), method='dense', axis=1)

        for row in range(attrA.shape[0]):
            # Calculate correlation on ranks (iterate through rows: https://stackoverflow.com/questions/44947030/how-to-get-scipy-stats-spearmanra-b-compute-correlation-only-between-variable)
            rho, _ = pearsonr(all_feat_ranksA[row, :], all_feat_ranksB[row, :])
            corrs.append(rho)
        
        # return metric's distribution and average
        return np.array(corrs), np.mean(corrs)

    def pairwise_comp(self):
        '''
        inputs
        attrA: np.array, n x p
        attrB: np.array, n x p
        outputs:
        pairwise_distr: 1D numpy array (dimensions=(n,)) of pairwise comparison agreement for each data point
        pairwise_avg: mean of pairwise_distr
        '''

        attrA = self.gt_feature_importances.detach().numpy().reshape(1, -1)
        attrB = self.explanation_x_f.detach().numpy().reshape(1, -1)

        n_datapoints = attrA.shape[0]
        n_feat = attrA.shape[1]

        # rank of all features --> manually calculate rankings (instead of using 0, 1, ..., k ranking based on argsort output) to account for ties
        # rankdata gives rank1 for smallest # --> we want rank1 for largest # (aka # with largest magnitude)
        all_feat_ranksA = rankdata(-np.abs(attrA), method='dense', axis=1)
        all_feat_ranksB = rankdata(-np.abs(attrB), method='dense', axis=1)

        # count # of pairs of features with same relative ranking
        feat_pairs_w_same_rel_rankings = np.zeros(n_datapoints)

        for feat1, feat2 in itertools.combinations_with_replacement(range(n_feat), 2):
            if feat1 != feat2:
                rel_rankingA = all_feat_ranksA[:, feat1] < all_feat_ranksA[:, feat2]
                rel_rankingB = all_feat_ranksB[:, feat1] < all_feat_ranksB[:, feat2]
                feat_pairs_w_same_rel_rankings += rel_rankingA == rel_rankingB

        pairwise_distr = feat_pairs_w_same_rel_rankings/comb(n_feat, 2)

        return pairwise_distr, np.mean(pairwise_distr)

    def agreement_fraction(self, metric=None):

        attrA = self.gt_feature_importances.detach().numpy().reshape(1, -1)
        attrB = self.explanation_x_f.detach().numpy().reshape(1, -1)
        k = self.input_dict['top_k']
        
        if metric is None:
            metric_type = self.input_dict['eval_metric']
        else:
            metric_type = metric

        # id of top-k features
        topk_idA = np.argsort(-np.abs(attrA), axis=1)[:, 0:k]
        topk_idB = np.argsort(-np.abs(attrB), axis=1)[:, 0:k]

        # rank of top-k features --> manually calculate rankings (instead of using 0, 1, ..., k ranking based on argsort output) to account for ties
        all_feat_ranksA = rankdata(-np.abs(attrA), method='dense', axis=1) #rankdata gives rank1 for smallest # --> we want rank1 for largest # (aka # with largest magnitude)
        all_feat_ranksB = rankdata(-np.abs(attrB), method='dense', axis=1)
        topk_ranksA = np.take_along_axis(all_feat_ranksA, topk_idA, axis=1)
        topk_ranksB = np.take_along_axis(all_feat_ranksB, topk_idB, axis=1)

        # sign of top-k features
        topk_signA = np.take_along_axis(np.sign(attrA), topk_idA, axis=1)  #pos=1; neg=-1
        topk_signB = np.take_along_axis(np.sign(attrB), topk_idB, axis=1)

        # overlap agreement = (# topk features in common)/k
        if metric_type == 'overlap':
            topk_setsA = [set(row) for row in topk_idA]
            topk_setsB = [set(row) for row in topk_idB]
            # check if: same id
            metric_distr = np.array([len(setA.intersection(setB))/k for setA, setB in zip(topk_setsA, topk_setsB)])

        # rank agreement
        elif metric_type == 'rank':
            topk_idA_df = pd.DataFrame(topk_idA).applymap(str)  # id
            topk_idB_df = pd.DataFrame(topk_idB).applymap(str)
            topk_ranksA_df = pd.DataFrame(topk_ranksA).applymap(str)  # rank (accounting for ties)
            topk_ranksB_df = pd.DataFrame(topk_ranksB).applymap(str)
            
            #check if: same id + rank
            topk_id_ranksA_df = ('feat' + topk_idA_df) + ('rank' + topk_ranksA_df)
            topk_id_ranksB_df = ('feat' + topk_idB_df) + ('rank' + topk_ranksB_df)
            metric_distr = (topk_id_ranksA_df == topk_id_ranksB_df).sum(axis=1).to_numpy()/k

        # sign agreement
        elif metric_type == 'sign':
            topk_idA_df = pd.DataFrame(topk_idA).applymap(str)  # id (contains rank info --> order of features in columns)
            topk_idB_df = pd.DataFrame(topk_idB).applymap(str)
            topk_signA_df = pd.DataFrame(topk_signA).applymap(str)  # sign
            topk_signB_df = pd.DataFrame(topk_signB).applymap(str)
            
            #check if: same id + sign
            topk_id_signA_df = ('feat' + topk_idA_df) + ('sign' + topk_signA_df)  # id + sign (contains rank info --> order of features in columns)
            topk_id_signB_df = ('feat' + topk_idB_df) + ('sign' + topk_signB_df)
            topk_id_signA_sets = [set(row) for row in topk_id_signA_df.to_numpy()]  # id + sign (remove order info --> by converting to sets)
            topk_id_signB_sets = [set(row) for row in topk_id_signB_df.to_numpy()]
            metric_distr = np.array([len(setA.intersection(setB))/k for setA, setB in zip(topk_id_signA_sets, topk_id_signB_sets)])

        # rank and sign agreement
        elif metric_type == 'ranksign':
            topk_idA_df = pd.DataFrame(topk_idA).applymap(str)  # id
            topk_idB_df = pd.DataFrame(topk_idB).applymap(str)
            topk_ranksA_df = pd.DataFrame(topk_ranksA).applymap(str)  # rank (accounting for ties)
            topk_ranksB_df = pd.DataFrame(topk_ranksB).applymap(str)
            topk_signA_df = pd.DataFrame(topk_signA).applymap(str)  # sign
            topk_signB_df = pd.DataFrame(topk_signB).applymap(str)
            
            # check if: same id + rank + sign
            topk_id_ranks_signA_df = ('feat' + topk_idA_df) + ('rank' + topk_ranksA_df) + ('sign' + topk_signA_df)
            topk_id_ranks_signB_df = ('feat' + topk_idB_df) + ('rank' + topk_ranksB_df) + ('sign' + topk_signB_df)
            metric_distr = (topk_id_ranks_signA_df == topk_id_ranks_signB_df).sum(axis=1).to_numpy()/k
        
        else:
            raise NotImplementedError("Please make sure that have chosen one of the following metrics: {ranksign, rank, overlap, sign}.")

        return metric_distr, np.mean(metric_distr)


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
            
    def eval_pred_faithfulness(self, num_samples: int = 100, invert: bool = False):
        """ Approximates the expected local faithfulness of the explanation
            in a neighborhood around input x.
        Args:
            num_perturbations: number of perturbations used for Monte Carlo expectation estimate
        """
        self._parse_and_check_input(eval_metric='eval_pred_faithfulness')

        if invert:
            self.top_k_mask = torch.logical_not(self.top_k_mask)
        
        # get perturbations of instance x
        x_perturbed = self.perturb_method.get_perturbed_inputs(original_sample=self.x,
                                                               feature_mask=self.top_k_mask,
                                                               num_samples=num_samples,
                                                               max_distance=self.perturb_max_distance,
                                                               feature_metadata=self.feature_metadata)

        # Average the expected absolute difference.
        y = self._arr(self.model(self.x.reshape(1, -1).float()))
        y_perturbed = self._arr(self.model(x_perturbed.float()))

        return np.mean(np.abs(y - y_perturbed), axis=0)[0]

    def _compute_Lp_norm_diff(self, vec1, vec2, normalize_to_relative_change: bool = True, eps: float = 0.001):
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
    
    def _get_predicted_class(self, x):
        """ Returns the predicted class of self.model(x).
        
        Args:
            x: single input of shape (0, d) with d features.
        """
        y_prbs = self.model(x.float())
        return torch.argmax(y_prbs, dim=1)

    def eval_relative_stability(self,
                                x_prime_samples=None,
                                exp_prime_samples=None,
                                rep_denominator_flag: bool = False,
                                num_samples: int = 1000,
                                num_perturbations: int = 50):
        """ Approximates the maximum L-p distance between explanations in a neighborhood around
            input x.

        Args:
            rep_denominator_flag: when true, normalizes the stability metric by the L-p distance
                between representations (instead of features).

        """
        exp_at_input = self.explanation_x_f
        self._parse_and_check_input(eval_metric='eval_relative_stability')

        stability_ratios = []
        rep_diffs = []
        x_diffs = []
        exp_diffs = []
        
        # get perturbations of instance x, and for each perturbed instance compute an explanation
        if x_prime_samples is None:
            # Perturb input
            x_prime_samples = self.perturb_method.get_perturbed_inputs(original_sample=self.x,
                                                                       feature_mask=self.top_k_mask,
                                                                       num_samples=num_samples,
                                                                       max_distance=self.perturb_max_distance,
                                                                       feature_metadata=self.feature_metadata)

            # Take the first num_perturbations points that have the same predicted class label
            y_prime_preds = self._get_predicted_class(x_prime_samples)

            ind_same_class = (y_prime_preds == self.y_pred).nonzero()[: num_perturbations].squeeze()
            x_prime_samples = torch.index_select(input=x_prime_samples,
                                                 dim=0,
                                                 index=ind_same_class)
            y_prime_preds = self._get_predicted_class(x_prime_samples)

            # For each perturbation, calculate the explanation
            exp_prime_samples = torch.zeros_like(x_prime_samples)
            for it, x_prime in enumerate(x_prime_samples):
                x_prime = x_prime.reshape(1, -1)
                lab = y_prime_preds[it].type(torch.int64)
                exp = self.explainer.get_explanation(x_prime.float(), label=lab)
                exp_prime_samples[it, :] = exp
            
        for sample_ind, x_prime in enumerate(x_prime_samples):
            x_prime = x_prime.unsqueeze(0)

            rep_diff = self._compute_Lp_norm_diff(self.L_map(self.x.float().reshape(1, -1)),
                                                  self.L_map(x_prime.float().reshape(1, -1)),
                                                  normalize_to_relative_change=True)
            rep_diffs.append(rep_diff)

            exp_at_perturbation = exp_prime_samples[sample_ind]

            # for predictions per perturbation
            explanation_diff = self._compute_Lp_norm_diff(exp_at_input,
                                                          exp_at_perturbation,
                                                          normalize_to_relative_change=True)

            exp_diffs.append(explanation_diff)

            if rep_denominator_flag:
                # compute norm between representations
                stability_measure = np.divide(explanation_diff, rep_diff)
            else:
                feature_difference = self._compute_Lp_norm_diff(self.x, x_prime)
                stability_measure = np.divide(explanation_diff, feature_difference)
                x_diffs.append(self._compute_Lp_norm_diff(self.x, x_prime))

            stability_ratios.append(stability_measure)
                
        ind_max = np.argmax(stability_ratios)

        return stability_ratios[ind_max]  # , stability_ratios, rep_diffs, x_diffs, exp_diffs, ind_max
