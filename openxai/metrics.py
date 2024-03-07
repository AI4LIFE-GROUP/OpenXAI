import itertools
from scipy.stats import pearsonr, rankdata
from scipy.special import comb
from sklearn.metrics import auc
import pandas as pd
import numpy as np
import torch
import warnings; warnings.filterwarnings("ignore")
from tqdm import tqdm
from joblib import Parallel, delayed
import openxai.experiment_utils as utils

# ==== METRICS ==== #

# PRA
def pairwise_comp(explanations, ground_truth):
    """
    Compute rank agreement between all possible pairs
    of features in explanations and ground truth.

    :param explanations: np.array of shape (n_inputs, n_features)
    :param ground_truth: np.array of shape (n_features,)
    :return: pairwise_distr: np.array of shape (n_inputs,)
             mean_pairwise_distr: float
    """
    explanations, ground_truth = _preprocess_attributions(explanations, ground_truth)
    feat_pairs_w_same_rel_rankings = np.zeros(explanations.shape[0])

    # use rankdata instead of argsort to account for ties
    exp_ranks = rankdata(-np.abs(explanations), method='dense', axis=1)
    gt_rank = rankdata(-np.abs(ground_truth), method='dense')

    # count # of pairs of features with same relative ranking
    n_feat = explanations.shape[1]
    for feat1, feat2 in itertools.combinations_with_replacement(range(n_feat), 2):
        if feat1 != feat2:
            rel_rankingA = exp_ranks[:, feat1] < exp_ranks[:, feat2]
            rel_rankingB = gt_rank[feat1] < gt_rank[feat2]
            feat_pairs_w_same_rel_rankings += rel_rankingA == rel_rankingB
    pairwise_distr = feat_pairs_w_same_rel_rankings/comb(n_feat, 2)
    return pairwise_distr, np.mean(pairwise_distr)

# RC
def rankcorr(explanations, ground_truth):
    """
    Compute rank correlation between explanations and ground truth.

    :param explanations: np.array of shape (n_inputs, n_features)
    :param ground_truth: np.array of shape (n_features,)
    :return: corrs_distr: np.array of shape (n_inputs,)
             mean_corrs: float
    """
    explanations, ground_truth = _preprocess_attributions(explanations, ground_truth)
    corrs_distr = np.zeros(explanations.shape[0])
    exp_ranks = rankdata(-np.abs(explanations), method='dense', axis=1)
    gt_rank = rankdata(-np.abs(ground_truth), method='dense')
    for row in range(exp_ranks.shape[0]):
        corrs_distr[row], _ = pearsonr(exp_ranks[row], gt_rank)
    return corrs_distr, np.mean(corrs_distr)

# FA, RA, SA, SRA
def eval_ground_truth_faithfulness(explanations, ground_truth, predictions, k, metric, AUC=False):
    """
    Compute agreement fraction between top-k features of two explanations.

    :param explanations: np.array of shape (n_inputs, n_features)
    :param ground_truth: np.array of shape (n_features,)
    :param predictions: np.array of shape (n_inputs,), since we invert the ground truth depending on the prediction we are explaining
    :param k: int, number of top-k features
    :param metric: str, 'overlap', 'rank', 'sign', or 'ranksign'
    :param AUC: bool, whether to compute the AUC of the metric distribution
    :return: metric_distr: np.array of shape (n_inputs,)
             mean_metric: float
    """
    predictions = utils.convert_to_numpy(predictions)
    if explanations.shape[0] != len(predictions):
        raise ValueError('Number of predictions must match number of explanations.')
    explanations, ground_truth = _preprocess_attributions(explanations, ground_truth)
    n_inputs, n_features = explanations.shape
    ground_truths = (predictions*2-1)[:, None] * np.repeat(ground_truth.reshape(1, -1), len(predictions), axis=0)
    max_k = utils.convert_k_to_int(k, n_features)
    if AUC and max_k > 1:
        metric_distr_all_ks = np.array([_single_k_gt_faith(explanations, ground_truths, k, metric) for k in range(1, max_k + 1)])
        metric_distr = np.array([auc(np.arange(max_k)/(max_k-1), metric_distr_all_ks[:, i]) for i in range(n_inputs)])
    else:
        metric_distr = _single_k_gt_faith(explanations, ground_truths, max_k, metric)
    
    return metric_distr, np.mean(metric_distr)

# PGI, PGU
def eval_pred_faithfulness(explanations, inputs, model, k, perturb_method, feature_metadata,
                           n_samples = 100, invert = False, seed=-1, n_jobs=None, AUC=False):
    """
    Approximates the expected absolute difference in predictions
    between the original instance and its perturbations on the (non) top-k features.

    :param explanations: torch.Tensor of shape (n_inputs, n_features)
    :param inputs: torch.Tensor of shape (n_inputs, n_features)
    :param model: torch.nn.Module
    :param k: int, number of top-k features
    :param perturb_method: PerturbMethod object
    :param feature_metadata: list of letters corresponding to feature types ('c' for continuous, 'd' for discrete)
    :param n_samples: int, number of perturbations
    :param invert: bool, whether to invert the top-k mask (True for PGU)
    :param seed: int, random seed (-1 to set the seed to the instance index)
    :param n_jobs: int, number of parallel jobs, -1 to use all available cores, None to disable parallelism
    :param AUC: bool, whether to compute the AUC of the metric distribution
    :return: float
    """
    # Preprocess
    explanations, inputs = utils.convert_to_tensor(explanations), utils.convert_to_tensor(inputs)
    if inputs.shape != explanations.shape:
        raise ValueError("The input and explanation shapes do not match.")
    n_inputs, n_features = explanations.shape
    max_k = utils.convert_k_to_int(k, n_features)

    # Compute metric
    params = [inputs, explanations, invert, n_jobs, [model, perturb_method, feature_metadata, n_samples, seed]]

    if AUC and max_k > 1:
        metric_distr_all_ks = np.array([_single_k_pred_faith(k, *params) for k in range(1, max_k + 1)])
        metric_distr = np.array([auc(np.arange(max_k)/(max_k-1), metric_distr_all_ks[:, i]) for i in range(n_inputs)])
    else:
        metric_distr = _single_k_pred_faith(max_k, *params)
    
    return metric_distr, np.mean(metric_distr)

# RIS, RRS, ROS
def eval_relative_stability(explainer, inputs, model, perturb_method, feature_metadata, metric,
                            p_norm=2, n_samples = 1000, n_perturbations = 100, seed=-1, n_jobs=None):
    """
    Compute the relative stability of the explanations with respect to the input representations
    :param explainer: Explainer object instance
    :param inputs: np.array of shape (n_inputs, n_features)
    :param model: instance of openxai ann/lr model
    :param perturb_method: instance perturb method object e.g. NormalPerturbation
    :param feature_metadata: list of letters corresponding to feature types ('c' for continuous, 'd' for discrete)
    :param metric: str, 'RIS', 'RRS', or 'ROS'
    :param p_norm: int, float
    :param n_samples: int, number of perturbations
    :param n_perturbations: int, number of perturbations
    :param seed: int, random seed (-1 to set the seed to the instance index)
    :param n_jobs: int, number of parallel jobs, -1 to use all available cores, None to disable parallelism
    """
    inputs = utils.convert_to_tensor(inputs)
    metric = 'ROS' if (model.abbrv=='lr') and (metric=='RRS') else metric  # RRS is equivalent to ROS for LR models
    params = [explainer, model, perturb_method, feature_metadata, metric, n_samples, n_perturbations, p_norm, seed]
    if n_jobs is not None:
        with utils.tqdm_joblib(tqdm(desc=f"Computing {metric}", total=len(inputs))) as progress_bar:
            stability_ratios = np.array(Parallel(n_jobs=n_jobs)(
                delayed(_single_idx_stability)(i, input, *params)\
                    for i, input in enumerate(inputs)))
    else:
        stability_ratios = np.array([_single_idx_stability(i, input, *params)\
                                     for i, input in enumerate(tqdm(inputs))])
    if np.isnan(stability_ratios).any():
        num_nans = np.sum(np.isnan(stability_ratios))
        print(f"{num_nans}/{len(stability_ratios)} NaNs in {metric} for {model.name}")
        stability_ratios = stability_ratios[~np.isnan(stability_ratios)]
    return stability_ratios, np.mean(stability_ratios)

# ==== HELPER FUNCTIONS ==== #

def _preprocess_attributions(explanations, ground_truth):
    """Convert to numpy and reshape (if necessary) for evaluation."""
    explanations, ground_truth = utils.convert_to_numpy(explanations), utils.convert_to_numpy(ground_truth)
    explanations = explanations.reshape(1, -1) if len(explanations.shape) == 1 else explanations
    return explanations, ground_truth

# eval_ground_truth_faithfulness
def _construct_topk_dfs(explanations, ground_truths, k, metric):
    """
    Construct dataframes for top-k features of explanations and ground truth.
    Each dataframe contains the top-k feature idxs and, if applicable,
    the ranks and signs of the top-k features (useful for debugging also).

    :param explanations: np.array of shape (n_inputs, n_features)
    :param ground_truths: np.array of shape (n_inputs, n_features,), since we invert the ground truth depending on the prediction we are explaining
    :param k: int, number of top-k features
    :param metric: str, 'overlap', 'rank', 'sign', or 'ranksign'
    :return: topk_idxs_dfs: size 2 list of pd.DataFrame of shape (n_inputs, k)
    """
    if metric not in ['overlap', 'rank', 'sign', 'ranksign']:
        raise NotImplementedError("Please make sure that have chosen one of the following metrics: {overlap, rank, sign, ranksign}.")
    attrs = [explanations, ground_truths]

    # Feature indices of top-k features (descending i.e. most important to least important)
    topk_idxs = [np.argsort(-np.abs(attr), axis=1)[:, :k] for attr in attrs]
    topk_idxs_dfs = [('feat' + pd.DataFrame(topk_idx).applymap(str)) for topk_idx in topk_idxs]
    if 'rank' in metric:  # RA, SRA
        all_feat_ranks = [rankdata(-np.abs(attr), method='dense', axis=1) for attr in attrs] # rankdata method used to support ties within topk
        topk_ranks = [np.take_along_axis(all_feat_rank, topk_idx, axis=1) for all_feat_rank, topk_idx in zip(all_feat_ranks, topk_idxs)]
        topk_idxs_dfs = [topk_idxs_df + ('rank' + pd.DataFrame(topk_rank).applymap(str)) for topk_idxs_df, topk_rank in zip(topk_idxs_dfs, topk_ranks)]
    if 'sign' in metric:  # SA, SRA
        topk_signs = [np.take_along_axis(np.sign(attr).astype(int), topk_idx, axis=1) for attr, topk_idx in zip(attrs, topk_idxs)]
        topk_idxs_dfs = [topk_idxs_df + ('sign' + pd.DataFrame(topk_sign).applymap(str)) for topk_idxs_df, topk_sign in zip(topk_idxs_dfs, topk_signs)]
    return topk_idxs_dfs

# eval_pred_faithfulness
def _get_perturbation_explanations(model, input, explainer,
                                   perturb_method, n_samples,
                                   feature_metadata, n_perturbations):
    y_pred = torch.argmax(model(input.float()), dim=1)[0]
    explanation = explainer.get_explanations(input.float(), label=y_pred)
    # Get perturbations of the input that have the same prediction
    x_prime_samples = perturb_method.get_perturbed_inputs(original_sample=input[0].float(),
                                                          feature_mask=torch.zeros(input.shape[-1], dtype=bool),
                                                          num_samples=n_samples,
                                                          feature_metadata=feature_metadata)
    y_prime_preds = torch.argmax(model(x_prime_samples.float()), dim=1)
    ind_same_class = (y_prime_preds == y_pred).nonzero()[: n_perturbations].squeeze()
    x_prime_samples = torch.index_select(input=x_prime_samples, dim=0, index=ind_same_class)
    y_prime_samples = torch.index_select(input=y_prime_preds, dim=0, index=ind_same_class)

    # For each perturbation, calculate the explanation
    exp_prime_samples = utils.convert_to_numpy(explainer.get_explanations(x_prime_samples, label=y_prime_samples))
    return x_prime_samples, exp_prime_samples, explanation

# eval_ground_truth_faithfulness
def _single_k_gt_faith(explanations, ground_truths, k, metric):
    """
    Evaluate ground truth faithfulness for a single k value and metric.
    :param explanations: np.array of shape (n_inputs, n_features)
    :param ground_truths: np.array of shape (n_inputs, n_features)
    :param k: int, number of top-k features
    :param metric: str, 'overlap', 'rank', 'sign', or 'ranksign'
    :return: metric_distr: np.array of shape (n_inputs,)
    """
    topk_idxs_df, topk_idxs_df_gt = _construct_topk_dfs(explanations, ground_truths, k, metric)
    if metric in ['overlap', 'sign']:  # FA, SA
        topk_sets = [set(list(row)) for row in topk_idxs_df.to_numpy()]
        topk_sets_gt = [set(list(row)) for row in topk_idxs_df_gt.to_numpy()]
        metric_distr = np.array([len(topk_set.intersection(topk_set_gt))/k\
                                 for topk_set, topk_set_gt in zip(topk_sets, topk_sets_gt)])
    elif metric in ['rank', 'ranksign']:  # RA, SRA
        metric_distr = (topk_idxs_df.to_numpy() == topk_idxs_df_gt.to_numpy()).sum(axis=1)/k
    return metric_distr

# eval_pred_faithfulness
def _single_k_pred_faith(k, inputs, explanations, invert, n_jobs, params):
    if n_jobs is not None:
        with utils.tqdm_joblib(tqdm(desc=f"Computing {'PGU' if invert else 'PGI'}", total=len(inputs))) as progress_bar:
            metric_distr = Parallel(n_jobs=n_jobs)(
                delayed(_single_idx_pred_faith)(i, input, explanation, k, invert, *params)\
                    for i, (input, explanation) in enumerate(zip(inputs, explanations)))
    else:
        metric_distr = np.array([_single_idx_pred_faith(i, input, explanation, k, invert, *params)\
                                 for i, (input, explanation) in enumerate(tqdm(zip(inputs, explanations)))])
    return metric_distr

# eval_pred_faithfulness
def _single_idx_pred_faith(i, input, explanation, k, invert, model, perturb_method,
                           feature_metadata, n_samples, seed):
    """
    Inner loop computation extracted from eval_pred_faithfulness to enable parallelization.
    """
    top_k_mask = utils.generate_mask(explanation, k)
    top_k_mask = torch.logical_not(top_k_mask) if invert else top_k_mask

    # get perturbations of instance x
    torch.manual_seed(i if seed==-1 else seed); np.random.seed(i if seed==-1 else seed)
    x_perturbed = perturb_method.get_perturbed_inputs(original_sample=input,
                                                      feature_mask=top_k_mask,
                                                      num_samples=n_samples,
                                                      feature_metadata=feature_metadata)
    
    # Average the expected absolute difference.
    y = utils.convert_to_numpy(model(input.reshape(1, -1).float()))
    y_perturbed = utils.convert_to_numpy(model(x_perturbed.float()))
    return np.mean(np.abs(y_perturbed - y)[:, 0])

# eval_relative_stability
def _single_idx_stability(i, input, explainer, model, perturb_method, feature_metadata,
                      metric, n_samples, n_perturbations, p_norm, seed):
    """
    Inner loop computation extracted from eval_relative_stability to enable parallelization.
    """
    input = input.reshape(1, -1)
    torch.manual_seed(i if seed == -1 else seed); np.random.seed(i if seed == -1 else seed)
    x_prime_samples, exp_prime_samples, explanation = \
        _get_perturbation_explanations(model, input, explainer, perturb_method,
                                       n_samples, feature_metadata, n_perturbations)
    max_measure, valid_perturb = 0, False
    for x_prime, exp_prime in zip(x_prime_samples, exp_prime_samples):
        x_prime = x_prime.float().reshape(1, -1)
        if metric == 'RIS':
            input_repr, pert_repr = input, x_prime
        elif metric == 'RRS':
            input_repr = model.predict_layer(input, hidden_layer_idx=0, post_act=True)
            pert_repr = model.predict_layer(x_prime, hidden_layer_idx=0, post_act=True)
        elif metric == 'ROS':
            input_repr = model.predict_with_logits(input)
            pert_repr = model.predict_with_logits(x_prime)
        else:
            raise ValueError(f"The metric {metric} is not implemented in the current OpenXAI version.")
        repr_diff = utils.compute_Lp_norm_diff(input_repr, pert_repr, p_norm, normalize_to_relative_change=True)
        exp_diff = utils.compute_Lp_norm_diff(explanation, exp_prime, p_norm, normalize_to_relative_change=True)
        if np.isnan(repr_diff) or np.isnan(exp_diff) or repr_diff == 0:
            continue  # if stability measure is undefined, skip
        else:
            valid_perturb = True  # at least one perturbation with defined stability, else return np.nan
            stability_measure = np.divide(exp_diff, repr_diff)
            if stability_measure > max_measure:
                max_measure = stability_measure
    return max_measure if valid_perturb else np.nan
