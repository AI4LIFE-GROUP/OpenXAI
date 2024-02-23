from scipy.stats import pearsonr, rankdata
import itertools
from scipy.special import comb
import pandas as pd
import numpy as np
import torch

def _convert_to_numpy(exp):
    """Converts input to numpy array."""
    if isinstance(exp, torch.Tensor):
        exp = exp.detach().cpu().numpy()  # in case of GPU
    elif not isinstance(exp, np.ndarray):
        exp = np.array(exp)
    return exp

def _convert_to_tensor(exp):
    """Converts input to torch tensor."""
    if not isinstance(exp, torch.Tensor):
        exp = torch.Tensor(exp)
    return exp

def _preprocess_attributions(explanations, ground_truth):
    """Preprocess explanations and ground truth for evaluation."""
    explanations, ground_truth = _convert_to_numpy(explanations), _convert_to_numpy(ground_truth)
    explanations = explanations.reshape(1, -1) if len(explanations.shape) == 1 else explanations
    return explanations, ground_truth

def _construct_topk_dfs(explanations, ground_truth, k, metric):
    """
    Construct dataframes for top-k features of explanations and ground truth.
    Each dataframe contains the top-k feature idxs and, if applicable,
    the ranks and signs of the top-k features (useful for debugging also).

    :param explanations: np.array of shape (n_samples, n_features)
    :param ground_truth: np.array of shape (n_features,)
    :param k: int, number of top-k features
    :param metric: str, 'overlap', 'rank', 'sign', or 'ranksign'
    :return: topk_idxs_df: pd.DataFrame
             topk_idxs_df_gt: pd.DataFrame
    """
    if metric not in ['overlap', 'rank', 'sign', 'ranksign']:
        raise NotImplementedError("Please make sure that have chosen one of the following metrics: {overlap, rank, sign, ranksign}.")
    ground_truth = ground_truth.reshape(1, -1)
    attrs = [explanations, ground_truth]

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

def _generate_mask(explanation, top_k):
    if not isinstance(explanation, torch.Tensor):
        explanation = torch.Tensor(explanation)
    mask_indices = torch.topk(explanation.abs(), top_k).indices
    mask = torch.ones(explanation.shape, dtype=bool)
    for i in mask_indices:
        mask[i] = False
    return mask

def _compute_Lp_norm_diff(vec1, vec2, p_norm, eps = 0.001,
                          normalize_to_relative_change = True):
    """ Returns the Lp norm of the difference between vec1 and vec2.
    Args:
        normalize_by_vec1: when true, normalizes the difference between vec1 and vec2 by vec1
    """

    # arrays can be flattened, so long as ordering is preserved
    flat_diff = _convert_to_numpy(vec1).flatten() - _convert_to_numpy(vec2).flatten()

    if normalize_to_relative_change:
        vec1_arr = _convert_to_numpy(vec1.flatten())
        vec1_arr = np.clip(vec1_arr, eps, None)
        flat_diff = np.divide(flat_diff, vec1_arr, where=vec1_arr != 0)

    return np.linalg.norm(flat_diff, ord=p_norm)

def pairwise_comp(explanations, ground_truth):
    """
    Compute rank agreement between all possible pairs
    of features in explanations and ground truth.

    :param explanations: np.array of shape (n_samples, n_features)
    :param ground_truth: np.array of shape (n_features,)
    :return: pairwise_distr: np.array of shape (n_samples,)
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

def rankcorr(explanations, ground_truth):
    """
    Compute rank correlation between explanations and ground truth.

    :param explanations: np.array of shape (n_samples, n_features)
    :param ground_truth: np.array of shape (n_features,)
    :return: corrs_distr: np.array of shape (n_samples,)
             mean_corrs: float
    """
    explanations, ground_truth = _preprocess_attributions(explanations, ground_truth)
    corrs_distr = np.zeros(explanations.shape[0])
    exp_ranks = rankdata(-np.abs(explanations), method='dense', axis=1)
    gt_rank = rankdata(-np.abs(ground_truth), method='dense')
    for row in range(exp_ranks.shape[0]):
        corrs_distr[row], _ = pearsonr(exp_ranks[row], gt_rank)
    return corrs_distr, np.mean(corrs_distr)

def eval_ground_truth_faithfulness(explanations, ground_truth, k, metric):
    """
    Compute agreement fraction between top-k features of two explanations.

    :param explanations: np.array of shape (n_samples, n_features)
    :param ground_truth: np.array of shape (n_features,)
    :param k: int, number of top-k features
    :param metric: str, 'overlap', 'rank', 'sign', or 'ranksign'
    :return: metric_distr: np.array of shape (n_samples,)
             mean_metric: float
    """
    explanations, ground_truth = _preprocess_attributions(explanations, ground_truth)
    topk_idxs_df, topk_idxs_df_gt = _construct_topk_dfs(explanations, ground_truth, k, metric)
    if metric in ['overlap', 'sign']:  # FA, SA
        topk_sets = [set(list(row)) for row in topk_idxs_df.to_numpy()]
        topk_sets_gt = set(list(topk_idxs_df_gt.to_numpy()[0]))
        metric_distr = np.array([len(topk_set.intersection(topk_sets_gt))/k for topk_set in topk_sets])
    elif metric in ['rank', 'ranksign']:  # RA, SRA
        metric_distr = (topk_idxs_df.to_numpy() == topk_idxs_df_gt.to_numpy()).sum(axis=1)/k
    return metric_distr, np.mean(metric_distr)

def eval_pred_faithfulness(explanations, inputs, model, k, perturb_method,
                           perturb_max_distance, feature_metadata,
                           num_samples = 100, invert = False):
    """
    Approximates the expected absolute difference in predictions
    between the original instance and its perturbations on the (non) top-k features.

    :param model: torch.nn.Module
    :param input: torch.Tensor of shape (n_features,)
    :param explanation: torch.Tensor of shape (n_features,)
    :param top_k: int, number of top-k features
    :param perturb_method: PerturbMethod object
    :param perturb_max_distance: float, maximum distance for perturbations
    :param feature_metadata: list of letters corresponding to feature types ('c' for continuous, 'd' for discrete)
    :param num_samples: int, number of perturbations
    :param invert: bool, whether to invert the top-k mask (True for PGU)
    :return: float
    """
    # Preprocess inputs and explanations
    explanations, inputs = _convert_to_tensor(explanations), _convert_to_tensor(inputs)
    if inputs.shape != explanations.shape:
        raise ValueError("The input and explanation shapes do not match.")
    
    # Compute the expected absolute difference in predictions
    metric_distr = np.zeros(explanations.shape[0])
    for i, (input, explanation) in enumerate(zip(inputs, explanations)):
        top_k_mask = _generate_mask(explanation, k)
        top_k_mask = torch.logical_not(top_k_mask) if invert else top_k_mask
        
        # get perturbations of instance x
        x_perturbed = perturb_method.get_perturbed_inputs(original_sample=input,
                                                          feature_mask=top_k_mask,
                                                          num_samples=num_samples,
                                                          max_distance=perturb_max_distance,
                                                          feature_metadata=feature_metadata)

        # Average the expected absolute difference.
        y = _convert_to_numpy(model(input.reshape(1, -1).float()))
        y_perturbed = _convert_to_numpy(model(x_perturbed.float()))
        metric_distr[i] = np.mean(np.abs(y_perturbed - y)[:, 0])
    return metric_distr, np.mean(metric_distr)

def eval_relative_stability(explainer, inputs, k, y_pred, perturb_method, perturb_max_distance,
                            L_map, feature_metadata, model, p_norm,
                            x_prime_samples=None,
                            exp_prime_samples=None,
                            rep_denominator_flag: bool = False,
                            num_samples: int = 1000,
                            num_perturbations: int = 50):
    """Approximates the maximum L-p distance between explanations in a neighborhood around the input.

    Args:
        rep_denominator_flag: when true, normalizes the stability metric by the L-p distance
            between representations (instead of features).
    """
    stability_ratios = []
    rep_diffs = []
    x_diffs = []
    exp_diffs = []

    for input in inputs:
        # Perturb input
        x_prime_samples = perturb_method.get_perturbed_inputs(original_sample=input,
                                                              feature_mask=top_k_mask,
                                                              num_samples=num_samples,
                                                              max_distance=perturb_max_distance,
                                                              feature_metadata=feature_metadata)

        # Take the first num_perturbations points that have the same predicted class label
        y_prime_preds = torch.argmax(model(x_prime_samples.float()), dim=1)

        ind_same_class = (y_prime_preds == y_pred).nonzero()[: num_perturbations].squeeze()
        x_prime_samples = torch.index_select(input=x_prime_samples, dim=0, index=ind_same_class)
        y_prime_preds = torch.argmax(model(x_prime_samples.float()), dim=1)

        # For each perturbation, calculate the explanation
        exp_prime_samples = torch.zeros_like(x_prime_samples)
        for it, x_prime in enumerate(x_prime_samples):
            x_prime = x_prime.reshape(1, -1)
            lab = y_prime_preds[it].type(torch.int64)
            exp = explainer.get_explanation(x_prime.float(), label=lab)
            exp_prime_samples[it, :] = exp
        
    for sample_ind, x_prime in enumerate(x_prime_samples):
        x_prime = x_prime.unsqueeze(0)

        rep_diff = _compute_Lp_norm_diff(L_map(input.float().reshape(1, -1)), L_map(x_prime.float().reshape(1, -1)),
                                         p_norm, normalize_to_relative_change=True)
        rep_diffs.append(rep_diff)
        exp_at_perturbation = exp_prime_samples[sample_ind]
        # for predictions per perturbation
        explanation_diff = _compute_Lp_norm_diff(explanation, exp_at_perturbation, p_norm,
                                                 normalize_to_relative_change=True)
        exp_diffs.append(explanation_diff)
        if rep_denominator_flag:
            # compute norm between representations
            stability_measure = np.divide(explanation_diff, rep_diff)
        else:
            feature_difference = _compute_Lp_norm_diff(input, x_prime, p_norm)
            stability_measure = np.divide(explanation_diff, feature_difference)
            x_diffs.append(_compute_Lp_norm_diff(input, x_prime, p_norm))
        stability_ratios.append(stability_measure)
            
    ind_max = np.argmax(stability_ratios)
    return stability_ratios[ind_max]  # , stability_ratios, rep_diffs, x_diffs, exp_diffs, ind_max
