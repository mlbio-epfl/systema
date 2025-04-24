from scipy.stats import pearsonr

def pearson_delta_reference_metrics(X_true, X_pred, reference):
    """
    Compute PearsonΔ and PearsonΔ20 metrics using a specific reference

    Arguments:
    * X_true: ground-truth post-perturbation profile. Shape: (n_genes,)
    * X_pred: predicted post-perturbation profile. Shape: (n_genes,)
    * reference: reference. Shape: (n_genes,)

    Returns a dictionary with 2 metrics: corr_all_allpert (PearsonΔ) and corr_20de_allpert (PearsonΔ20)
    """
    delta_true_allpert = X_true - reference
    delta_pred_allpert = X_pred - reference

    out = {
        'corr_all_allpert': pearsonr(delta_true_allpert, delta_pred_allpert)[0],
        'corr_20de_allpert': pearsonr(delta_true_allpert[top20_de_idxs], delta_pred_allpert[top20_de_idxs])[0],
    }
    return out