import numpy as np
import pandas as pd
from sklearn.metrics import RocCurveDisplay, roc_curve
import matplotlib.pyplot as plt


def get_perts(test_perts, phenotypes, phenotype_names):
    """
    Given a set of test_perturbations, a dictionary phenotype -> perturbation list, and a list of phenotype names, returns the test perturbations that lead to the indicated phenotypes.

    Arguments:
    * test_perts: List of perturbation names
    * phenotypes: Dictionary phenotype -> perturbation list
    * phenotype_names: Names of selected phenotypes (should be in the dictionary)

    Returns a list of test perturbations that lead to the indicated phenotypes.
    """
    perts = []
    for p in phenotype_names:
        perts.extend(list(phenotypes[p]))
    perts = np.intersect1d(test_perts, perts)
    return perts

def score_centroids(post_gt_df_seed, post_pred_df_seed, perts_dict, methods):
    """
    Calculates scores (i.e. negative of Euclidean distances) for each test perturbation to each phenotype centroid for all methods.

    Arguments:
    * post_gt_df_seed: Pandas dataframe with the ground truth post-perturbation profiles. Rows: perturbations, columns: genes.
    * post_pred_df_seed: Pandas dataframe with the inferred post-perturbation profiles of each method. Columns correspond to genes. Each row corresponds to the predictions of a method for a given test perturbation. Expects a DataFrame with a _MultiIndex_, where the first index is the condition (test perturbation) and the second is the method. For example:
        > post_pred_df_seed.index
        MultiIndex([('ACSL3',         'cpa'),
                ('ACSL3',       'gears'),
                ('ACSL3', 'nonctl-mean'),
                ('ACSL3',       'scgpt'),
                ('ACSL3',    'scgpt_ft'),
                ('AEBP1',         'cpa'),
                ('AEBP1',       'gears'),
                ('AEBP1', 'nonctl-mean'),
                ('AEBP1',       'scgpt'),
                ('AEBP1',    'scgpt_ft'),
                ...
                ('VDAC2',         'cpa'),
                ('VDAC2',       'gears'),
                ('VDAC2', 'nonctl-mean'),
                ('VDAC2',       'scgpt'),
                ('VDAC2',    'scgpt_ft'),
                ( 'WBP2',         'cpa'),
                ( 'WBP2',       'gears'),
                ( 'WBP2', 'nonctl-mean'),
                ( 'WBP2',       'scgpt'),
                ( 'WBP2',    'scgpt_ft')],
               names=['condition', 'method'], length=495)
    * perts_dict: Dictionary of reference centroids (i.e. coarse phenotype) -> list of test perturbations
    * methods: List of methods with predictions in post_pred_df_seed

    Returns:
    * labels: np.array of shape (n_test_perturbations, n_reference_centroids) indicating what phenotype each test perturbation belongs to
    * scores_dict: dictionary method_name -> np.array of shape (n_test_perturbations, n_reference_centroids) with scores for each test perturbation and reference centroid. Higher scores indicates higher likelihoods of perturbation inducing a certain phenotype
    """
    # Calculate prototypes (TODO: Use train perturbations too?)
    prototypes = {}
    for k, v in perts_dict.items():
        prototypes[k] = post_gt_df_seed.loc[v].mean(axis=0)
    
    # Get list of all perturbations that play a role for at least one phenotype
    perts = []
    for k, p in perts_dict.items():
        perts.extend(list(p))
    
    # Display ROC
    labels = np.zeros((len(perts), len(prototypes))) # np.zeros_like(scores)
    scores_dict = {}
    for method in methods:
        preds = post_pred_df_seed.xs(method, level=1).loc[perts]
        scores = np.zeros((len(preds), len(prototypes)))
        for i, (k, v) in enumerate(prototypes.items()):
            distances = ((preds - v) ** 2).mean(axis=1)
            scores[:, i] = -distances
            labels[np.isin(perts, perts_dict[k]), i] = 1
        scores_dict[method] = scores
    return labels, scores_dict

def plot_binary_roc(labels, scores_dict, methods, title="CIN driver ROC"):
    """
    Plots ROC curve given the outputs of score_centroids

    Arguments:
    * labels: np.array of shape (n_test_perturbations, n_reference_centroids) indicating what phenotype each test perturbation belongs to
    * scores_dict: dictionary method_name -> np.array of shape (n_test_perturbations, n_reference_centroids) with scores for each test perturbation and reference centroid. Higher scores indicates higher likelihoods of perturbation inducing a certain phenotype
    """
    ax = plt.gca()
    for method in methods:
        v = scores_dict[method]
        display = RocCurveDisplay.from_predictions(
            labels[:, 1],
            v[:, 1] - v[:, 0],
            name=method,
            ax=ax,
            # color="darkorange",
            plot_chance_level=method==methods[-1],
            # despine=True,
        )
        _ = display.ax_.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=title,
        )
    return plt.gca()