import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist


def calculate_centroid_accuracies(agg_post_pred_df, post_gt_all_df):
    """
    Calculates the centroid accuracies for all given perturbations and methods

    Arguments:
    * agg_post_pred_df: Pandas dataframe with the inferred post-perturbation profiles of each method. Columns correspond to genes. Each row corresponds to the predictions of a method for a given test perturbation. Expects a DataFrame with a _MultiIndex_, where the first index is the condition (test perturbation) and the second is the method. For example:

        > agg_post_pred_df.index
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
    * post_gt_all_df:  Pandas dataframe with the ground truth post-perturbation profiles. Rows: perturbations, columns: genes.

    Returns a Pandas DataFrame with the centroid accuracies of each method for each test perturbation.
    """
    distances = cdist(agg_post_pred_df, post_gt_all_df, metric='euclidean')
    dist_df = pd.DataFrame(distances, index=agg_post_pred_df.index, columns=post_gt_all_df.index)
    
    # Self distances
    index = [g for g in dist_df.index.get_level_values(0) if g in dist_df.columns]
    multiindex = [(g, m) for g, m in dist_df.index if g in dist_df.columns]
    col_idxs = [np.argwhere(dist_df.columns == g)[0][0] for g in index]
    self_distances = np.diag(dist_df.iloc[np.arange(len(dist_df)), col_idxs])
    self_distances_df = pd.DataFrame(self_distances, index=pd.MultiIndex.from_tuples(multiindex))
    assert np.all(dist_df.index == self_distances_df.index)

    scores = {}
    methods = agg_post_pred_df.index.get_level_values(1).unique()
    for method in methods:
        x_df = dist_df.xs(method, level=1).sort_index()
        y_df = self_distances_df.xs(method, level=1).sort_index()
        assert np.all(x_df.index == y_df.index)
        scores[method] = ((x_df > y_df.values).sum(axis=1)) / (x_df.shape[-1] - 1)
    scores_df = pd.DataFrame(scores)
    return scores_df