import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import scanpy as sc
import pandas as pd

def jaccard_similarity(list1, list2):
    """
    Compute the Jaccard similarity between two lists.
    """
    set1 = set(list1)
    set2 = set(list2)
    return len(set1.intersection(set2)) / len(set1.union(set2))

def get_topk_de_gene_ids(ctrl, post, k=20):
    """
    Get the top k differentially expressed genes from the results.
    """
    # Get the top k differentially expressed genes
    diff = post - ctrl
    diff_genes_ids = np.argsort(np.abs(diff))[-k:]
    return diff_genes_ids

def average_of_perturbation_centroids(adata):
    pert_means = []
    pert_adata = adata[adata.obs['control'] == 0]
    for cond in pert_adata.obs['condition'].unique():
        adata_cond = pert_adata[pert_adata.obs['condition'] == cond]
        pert_mean = np.array(adata_cond.X.mean(axis=0))[0]
        pert_means.append(pert_mean)
    pert_means = np.array(pert_means)
    return np.mean(pert_means, axis=0)

def get_perturbation_shifts(adata, reference, top_20=False):
    pert_shifts = []
    conditions = set(adata.obs['condition'].unique()) - set(['ctrl'])
    for condition in conditions:
        adata_condition = adata[adata.obs["condition"] == condition]
        pert_shift = np.array(adata_condition.X.mean(axis=0))[0] - reference
        
        if top_20:
            # Select top 20 DE genes
            top20_de_idxs = adata.uns["top_non_dropout_de_20"][
                adata_condition.obs["condition_name"].values[0]
            ]
            top20_de_idxs = np.argwhere(
                np.isin(adata.var.index, top20_de_genes)
            ).ravel()
            pert_shift = pert_shift[top20_de_idxs]

        # Compute shift
        pert_shifts.append(pert_shift)
    return conditions, np.array(pert_shifts)

def calculate_cosine_similarities(pert_shifts, reference):
    sims = cosine_similarity(pert_shifts, reference[None, :]).ravel()
    return sims

def calculate_pairwise_cosine_similarities(pert_shifts):
    sims = cosine_similarity(np.array(pert_shifts))
    return sims[np.triu_indices(len(sims), k=1)]

def average_of_perturbation_centroids(adata):
    pert_means = []
    pert_adata = adata[adata.obs['control'] == 0]
    for cond in pert_adata.obs['condition'].unique():
        adata_cond = pert_adata[pert_adata.obs['condition'] == cond]
        pert_mean = np.array(adata_cond.X.mean(axis=0))[0]
        pert_means.append(pert_mean)
    pert_means = np.array(pert_means)
    return np.mean(pert_means, axis=0)

def calculate_norms(pert_shifts):
    return np.linalg.norm(pert_shifts, axis=1)

def compute_shift_similarities(adata, avg_pert_centroids=True, control_mean=None):
    pert_adata = adata[adata.obs['control'] == 0]
    if control_mean is None:
        control_adata = adata[adata.obs['control'] == 1]
        control_mean = np.array(control_adata.X.mean(axis=0))[0]
    
    if avg_pert_centroids:
        pert_mean = average_of_perturbation_centroids(pert_adata)
    else:
        pert_mean = np.array(pert_adata.X.mean(axis=0))[0]
    avg_shift = pert_mean-control_mean
    
    pert_shifts = {
        'avg_ctl': get_perturbation_shifts(adata, reference=control_mean),
        'avg_pert': get_perturbation_shifts(adata, reference=pert_mean)
    }

    pert_names = []
    similarities = {}
    pairwise_similarities = {}
    norms = {}
    for k, (perts, v) in pert_shifts.items():
        similarities[k] = calculate_cosine_similarities(v, avg_shift)
        pairwise_similarities[k] = calculate_pairwise_cosine_similarities(v)
        norms[k] = calculate_norms(v)
        pert_names.extend(perts)
    
    df = pd.DataFrame(similarities).melt()
    df_pair = pd.DataFrame(pairwise_similarities).melt()
    df_norm = pd.DataFrame(norms).melt()
    df['pert_names'] = pert_names
    df_norm['pert_names'] = pert_names
    
    return df, df_pair, df_norm, pert_names