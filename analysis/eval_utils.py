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

def compute_shift_similarities(adata, avg_pert_centroids=True):
    pert_adata = adata[adata.obs['control'] == 0]
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


#### Differential expression analysis
def rank_genes_groups_allpert(adata_pert, groupby='condition_name', reference='rest'):
    # Following GEARS rank_genes_groups: https://github.com/snap-stanford/GEARS/blob/e0c27b69f8a1a611d56c3f8c5f5a168cb2cde5f6/gears/data_utils.py#L15
    
    adata_pert.X = adata_pert.X.toarray()
    
    #compute DEGs
    sc.tl.rank_genes_groups(
        adata_pert,
        groupby='condition_name',
        reference='rest',
        rankby_abs=True,
        n_genes=None,
        use_raw=False
    )
    
    #add entries to dictionary of gene sets
    de_genes = pd.DataFrame(adata_pert.uns['rank_genes_groups']['names'])
    gene_dict = {}
    for group in de_genes:
        gene_dict[group] = de_genes[group].tolist()
    adata_pert.uns['rank_genes_groups_cov_allpert'] = gene_dict
    
    return gene_dict


def get_dropout_non_zero_genes_allpert(adata, uns_key = 'rank_genes_groups_cov_allpert'):
    # calculate mean expression for each condition
    unique_conditions = adata.obs.condition.unique()
    conditions2index = {}
    for i in unique_conditions:
        conditions2index[i] = np.where(adata.obs.condition == i)[0]

    condition2mean_expression = {}
    for i, j in conditions2index.items():
        condition2mean_expression[i] = np.mean(adata.X[j], axis = 0)
    pert_list = np.array(list(condition2mean_expression.keys()))
    mean_expression = np.array(list(condition2mean_expression.values())).reshape(len(adata.obs.condition.unique()), adata.X.toarray().shape[1])
    ctrl = mean_expression[np.where(pert_list == 'ctrl')[0]]
    
    ## in silico modeling and upperbounding
    pert2pert_full_id = dict(adata.obs[['condition', 'condition_name']].values)
    pert_full_id2pert = dict(adata.obs[['condition_name', 'condition']].values)

    gene_id2idx = dict(zip(adata.var.index.values, range(len(adata.var))))
    gene_idx2id = dict(zip(range(len(adata.var)), adata.var.index.values))

    non_zeros_gene_idx = {}
    top_non_dropout_de_20 = {}
    top_non_zero_de_20 = {}
    non_dropout_gene_idx = {}

    for pert in adata.uns[uns_key].keys():
        p = pert_full_id2pert[pert]
        X = np.mean(adata[adata.obs.condition == p].X, axis = 0)

        non_zero = np.where(np.array(X)[0] != 0)[0]
        zero = np.where(np.array(X)[0] == 0)[0]
        true_zeros = np.intersect1d(zero, np.where(np.array(ctrl)[0] == 0)[0])
        non_dropouts = np.concatenate((non_zero, true_zeros))

        top = adata.uns['rank_genes_groups_cov_all'][pert]
        gene_idx_top = [gene_id2idx[i] for i in top]

        non_dropout_20 = [i for i in gene_idx_top if i in non_dropouts][:20]
        non_dropout_20_gene_id = [gene_idx2id[i] for i in non_dropout_20]

        non_zero_20 = [i for i in gene_idx_top if i in non_zero][:20]
        non_zero_20_gene_id = [gene_idx2id[i] for i in non_zero_20]

        non_zeros_gene_idx[pert] = np.sort(non_zero)
        non_dropout_gene_idx[pert] = np.sort(non_dropouts)
        top_non_dropout_de_20[pert] = np.array(non_dropout_20_gene_id)
        top_non_zero_de_20[pert] = np.array(non_zero_20_gene_id)
        
    non_zero = np.where(np.array(X)[0] != 0)[0]
    zero = np.where(np.array(X)[0] == 0)[0]
    true_zeros = np.intersect1d(zero, np.where(np.array(ctrl)[0] == 0)[0])
    non_dropouts = np.concatenate((non_zero, true_zeros))
    
    adata.uns['top_non_dropout_de_20_allpert'] = top_non_dropout_de_20
    adata.uns['non_dropout_gene_idx_allpert'] = non_dropout_gene_idx
    adata.uns['non_zeros_gene_idx_allpert'] = non_zeros_gene_idx
    adata.uns['top_non_zero_de_20_allpert'] = top_non_zero_de_20
    
    return adata