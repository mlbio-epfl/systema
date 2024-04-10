import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr
from data import get_pert_data
import argparse

# GEARS installation
# ! pip install torch-geometric
# ! pip install cell-gears

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Norman2019')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--outdir', default='results')

args = parser.parse_args()

if __name__ == '__main__':
    pert_data = get_pert_data(dataset=args.dataset,
                              seed=args.seed)

    # Split train and test
    test_adata = pert_data.adata[pert_data.adata.obs['split'] == 'test']
    train_adata = pert_data.adata[pert_data.adata.obs['split'] == 'train']

    # Get control mean, non control mean (pert_mean), and non control mean differential
    control_adata = train_adata[train_adata.obs['control'] == 1]
    pert_adata = train_adata[train_adata.obs['control'] == 0]
    control_mean = np.array(control_adata.X.mean(axis=0))[0]
    pert_mean = np.array(pert_adata.X.mean(axis=0))[0]
    delta_pert = pert_mean - control_mean

    # Store results
    results_df = pd.DataFrame(columns=['method', 'pert', 'corr_all', 'corr_20de', 'one gene', 'train'])
    unique_conds = set(test_adata.obs['condition'].unique()) - set(['ctrl'])
    for condition in tqdm(unique_conds):
        gene_list = condition.split('+')
        one_gene = False
        if 'ctrl' in gene_list:
            gene_list.remove('ctrl')
            one_gene = True
        one_gene_str = '1-gene' if one_gene else '2-gene'

        # Select adata condition
        adata_condition = test_adata[test_adata.obs['condition'] == condition]
        X_post = np.array(adata_condition.X.mean(axis=0))[
            0]  # adata_condition.X.mean(axis=0) is a np.matrix of shape (1, n_genes)
        delta_true = X_post - control_mean

        # Select top 20 DE genes
        top20_de_genes = pert_data.adata.uns['top_non_dropout_de_20'][adata_condition.obs['condition_name'].values[0]]
        top20_de_idxs = np.argwhere(np.isin(pert_data.adata.var.index, top20_de_genes)).ravel()

        # Matching mean. Get matching differentials from train data
        diffs = []
        n_train = 0
        for g in gene_list:
            if f'{g}+ctrl' in train_adata.obs['condition'].values:
                assert not one_gene
                cond_adata = train_adata[train_adata.obs['condition'] == f'{g}+ctrl']
                n_train += 1
            elif f'ctrl+{g}' in train_adata.obs['condition'].values:
                assert not one_gene
                cond_adata = train_adata[train_adata.obs['condition'] == f'ctrl+{g}']
                n_train += 1
            else:
                cond_adata = pert_adata
            X_pert = np.array(cond_adata.X.mean(axis=0))[0]
            diffs.append(X_pert - control_mean)
        match_mean = np.mean(diffs, axis=0)
        results_df.loc[len(results_df)] = ['matching mean',
                                           condition,
                                           pearsonr(delta_true, match_mean)[0],
                                           pearsonr(delta_true[top20_de_idxs], match_mean[top20_de_idxs])[0],
                                           one_gene_str,
                                           n_train]

    results_df.to_csv(f'{args.outdir}/{args.dataset}_{args.seed}_matching-mean_results.csv', index=False)