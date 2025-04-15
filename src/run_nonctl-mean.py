import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr
from data import get_pert_data
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Norman2019')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument("--data_dir", default="data")
parser.add_argument('--outdir', default='results')

args = parser.parse_args()

if __name__ == '__main__':
    pert_data = get_pert_data(dataset=args.dataset,
                              seed=args.seed,
                              data_dir=args.data_dir)

    # Split train and test
    test_adata = pert_data.adata[pert_data.adata.obs['split'] == 'test']
    train_adata = pert_data.adata[pert_data.adata.obs['split'] == 'train']

    # Get control mean, non control mean (pert_mean), and non control mean differential
    control_adata = train_adata[train_adata.obs['control'] == 1]
    pert_adata = train_adata[train_adata.obs['control'] == 0]
    control_mean = np.array(control_adata.X.mean(axis=0))[0]
    pert_mean = np.array(pert_adata.X.mean(axis=0))[0]

    # Store results
    unique_conds = list(set(test_adata.obs['condition'].unique()) - set(['ctrl']))
    post_gt_df = pd.DataFrame(columns=pert_data.adata.var['gene_name'].values)
    post_pred_df = pd.DataFrame(columns=pert_data.adata.var['gene_name'].values)
    train_counts = []
    for condition in tqdm(unique_conds):
        gene_list = condition.split('+')

        # Select adata condition
        adata_condition = test_adata[test_adata.obs['condition'] == condition]
        X_post = np.array(adata_condition.X.mean(axis=0))[0]  # adata_condition.X.mean(axis=0) is a np.matrix of
        # shape (1, n_genes)

        # Store number of train perturbations
        n_train = 0
        for g in gene_list:
            if f'{g}+ctrl' in train_adata.obs['condition'].values:
                n_train += 1
            elif f'ctrl+{g}' in train_adata.obs['condition'].values:
                n_train += 1
        train_counts.append(n_train)

        # Non-ctl mean
        post_gt_df.loc[len(post_gt_df)] = X_post
        post_pred_df.loc[len(post_pred_df)] = pert_mean

    index = pd.MultiIndex.from_tuples(list(zip(unique_conds, train_counts)), names=['condition', 'n_train'])
    post_gt_df.index = index
    post_pred_df.index = index
    
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    post_gt_df.to_csv(f'{args.outdir}/{args.dataset}_{args.seed}_nonctl-mean_post-gt.csv')
    post_pred_df.to_csv(f'{args.outdir}/{args.dataset}_{args.seed}_nonctl-mean_post-pred.csv')