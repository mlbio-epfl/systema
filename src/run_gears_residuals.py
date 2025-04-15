import os.path

from gears import PertData, GEARS
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr
from data import get_pert_data
import argparse
from pathlib import Path

# GEARS installation
# ! pip install torch-geometric
# ! pip install cell-gears

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Norman2019')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--outdir', default='results')
parser.add_argument('--device', default=0, type=int)
parser.add_argument('--hiddendim', default=64, type=int)
parser.add_argument('--batchsize', default=32, type=int)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--lr', default=1e-3, type=int)

args = parser.parse_args()

def average_of_perturbation_centroids(adata):
    pert_means = []
    pert_adata = adata[adata.obs['control'] == 0]
    for cond in pert_adata.obs['condition'].unique():
        adata_cond = pert_adata[pert_adata.obs['condition'] == cond]
        pert_mean = np.array(adata_cond.X.mean(axis=0))[0]
        pert_means.append(pert_mean)
    pert_means = np.array(pert_means)
    return np.mean(pert_means, axis=0)

if __name__ == '__main__':
    pert_data = get_pert_data(dataset=args.dataset,
                              seed=args.seed)

    # Ref: https://github.com/snap-stanford/GEARS/blob/719328bd56745ab5f38c80dfca55cfd466ee356f/demo/model_tutorial.ipynb
    pert_data.get_dataloader(batch_size=args.batchsize,
                             test_batch_size=args.batchsize)
    
    # Split train and test
    test_adata = pert_data.adata[pert_data.adata.obs['split'] == 'test'].copy()
    train_adata = pert_data.adata[pert_data.adata.obs['split'] == 'train'].copy()

    # Get control mean, non control mean (pert_mean), and non control mean differential
    control_adata = train_adata[train_adata.obs['control'] == 1]
    pert_adata = train_adata[train_adata.obs['control'] == 0]
    control_mean = np.array(control_adata.X.mean(axis=0))[0]
    pert_mean = np.array(pert_adata.X.mean(axis=0))[0]
    pert_centroids_mean = average_of_perturbation_centroids(pert_adata)
    delta_pert = pert_mean - control_mean

    # Train model to predict residuals
    pert_data.adata.X = pert_data.adata.X - pert_centroids_mean

    # Train/load model
    model_path = f'{args.outdir}/checkpoints/gears-residuals_seed{args.seed}_{args.dataset}'
    gears_model = GEARS(pert_data, device=f'cuda:{args.device}',
                        weight_bias_track=False,
                        proj_name='pertnet',
                        exp_name='pertnet')
    gears_model.model_initialize(hidden_size=args.hiddendim)
    if os.path.exists(model_path):
        print(f'Loading model from {model_path}')
        gears_model.load_pretrained(model_path)
    else:
        gears_model.train(epochs=args.epochs, lr=args.lr)
        Path(f'{args.outdir}/checkpoints').mkdir(parents=True, exist_ok=True)
        gears_model.save_model(model_path)

    # Store results
    unique_conds = list(set(test_adata.obs['condition'].unique()) - set(['ctrl']))
    post_gt_df = pd.DataFrame(columns=pert_data.adata.var['gene_name'].values)
    post_pred_df = pd.DataFrame(columns=pert_data.adata.var['gene_name'].values)
    train_counts = []
    for condition in tqdm(unique_conds):
        gene_list = condition.split('+')
        if 'ctrl' in gene_list:
            gene_list.remove('ctrl')

        # Select adata condition
        adata_condition = test_adata[test_adata.obs['condition'] == condition]
        X_post = np.array(adata_condition.X.mean(axis=0))[
            0]  # adata_condition.X.mean(axis=0) is a np.matrix of shape (1, n_genes)

        # Store number of train perturbations
        n_train = 0
        for g in gene_list:
            if f'{g}+ctrl' in train_adata.obs['condition'].values:
                n_train += 1
            elif f'ctrl+{g}' in train_adata.obs['condition'].values:
                n_train += 1
        train_counts.append(n_train)

        # Get GEARS predictions
        gears_pred = list(gears_model.predict([gene_list]).values())[0] + pert_centroids_mean
        post_gt_df.loc[len(post_gt_df)] = X_post
        post_pred_df.loc[len(post_pred_df)] = gears_pred

    index = pd.MultiIndex.from_tuples(list(zip(unique_conds, train_counts)), names=['condition', 'n_train'])
    post_gt_df.index = index
    post_pred_df.index = index
    post_gt_df.to_csv(f'{args.outdir}/{args.dataset}_{args.seed}_gears-residuals_post-gt.csv')
    post_pred_df.to_csv(f'{args.outdir}/{args.dataset}_{args.seed}_gears-residuals_post-pred.csv')