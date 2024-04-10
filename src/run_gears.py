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

if __name__ == '__main__':
    pert_data = get_pert_data(dataset=args.dataset,
                              seed=args.seed)

    # Ref: https://github.com/snap-stanford/GEARS/blob/719328bd56745ab5f38c80dfca55cfd466ee356f/demo/model_tutorial.ipynb
    pert_data.get_dataloader(batch_size=args.batchsize,
                             test_batch_size=args.batchsize)
    gears_model = GEARS(pert_data, device=f'cuda:{args.device}',
                        weight_bias_track=False,
                        proj_name='pertnet',
                        exp_name='pertnet')
    gears_model.model_initialize(hidden_size=args.hiddendim)
    gears_model.train(epochs=args.epochs, lr=args.lr)
    Path(f'{args.outdir}/checkpoints').mkdir(parents=True, exist_ok=True)
    gears_model.save_model(f'{args.outdir}/checkpoints/gears_seed{args.seed}_{args.dataset}')

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

        # Store number of train perturbations
        n_train = 0
        for g in gene_list:
            if f'{g}+ctrl' in train_adata.obs['condition'].values:
                n_train += 1
            elif f'ctrl+{g}' in train_adata.obs['condition'].values:
                n_train += 1

        # Get GEARS predictions
        gears_pred = list(gears_model.predict([gene_list]).values())[0]
        gears_delta = gears_pred - control_mean
        results_df.loc[len(results_df)] = ['GEARS', condition,
                                           pearsonr(delta_true, gears_delta)[0],
                                           pearsonr(delta_true[top20_de_idxs], gears_delta[top20_de_idxs])[0],
                                           one_gene_str,
                                           n_train]

    results_df.to_csv(f'{args.outdir}/{args.dataset}_{args.seed}_gears_results.csv', index=False)