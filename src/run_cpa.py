import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr
from data import get_pert_data
import argparse
import cpa
from pathlib import Path
import os

# CPA installation
# !pip install cpa-tools

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Norman2019')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument("--data_dir", default="data")
parser.add_argument('--outdir', default='results')
parser.add_argument('--device', default=1, type=int)
parser.add_argument('--hiddendim', default=64, type=int)
parser.add_argument('--batchsize', default=32, type=int)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--lr', default=1e-3, type=int)
parser.add_argument("--load_model", action="store_true")
parser.set_defaults(load_model=False)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

model_params = {'n_latent': args.hiddendim,
                'recon_loss': 'gauss',
                'doser_type': 'linear',
                'n_hidden_encoder': 256,
                'n_layers_encoder': 4,
                'n_hidden_decoder': 256,
                'n_layers_decoder': 2,
                'use_batch_norm_encoder': True,
                'use_layer_norm_encoder': False,
                'use_batch_norm_decoder': False,
                'use_layer_norm_decoder': False,
                'dropout_rate_encoder': 0.2,
                'dropout_rate_decoder': 0.0,
                'variational': False,
                'seed': args.seed}

trainer_params = {'n_epochs_kl_warmup': None,
                  'n_epochs_pretrain_ae': 50,
                  'n_epochs_adv_warmup': 10,
                  'n_epochs_mixup_warmup': 10,
                  'mixup_alpha': 0.1,
                  'adv_steps': 3,
                  'n_hidden_adv': 128,
                  'n_layers_adv': 3,
                  'use_batch_norm_adv': True,
                  'use_layer_norm_adv': False,
                  'dropout_rate_adv': 0.3,
                  'reg_adv': 10.0,
                  'pen_adv': 20.0,
                  'lr': args.lr,
                  'wd': 4e-07,
                  'adv_lr': 0.0003,
                  'adv_wd': 4e-07,
                  'adv_loss': 'cce',
                  'doser_lr': 0.001,
                  'doser_wd': 4e-07,
                  'do_clip_grad': False,
                  'gradient_clip_value': 5.0,
                  'step_size_lr': 25}

if __name__ == '__main__':
    pert_data = get_pert_data(dataset=args.dataset,
                              seed=args.seed,
                              data_dir=args.data_dir)

    # Ref: https://cpa-tools.readthedocs.io/en/latest/tutorials/Norman.html
    cpa_adata = pert_data.adata.copy()
    cpa.CPA.setup_anndata(cpa_adata,
                          perturbation_key='condition',
                          control_group='ctrl',
                          dosage_key='dose_val',
                          categorical_covariate_keys=['cell_type'],
                          is_count_data=False,
                          deg_uns_key='rank_genes_groups_cov_all',
                          deg_uns_cat_key='condition_name',
                          max_comb_len=2,
                          )
    cpa_model = cpa.CPA(adata=cpa_adata,
                        split_key='split',
                        train_split='train',
                        valid_split='val',
                        test_split='test',
                        **model_params,
                        )
    Path(f'{args.outdir}/checkpoints').mkdir(parents=True, exist_ok=True)
    model_path = f'{args.outdir}/checkpoints/cpa_seed{args.seed}_{args.dataset}'
    if not args.load_model or not os.path.exists(model_path):
        cpa_model.train(max_epochs=args.epochs,
                        use_gpu=True,
                        batch_size=args.batchsize,
                        plan_kwargs=trainer_params,
                        early_stopping_patience=10,
                        # check_val_every_n_epoch=5,
                        check_val_every_n_epoch=0,
                        limit_val_batches=0,
                        save_path=model_path)

    # Load best CPA model
    print(f'Loading model from {model_path}')
    cpa_model = cpa.CPA.load(dir_path=model_path,
                             adata=cpa_adata,
                             use_gpu=True)

    # Split train and test
    test_adata = pert_data.adata[pert_data.adata.obs['split'] == 'test']
    train_adata = pert_data.adata[pert_data.adata.obs['split'] == 'train']

    # CPA adata
    cpa_adata_test = cpa_adata[cpa_adata.obs['split'] == 'test']
    cpa_control_adata = cpa_adata[cpa_adata.obs['control'] == 1]

    # Get control mean, non control mean (pert_mean), and non control mean differential
    control_adata = train_adata[train_adata.obs['control'] == 1]
    pert_adata = train_adata[train_adata.obs['control'] == 0]
    control_mean = np.array(control_adata.X.mean(axis=0))[0]
    pert_mean = np.array(pert_adata.X.mean(axis=0))[0]
    delta_pert = pert_mean - control_mean

    # Store results
    unique_conds = list(set(test_adata.obs['condition'].unique()) - set(['ctrl']))
    post_gt_df = pd.DataFrame(columns=pert_data.adata.var['gene_name'].values)
    post_pred_df = pd.DataFrame(columns=pert_data.adata.var['gene_name'].values)
    train_counts = []
    for condition in tqdm(unique_conds):
        gene_list = condition.split('+')

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

        # Get CPA predictions, sampling random control cells
        cpa_adata_condition = cpa_adata[cpa_adata.obs['condition'] == condition].copy()
        idxs_control = np.random.choice(len(cpa_control_adata), len(cpa_adata_condition), replace=True)
        cpa_adata_condition.X = cpa_control_adata[idxs_control].X.toarray()
        cpa_model.predict(cpa_adata_condition, batch_size=args.batchsize)
        cpa_pred = np.mean(cpa_adata_condition.obsm['CPA_pred'], axis=0)
        del cpa_adata_condition
        post_gt_df.loc[len(post_gt_df)] = X_post
        post_pred_df.loc[len(post_pred_df)] = cpa_pred

    index = pd.MultiIndex.from_tuples(list(zip(unique_conds, train_counts)), names=['condition', 'n_train'])
    post_gt_df.index = index
    post_pred_df.index = index

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    post_gt_df.to_csv(f'{args.outdir}/{args.dataset}_{args.seed}_cpa_post-gt.csv')
    post_pred_df.to_csv(f'{args.outdir}/{args.dataset}_{args.seed}_cpa_post-pred.csv')