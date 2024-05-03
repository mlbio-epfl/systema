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
parser.add_argument("--dataset", default="Norman2019")
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--outdir", default="results")
parser.add_argument("--mode", default="all", choices=["all", "de"])

args = parser.parse_args()

if __name__ == "__main__":
    pert_data = get_pert_data(dataset=args.dataset, seed=args.seed)

    # Split train and test
    test_adata = pert_data.adata[pert_data.adata.obs["split"] == "test"]
    train_adata = pert_data.adata[pert_data.adata.obs["split"] == "train"]

    # Get control mean, non control mean (pert_mean), and non control mean differential
    control_adata = train_adata[train_adata.obs["control"] == 1]
    pert_adata = train_adata[train_adata.obs["control"] == 0]
    control_mean = np.array(control_adata.X.mean(axis=0))[0]
    pert_mean = np.array(pert_adata.X.mean(axis=0))[0]
    delta_pert = pert_mean - control_mean

    # map of gene id and gene name, e.g., 'RP11-34P13.8' -> 'ENSG00000239945'
    id2gene = train_adata.var.to_dict()["gene_name"]
    gene2id = {v: k for k, v in id2gene.items()}

    # Store results
    unique_conds = list(set(test_adata.obs["condition"].unique()) - set(["ctrl"]))
    post_gt_df = pd.DataFrame(columns=pert_data.adata.var["gene_name"].values)
    post_pred_df = pd.DataFrame(columns=pert_data.adata.var["gene_name"].values)
    train_counts = []
    for condition in tqdm(unique_conds):
        gene_list = condition.split("+")

        # Select adata condition
        adata_condition = test_adata[test_adata.obs["condition"] == condition]
        X_post = np.array(adata_condition.X.mean(axis=0))[
            0
        ]  # adata_condition.X.mean(axis=0) is a np.matrix of shape (1, n_genes)

        # Matching mean. Get matching differentials from train data
        X_perts = []
        n_train = 0
        for g in gene_list:
            if f"{g}+ctrl" in train_adata.obs["condition"].values:
                cond_adata = train_adata[train_adata.obs["condition"] == f"{g}+ctrl"]
                n_train += 1
            elif f"ctrl+{g}" in train_adata.obs["condition"].values:
                cond_adata = train_adata[train_adata.obs["condition"] == f"ctrl+{g}"]
                n_train += 1
            else:
                if args.mode == "all" or g == "ctrl":
                    cond_adata = pert_adata
                elif args.mode == "de":
                    gene_name = gene2id[g]  # e.g., ENSG00000228980
                    cond_name = [  # check in which conditions the gene is in the top 20 non-dropout de genes
                        k
                        for k, v in pert_adata.uns["top_non_dropout_de_20"].items()
                        if gene_name in v
                    ]  # e.g., ['A549_MAP2K6+SPI1_1+1', 'A549_SPI1+ctrl_1+1']
                    cond_adata = pert_adata[
                        pert_adata.obs["condition_name"].isin(cond_name)
                    ]
                    if len(cond_adata) == 0:
                        cond_adata = pert_adata
                else:
                    raise ValueError("Invalid mode")
            X_pert = np.array(cond_adata.X.mean(axis=0))[0]
            X_perts.append(X_pert)
        train_counts.append(n_train)
        match_mean = np.mean(X_perts, axis=0)
        post_gt_df.loc[len(post_gt_df)] = X_post
        post_pred_df.loc[len(post_pred_df)] = match_mean

    index = pd.MultiIndex.from_tuples(
        list(zip(unique_conds, train_counts)), names=["condition", "n_train"]
    )
    post_gt_df.index = index
    post_pred_df.index = index
    post_gt_df.to_csv(
        f"{args.outdir}/{args.dataset}_{args.seed}_matching-mean-{args.mode}_post-gt.csv"
    )
    post_pred_df.to_csv(
        f"{args.outdir}/{args.dataset}_{args.seed}_matching-mean-{args.mode}_post-pred.csv"
    )
