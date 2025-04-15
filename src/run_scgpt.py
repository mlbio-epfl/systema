import json
import time
import copy
import argparse
import warnings
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from torchtext.vocab import Vocab
from torchtext._torchtext import Vocab as VocabPybind
from torch_geometric.loader import DataLoader
from gears.utils import create_cell_graph_dataset_for_prediction

from scgpt.model import TransformerGenerator
from scgpt.loss import masked_mse_loss
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import set_seed, map_raw_id_to_vocab_id

from data import get_pert_data

warnings.filterwarnings("ignore")

# scGPT installation
# ! conda install r-base=3.6.1
# ! pip install scgpt "flash-attn<1.0.5"
# check https://github.com/bowang-lab/scGPT/blob/main/README.md if the above doesn't work

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="Norman2019")
parser.add_argument("--data_dir", default="data")
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--outdir", default="results")
parser.add_argument("--device", default=1, type=int)
parser.add_argument("--finetune", action="store_true")
parser.add_argument("--train", action="store_true")
parser.set_defaults(finetune=False)

# model specific
parser.add_argument("--modeldir", default="results/checkpoints/scGPT_human")
parser.add_argument("--modelfile", default=None)

# model hyperparameters
# default values are from the original scGPT implementation
parser.add_argument("--batchsize", default=8, type=int)
parser.add_argument("--epochs", default=15, type=int)
parser.add_argument("--lr", default=1e-4, type=int)

args = parser.parse_args()

model_params = {
    "load_model": args.modeldir,
    "model_file": args.modelfile,  # Added Ramon
    "load_param_prefixs": [
        "encoder",
        "value_encoder",
        "transformer_encoder",
    ],
    "embsize": 512,
    "d_hid": 512,
    "nlayers": 12,
    "nhead": 8,
    "n_layers_cls": 3,
    "dropout": 0.2,
    "use_fast_transformer": True,
}

trainer_params = {
    "mlm": True,  # masked language modeling
    "cls": False,  # celltype classification
    "cce": False,  # contrastive cell embedding
    "mvc": False,  # masked value prediction
    "ecs": False,  # elastic cell similarity
    "cell_emb_style": "cls",
    "mvc_decoder_style": "inner product, detach",
    "amp": True,
    "lr": args.lr,
    "batch_size": args.batchsize,
    "eval_batch_size": args.batchsize,
    "epochs": args.epochs,
    "schedule_interval": 1,
    "early_stop": 5,
    "log_interval": 100,
}

data_params = {
    "pad_tokens": "<pad>",
    "special_tokens": ["<pad>", "<cls>", "<eoc>"],
    "pad_value": 0,
    "pert_pad_id": 2,
    "n_hvg": 0,  # number of highly variable genes
    "include_zero_gene": "all",  # include zero expr genes in training input, "all", "batch-wise", "row-wise", or False
    "max_seq_len": 10000, #5100,  # 100,  # 1536,
    "control_pool_size": 100,  # None,
    # number of cells in the control and predict their perturbation results. If `None`, use all control cells.
}


def scgpt_forward(
    batch_data,
    model,
    criterion,
    gene_ids,
    data_params,
    trainer_params,
    n_genes,
    device,
    test=False,
):
    batch_size = int(batch_data.x.shape[0] / n_genes)
    batch_data.to(device)
    x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 1)
    ori_gene_values = x[:, 0].view(batch_size, n_genes)
    # pert_flags = x[:, 1].long().view(batch_size, n_genes)
    # Reconstruct the perturbation flags
    pert_flags = torch.zeros_like(ori_gene_values, dtype=torch.long, device=device)
    # print('Dev:', device)
    if batch_data.pert is not None:
        for i, p in enumerate(batch_data.pert):
            if type(p) is list:
                gene_list = p
                if 'ctrl' in gene_list:
                    gene_list.remove('ctrl')
            else:  # Perturbation in A + B format
                gene_list = list(set(p.split("+")) - set(["ctrl"]))
            for g in gene_list:
                if g in data_params["genes"]:
                    # Replicating GEARS behaviour: https://github.com/snap-stanford/GEARS/blob/719328bd56745ab5f38c80dfca55cfd466ee356f/gears/model.py#L151
                    pert_flags[i, data_params["genes"][g]] = 1

    if data_params["include_zero_gene"] in ["all", "batch-wise"]:
        if data_params["include_zero_gene"] == "all":
            input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
        else:  # when batch-wise
            input_gene_ids = (
                ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
            )

        # sample input_gene_id
        if not test and len(input_gene_ids) > data_params["max_seq_len"]:
            input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[
                : data_params["max_seq_len"]
            ]
        input_values = ori_gene_values[:, input_gene_ids]
        input_pert_flags = pert_flags[:, input_gene_ids]
        if not test:
            target_gene_values = batch_data.y  # (batch_size, n_genes)
            target_values = target_gene_values[:, input_gene_ids]

        mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
        mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

        src_key_padding_mask = torch.zeros_like(
            input_values, dtype=torch.bool, device=input_values.device
        )
    with torch.cuda.amp.autocast(enabled=trainer_params["amp"]):
        output_dict = model(
            mapped_input_gene_ids,
            input_values,
            input_pert_flags,
            src_key_padding_mask=src_key_padding_mask,
            CLS=trainer_params["cls"],
            CCE=trainer_params["cce"],
            MVC=trainer_params["mvc"],
            ECS=trainer_params["ecs"],
            do_sample=True,
        )
        output_values = output_dict["mlm_output"]

    if not test:
        masked_positions = torch.ones_like(
            input_values, dtype=torch.bool, device=input_values.device
        )
        loss = criterion(output_values, target_values, masked_positions)
        return loss
    return output_values


if __name__ == "__main__":
    set_seed(args.seed)
    # device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    device = f"cuda:{args.device}"

    # Set different name for scGPT finetuned
    model_name = "scgpt"
    if args.finetune:
        model_name += "_ft"
    else:
        model_params["load_model"] = None

    # Load data
    pert_data = get_pert_data(dataset=args.dataset, seed=args.seed, data_dir=args.data_dir)
    pert_data.get_dataloader(batch_size=args.batchsize, test_batch_size=args.batchsize)

    # Load metadata of the model and data
    if args.finetune and model_params["load_model"] is not None:
        model_dir = Path(model_params["load_model"])
        model_config_file = model_dir / "args.json"
        model_file = model_dir / "best_model.pt"
        vocab_file = model_dir / "vocab.json"

        # Setup gene vocabulary, then check how many genes in the data are in the vocabulary
        vocab = GeneVocab.from_file(vocab_file)
        for s in data_params["special_tokens"]:
            if s not in vocab:
                vocab.append_token(s)

        # Sanity check how many genes in the data are in the vocabulary
        pert_data.adata.var["id_in_vocab"] = [
            1 if gene in vocab else -1 for gene in pert_data.adata.var["gene_name"]
        ]
        gene_ids_in_vocab = np.array(pert_data.adata.var["id_in_vocab"])
        print(
            f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(vocab)}."
        )

        genes = pert_data.adata.var["gene_name"].tolist()

        with open(model_config_file, "r") as f:
            model_configs = json.load(f)
        print(
            f"Resume model from {model_file}, the model args will be overriden by the config {model_config_file}."
        )
        model_params["embsize"] = model_configs["embsize"]
        model_params["nhead"] = model_configs["nheads"]
        model_params["d_hid"] = model_configs["d_hid"]
        model_params["nlayers"] = model_configs["nlayers"]
        model_params["n_layers_cls"] = model_configs["n_layers_cls"]
    else:
        model_file = None

        # Rename duplicate genes (not supported by VocabPybind)
        pert_data.adata.var["gene_name"] = (
            pert_data.adata.var["gene_name"]
            .astype(str)
            .where(
                ~pert_data.adata.var["gene_name"].duplicated(),
                pert_data.adata.var["gene_name"].astype(str) + "_dp",
            )
        )

        genes = pert_data.adata.var["gene_name"].tolist()
        vocab = Vocab(
            VocabPybind(genes + data_params["special_tokens"], None)
        )  # bidirectional lookup [gene <-> int]

    #### Added Ramon
    if model_params['model_file'] is not None:
        model_file = model_params['model_file']
        print('Using fine-tuned model from', model_file)

    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(
        [vocab[gene] if gene in vocab else vocab["<pad>"] for gene in genes], dtype=int
    )
    # data_params["genes"] = {value: index for index, value in enumerate(genes)}
    # data_params["pert_names"] = pert_data.pert_names
    data_params["genes"] = {value: index for index, value in enumerate(genes)}
    n_genes = len(genes)

    # Set up scGPT model
    model = TransformerGenerator(
        len(vocab),  # size of vocabulary
        model_params["embsize"],
        model_params["nhead"],
        model_params["d_hid"],
        model_params["nlayers"],
        nlayers_cls=model_params["n_layers_cls"],
        n_cls=1,
        vocab=vocab,
        dropout=model_params["dropout"],
        pad_token=data_params["pad_tokens"],
        pad_value=data_params["pad_value"],
        pert_pad_id=data_params["pert_pad_id"],
        do_mvc=trainer_params["mvc"],
        cell_emb_style=trainer_params["cell_emb_style"],
        mvc_decoder_style=trainer_params["mvc_decoder_style"],
        use_fast_transformer=model_params["use_fast_transformer"],
    )

    # Load model weights, either partially or try to load all
    if (
        model_params["load_param_prefixs"] is not None
        and model_params["load_model"] is not None
    ):
        # Only load params that start with the prefix
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file, map_location=device)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if any(
                [k.startswith(prefix) for prefix in model_params["load_param_prefixs"]]
            )
        }
        for k, v in pretrained_dict.items():
            print(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    elif model_params["load_model"] is not None:
        try:
            model.load_state_dict(torch.load(model_file))
            print(f"Loading all model params from {model_file}")
        except:
            # Only load params that are in the model and match the size
            model_dict = model.state_dict()
            pretrained_dict = torch.load(model_file, map_location=device)
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            for k, v in pretrained_dict.items():
                print(f"Loading params {k} with shape {v.shape}")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
    model.to(device)

    # Training
    criterion = None
    if args.train:
        criterion = masked_mse_loss
        optimizer = torch.optim.Adam(model.parameters(), lr=trainer_params["lr"])
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, trainer_params["schedule_interval"], gamma=0.9
        )
        scaler = torch.cuda.amp.GradScaler(enabled=trainer_params["amp"])
        best_val_loss = float("inf")
        best_model = None
        patience = 0

        for epoch in range(1, trainer_params["epochs"] + 1):
            epoch_start_time = time.time()
            train_loader = pert_data.dataloader["train_loader"]
            valid_loader = pert_data.dataloader["val_loader"]

            # Training epoch
            model.train()
            total_loss = 0.0
            start_time = time.time()  # time of the last log

            num_batches = len(train_loader)
            for batch, batch_data in enumerate(train_loader):
                loss = scgpt_forward(
                    batch_data,
                    model,
                    criterion,
                    gene_ids,
                    data_params,
                    trainer_params,
                    n_genes,
                    device,
                )
                model.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                with warnings.catch_warnings(record=True) as w:
                    warnings.filterwarnings("always")
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        1.0,
                        error_if_nonfinite=False if scaler.is_enabled() else True,
                    )
                    if len(w) > 0:
                        print(
                            f"Found infinite gradient. This may be caused by the gradient "
                            f"scaler. The current scale is {scaler.get_scale()}. This warning "
                            "can be ignored if no longer occurs after autoscaling of the scaler."
                        )
                scaler.step(optimizer)
                scaler.update()

                # torch.cuda.empty_cache()

                total_loss += loss.item()
                if batch % trainer_params["log_interval"] == 0 and batch > 0:
                    lr = scheduler.get_last_lr()[0]
                    ms_per_batch = (
                        (time.time() - start_time)
                        * 1000
                        / trainer_params["log_interval"]
                    )
                    cur_loss = total_loss / trainer_params["log_interval"]
                    # ppl = math.exp(cur_loss)
                    print(
                        f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                        f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                        f"loss {cur_loss:5.2f} |"
                    )
                    total_loss = 0
                    start_time = time.time()

            # Validation
            model.eval()
            total_loss = 0.0

            with torch.no_grad():
                for batch, batch_data in enumerate(valid_loader):
                    loss = scgpt_forward(
                        batch_data,
                        model,
                        criterion,
                        gene_ids,
                        data_params,
                        trainer_params,
                        n_genes,
                        device,
                    )
                    total_loss += loss.item()
            val_loss = total_loss / len(valid_loader)

            # Print epoch summary
            elapsed = time.time() - epoch_start_time
            print("-" * 20)
            print(
                f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
                f"valid loss/mse {val_loss:5.4f} |"
            )
            print("-" * 20)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(model)
                print(f"Best model with score {best_val_loss:5.4f}")
                patience = 0
            else:
                patience += 1
                if patience >= trainer_params["early_stop"]:
                    print(f"Early stop at epoch {epoch}")
                    break

            scheduler.step()

        # Save the best model
        Path(f"{args.outdir}/checkpoints").mkdir(parents=True, exist_ok=True)
        torch.save(
            best_model.state_dict(),
            f"{args.outdir}/checkpoints/{model_name}_seed{args.seed}_{args.dataset}_epoch{epoch}",
        )

    # Split train and test
    test_adata = pert_data.adata[pert_data.adata.obs["split"] == "test"]
    train_adata = pert_data.adata[pert_data.adata.obs["split"] == "train"]

    # Get control mean, non control mean (pert_mean), and non control mean differential
    control_adata = train_adata[train_adata.obs["control"] == 1]
    pert_adata = train_adata[train_adata.obs["control"] == 0]
    control_mean = np.array(control_adata.X.mean(axis=0))[0]
    pert_mean = np.array(pert_adata.X.mean(axis=0))[0]
    delta_pert = pert_mean - control_mean

    # Store results
    unique_conds = list(set(test_adata.obs["condition"].unique()) - set(["ctrl"]))
    post_gt_df = pd.DataFrame(columns=pert_data.adata.var["gene_name"].values)
    post_pred_df = pd.DataFrame(columns=pert_data.adata.var["gene_name"].values)
    train_counts = []
    model.eval()
    with torch.no_grad():
        for condition in tqdm(unique_conds):
            gene_list = condition.split("+")
            if "ctrl" in gene_list:
                gene_list.remove("ctrl")

            # Select adata condition
            adata_condition = test_adata[test_adata.obs["condition"] == condition]
            X_post = np.array(adata_condition.X.mean(axis=0))[
                0
            ]  # adata_condition.X.mean(axis=0) is a np.matrix of shape (1, n_genes)

            # Store number of train perturbations
            n_train = 0
            for g in gene_list:
                if f"{g}+ctrl" in train_adata.obs["condition"].values:
                    n_train += 1
                elif f"ctrl+{g}" in train_adata.obs["condition"].values:
                    n_train += 1
            train_counts.append(n_train)

            # Predict and collect the results for scGPT
            ctrl_adata = pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
            if data_params["control_pool_size"] is None:
                data_params["control_pool_size"] = len(ctrl_adata.obs)

            """ Removing check for Replogle: some perturbations are not observed genes
            for i in gene_list:
                if i not in pert_data.gene_names.values.tolist():
                    print(i)
                    raise ValueError(
                        "The gene is not in the perturbation graph. Please select from GEARS.gene_list!"
                    )
            """
            for i in gene_list:
                if i not in pert_data.gene_names.values.tolist():
                    print(f"Warning: {i} is not in the perturbation graph")

            pert = gene_list  # condition.split("+").remove('ctrl')
            cell_graphs = create_cell_graph_dataset_for_prediction(
                pert,
                ctrl_adata,
                pert_data.gene_names.values.tolist(),
                device,
                num_samples=data_params["control_pool_size"],
            )
            loader = DataLoader(
                cell_graphs,
                batch_size=trainer_params["eval_batch_size"],
                shuffle=False,
            )
            preds = []
            for batch_data in loader:
                pred_gene_values = scgpt_forward(
                    batch_data,
                    model,
                    None,
                    gene_ids,
                    data_params,
                    trainer_params,
                    n_genes,
                    device,
                    test=True,
                )
                """
                pred_gene_values = model.pred_perturb(
                    batch_data, data_params["include_zero_gene"], gene_ids=gene_ids, amp=trainer_params["amp"]
                )
                """
                preds.append(pred_gene_values)
            preds = torch.cat(preds, dim=0)
            print(preds.shape, post_pred_df.shape)
            preds = np.mean(preds.detach().cpu().numpy(), axis=0)

            # scGPT predictions
            post_gt_df.loc[len(post_gt_df)] = X_post
            post_pred_df.loc[len(post_pred_df)] = preds

        index = pd.MultiIndex.from_tuples(
            list(zip(unique_conds, train_counts)), names=["condition", "n_train"]
        )
        post_gt_df.index = index
        post_pred_df.index = index

        Path(args.outdir).mkdir(parents=True, exist_ok=True)
        post_gt_df.to_csv(
            f"{args.outdir}/{args.dataset}_{args.seed}_{model_name}_post-gt.csv"
        )
        post_pred_df.to_csv(
            f"{args.outdir}/{args.dataset}_{args.seed}_{model_name}_post-pred.csv"
        )
