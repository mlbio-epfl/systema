[![DOI](https://zenodo.org/badge/784794939.svg)](https://doi.org/10.5281/zenodo.15746994)

# Evaluating Single-cell Perturbation Response Prediction Beyond Systematic Variation

This repository implements several single-cell perturbation response prediction methods under a unified pipeline. In this work, we demonstrate that existing single-cell perturbation benchmarks are often influenced by systematic variation (_i.e._, systematic differences between perturbed and control cells), which may arise as a result of selection biases and confounding factors. We study to what extent perturbation response prediction methods can generalize beyond these systematic effects. We also show that existing reference-based metrics are susceptible to systematic variation and demonstrate that predicting transcriptional outcomes of unseen genetic perturbations is remarkably hard. For more details, please check out our paper.

---

#### Methods
Currently, we support the following methods:
* CPA (`cpa`)
* GEARS (`gears`)
* scGPT (`scgpt`)

We implemented the following two non-parametric baselines:
* Non-control mean (`nonctl-mean`)
* Matching mean (`matching-mean`)

To run a method, execute the following command:
```
python src/run_METHOD.py --dataset DATASET (--finetune --seed SEED --device DEVICE --epochs EPOCHS)
```
This will train the method and store the test predictions for a given train/test split (split controlled by the random seed). The script `run_all.sh` executes all baselines for every dataset. We welcome contributions implementing additional methods.

#### Datasets
Currently, we support the following datasets:
* Adamson (2016), downloaded via GEARS. Gene Expression Omnibus accession number: [GSE90546](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90546)
* Norman (2019), downloaded via GEARS. Gene Expression Omnibus accession number: [GSE14619](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE14619)
* Xu (2024). Gene Expression Omnibus accession number: [GSE218566](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE218566)
* Replogle essential RPE1 (2022) available at: https://doi.org/10.25452/figshare.plus.20022944
* Replogle gwps K562 (2022) available at: https://doi.org/10.25452/figshare.plus.20022944
* Frangieh (2021) available at: https://singlecell.broadinstitute.org/single_cell/study/SCP1064/multi-modal-pooled-perturb-cite-seq-screens-in-patient-models-define-novel-mechanisms-of-cancer-immune-evasion. 
* Tian CRISPRa (2019) available via scPerturb (Peidli et al., 2024) at: https://doi.org/10.5281/zenodo.13350497
* Tian CRISPRi (2019) available via scPerturb (Peidli et al., 2024) at: https://doi.org/10.5281/zenodo.13350497

The notebooks in the [processing](https://github.com/mlbio-epfl/perturb-bench/tree/main/processing) folder contain instructions and code to download and process the datasets.

#### Evaluation

The [evaluation](https://github.com/mlbio-epfl/perturb-bench/tree/main/evaluation) folder contains an easy-to-use implementation of the evaluation metrics developed in this paper, together with instructions and examples. The notebook `evaluation/benchmark_all.ipynb` evaluates perturbation response prediction performance on different datasets. We load the predictions of different methods for a given dataset and multiple train/test splits. We then compute evaluation metrics based on these predictions.

## Environment
We recommend using Python >=3.10. The file `env/requirements.txt` contains the library versions of our environment. We used a separate environment for running scGPT (because of incompatibility issues with CPA/GEARS), which can be created by `bash env/scgpt_env.sh`.

## Citing
If you find this repository useful, please consider citing:

```
@article{vinas2025systema,
  title={Systema: A Framework for Evaluating Genetic Perturbation Response Prediction Beyond Systematic Variation},
  author={Vinas Torne, Ramon and Wiatrak, Maciej and Piran, Zoe and Fan, Shuyang and Jiang, Liangze and A. Teichmann, Sarah and Nitzan, Mor and Brbic, Maria},
  journal={Nature Biotechnology},
  year={2025},
}
```
