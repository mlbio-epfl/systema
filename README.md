# Unmasking Systematic Variation in Single-cell Perturbation Response Prediction

This repository implements several single-cell perturbation response prediction methods under a unified pipeline. In this work, we demonstrate that existing single-cell perturbation datasets are often influenced by systematic variation (_i.e._, systematic differences between perturbed and control cells), which may arise as a result of selection biases and confounding factors. We study to what extent perturbation response prediction methods can generalize beyond these systematic effects. We also show that existing reference-based metrics are susceptible to systematic variation and demonstrate that predicting transcriptional outcomes of unseen genetic perturbations is remarkably hard. For more details, please check out our paper.

---

#### Methods
Currently, we support the following methods:
* CPA
* GEARS
* scGPT

To run a method, execute the following command:
```
python src/run_METHOD.py --dataset DATASET (--seed SEED --device DEVICE --epochs EPOCHS)
```
This will train the method and store the test predictions for a given train/test split (split controlled by the random seed). The script `run_all.sh` executes all baselines for every dataset. We welcome contributions implementing additional methods.

#### Datasets
Currently, we support the following datasets:
* Dixit (2016)
* Adamson (2016)
* Norman (2019)
* Replogle RPE1 (2022)
* Replogle K562 (2022)

#### Evaluation

The notebooks `analysis/benchmark_*.ipynb` evaluate perturbation response prediction performance on different datasets. We load the predictions of different methods for a given dataset and multiple train/test splits. We then compute evaluation metrics based on these predictions.

## Environment
We recommend using Python >=3.10. The file `requirements.txt` contains the library versions of our environment. We used a separate environment for running scGPT (because of incompatibility issues with CPA/GEARS), the versions of this environment can be found in `scgpt_requirements.txt`.

## Citing
If you find this repository useful, please consider citing:

```
@article{vinas2024unmasking,
  title={Unmasking Systematic Variation in Single-cell Perturbation Response Prediction},
  author={Vinas Torne, Ramon and Wiatrak, Maciej and Piran, Zoe and Fan, Shuyang and Jiang, Liangze and Nitzan, Mor and Brbic, Maria},
  journal={biorxiv},
  year={2024},
}
```
