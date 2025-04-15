#!/bin/bash

# List of datasets
DATASETS=("Norman2019")  # Dataset names here

# List of seeds
SEEDS=(1 2 3)

device=0
epochs=20

# Loop through datasets
for dataset in "${DATASETS[@]}"; do
	# Loop through seed
	for seed in "${SEEDS[@]}"; do
		python src/run_nonctl-mean.py --dataset $dataset --seed $seed
		python src/run_matching-mean.py --dataset $dataset --seed $seed
		python src/run_cpa.py --dataset $dataset --seed $seed --device $device --epochs $epochs
		python src/run_gears.py --dataset $dataset --seed $seed --device $device --epochs $epochs
		# python src/run_scgpt.py --dataset $dataset --seed $seed --device $device --epochs $epochs
	done
done
