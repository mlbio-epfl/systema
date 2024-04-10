import os
from pathlib import Path
from gears import PertData
from gears.utils import dataverse_download
import tarfile
import anndata


def norman2019(seed, data_dir='data'):
    d = f'{data_dir}/norman2019'
    file = f'{d}/norman2019_{seed}.h5ad'

    if not os.path.exists(file):
        Path(d).mkdir(parents=True, exist_ok=True)
        tar_file = f'{d}/norman_umi_go.tar.gz'
        if not os.path.exists(tar_file):
            # Download dataloader from dataverse
            dataverse_download('https://dataverse.harvard.edu/api/access/datafile/6979957', tar_file)

            # Extract and set up dataloader directory
            with tarfile.open(tar_file, 'r:gz') as tar:
                tar.extractall(path=d)

        # Create split and save data (when GEARS restores saved splits,
        # these are not stored in pert_adata.adata.obs['split'])
        pert_data = PertData(d)
        pert_data.load(data_name='norman')
        pert_data.prepare_split(split='simulation', seed=seed)
        pert_data.adata.write_h5ad(file)
    else:
        pert_data = PertData(d)
        pert_data.load(data_name='norman')
        pert_data.prepare_split(split='simulation', seed=seed)
        pert_data.adata = anndata.read_h5ad(file)  # Manually setting the data to ensure same split

    return pert_data


def get_pert_data(dataset, seed):
    if dataset == 'Norman2019':
        return norman2019(seed=seed)
    else:
        raise ValueError(f'Dataset {dataset} not supported')
