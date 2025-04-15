import os
from pathlib import Path
from gears import PertData
import anndata
import numpy as np

def norman2019(seed, data_dir='data', suffix=''):
    d = f'{data_dir}/norman2019'
    file = f'{d}/norman2019_{seed}.h5ad'
    Path(d).mkdir(parents=True, exist_ok=True)

    # Create split and save data (when GEARS restores saved splits,
    # these are not stored in pert_adata.adata.obs['split'])
    pert_data = PertData(f'{d}{suffix}')
    pert_data.load(data_name='norman', data_path='norman2019')
    pert_data.prepare_split(split='simulation', seed=seed)
    if not os.path.exists(file):
        pert_data.adata.write_h5ad(file)
    else:
        pert_data.adata = anndata.read_h5ad(file)  # Manually setting the data to ensure same split
    return pert_data

def adamson2016(seed, data_dir='data', suffix=''):
    d = f'{data_dir}/adamson2016'
    file = f'{d}/adamson2016_{seed}.h5ad'
    Path(d).mkdir(parents=True, exist_ok=True)

    # Create split and save data (when GEARS restores saved splits,
    # these are not stored in pert_adata.adata.obs['split'])
    pert_data = PertData(f'{d}{suffix}')
    pert_data.load(data_name='adamson', data_path='adamson2016')
    pert_data.prepare_split(split='simulation', seed=seed)
    if not os.path.exists(file):
        pert_data.adata.write_h5ad(file)
    else:
        pert_data.adata = anndata.read_h5ad(file)  # Manually setting the data to ensure same split
    return pert_data

def dixit2016(seed, data_dir='data', suffix=''):
    d = f'{data_dir}/dixit2016'
    file = f'{d}/dixit2016_{seed}.h5ad'
    Path(d).mkdir(parents=True, exist_ok=True)

    # Create split and save data (when GEARS restores saved splits,
    # these are not stored in pert_adata.adata.obs['split'])
    pert_data = PertData(f'{d}{suffix}')
    pert_data.load(data_name='dixit', data_path='dixit2016')
    pert_data.prepare_split(split='simulation', seed=seed)
    if not os.path.exists(file):
        pert_data.adata.write_h5ad(file)
    else:
        pert_data.adata = anndata.read_h5ad(file)  # Manually setting the data to ensure same split
    return pert_data

def replogle_k562_gwps_2022(seed, data_dir='data'):
    d = f'{data_dir}/replogle_k562_gwps_2022'
    file = f'{d}/replogle_k562_gwps_2022_{seed}.h5ad'
    Path(d).mkdir(parents=True, exist_ok=True)
    # adata = anndata.read_h5ad(f'{d}/perturb_processed.h5ad')

    # Create split and save data (when GEARS restores saved splits,
    # these are not stored in pert_adata.adata.obs['split'])
    data_name = 'k562_1900_100'
    data_path = f'{d}/{data_name}'
    pert_data = PertData(data_path)
    pert_data.load(data_name=data_name, data_path=data_path)  # load the processed data, the path is saved folder + dataset_name
    pert_data.prepare_split(split='simulation', seed=seed)  # get data split with seed
    pert_data.dataset_name = ''

    if not os.path.exists(file):
        pert_data.adata.write_h5ad(file)
    else:
        pert_data.adata = anndata.read_h5ad(file)

    return pert_data

def replogle_rpe1_2022_v2(seed, data_dir='data'):
    d = f'{data_dir}/replogle_rpe1_v2_2022'
    file = f'{d}/replogle_rpe1_v2_2022_{seed}.h5ad'
    Path(d).mkdir(parents=True, exist_ok=True)
    # adata = anndata.read_h5ad(f'{d}/perturb_processed.h5ad')

    # Create split and save data (when GEARS restores saved splits,
    # these are not stored in pert_adata.adata.obs['split'])
    data_name = 'rpe1_1900_100'
    data_path = f'{d}/{data_name}'
    pert_data = PertData(data_path)
    # pert_data.new_data_process(dataset_name='replogle_k562_v2_2022', adata=adata, skip_calc_de = False)  # specific dataset name and adata object
    pert_data.load(data_name=data_name, data_path=data_path)  # load the processed data, the path is saved folder + dataset_name
    pert_data.prepare_split(split='simulation', seed=seed)  # get data split with seed
    pert_data.dataset_name = ''

    if not os.path.exists(file):
        pert_data.adata.write_h5ad(file)
    else:
        pert_data.adata = anndata.read_h5ad(file)

    return pert_data

def replogle_rpe1_cc_2022(seed, data_dir='data'):
    d = f'{data_dir}/replogle_rpe1_cc_2022'
    file = f'{d}/replogle_rpe1_cc_2022_{seed}.h5ad'
    Path(d).mkdir(parents=True, exist_ok=True)
    # adata = anndata.read_h5ad(f'{d}/perturb_processed.h5ad')

    # Create split and save data (when GEARS restores saved splits,
    # these are not stored in pert_adata.adata.obs['split'])
    data_name = 'rpe1_cc'
    data_path = f'{d}/{data_name}'
    pert_data = PertData(data_path)
    # pert_data.new_data_process(dataset_name='replogle_k562_v2_2022', adata=adata, skip_calc_de = False)  # specific dataset name and adata object
    pert_data.load(data_name=data_name, data_path=data_path)  # load the processed data, the path is saved folder + dataset_name
    pert_data.prepare_split(split='simulation', seed=seed)  # get data split with seed
    pert_data.dataset_name = ''

    if not os.path.exists(file):
        pert_data.adata.write_h5ad(file)
    else:
        pert_data.adata = anndata.read_h5ad(file)

    return pert_data

def tian_crispra_2021(seed, data_dir='data'):
    d = f'{data_dir}/tian_CRISPRa_2021_scperturb'
    file = f'{d}/tian_CRISPRa_2021_scperturb_{seed}.h5ad'
    Path(d).mkdir(parents=True, exist_ok=True)
    # adata = anndata.read_h5ad(f'{d}/perturb_processed.h5ad')

    # Create split and save data (when GEARS restores saved splits,
    # these are not stored in pert_adata.adata.obs['split'])
    data_name = 'crispra'
    data_path = f'{d}/{data_name}'
    pert_data = PertData(data_path)
    # pert_data.new_data_process(dataset_name='replogle_k562_v2_2022', adata=adata, skip_calc_de = False)  # specific dataset name and adata object
    pert_data.load(data_name=data_name, data_path=data_path)  # load the processed data, the path is saved folder + dataset_name
    pert_data.prepare_split(split='simulation', seed=seed)  # get data split with seed
    pert_data.dataset_name = ''

    if not os.path.exists(file):
        pert_data.adata.write_h5ad(file)
    else:
        pert_data.adata = anndata.read_h5ad(file)

    return pert_data

def tian_crispri_2021(seed, data_dir='data'):
    d = f'{data_dir}/tian_CRISPRi_2021_scperturb'
    file = f'{d}/tian_CRISPRi_2021_scperturb_{seed}.h5ad'
    Path(d).mkdir(parents=True, exist_ok=True)
    # adata = anndata.read_h5ad(f'{d}/perturb_processed.h5ad')

    # Create split and save data (when GEARS restores saved splits,
    # these are not stored in pert_adata.adata.obs['split'])
    data_name = 'crispri'
    data_path = f'{d}/{data_name}'
    pert_data = PertData(data_path)
    # pert_data.new_data_process(dataset_name='replogle_k562_v2_2022', adata=adata, skip_calc_de = False)  # specific dataset name and adata object
    pert_data.load(data_name=data_name, data_path=data_path)  # load the processed data, the path is saved folder + dataset_name
    pert_data.prepare_split(split='simulation', seed=seed)  # get data split with seed
    pert_data.dataset_name = ''

    if not os.path.exists(file):
        pert_data.adata.write_h5ad(file)
    else:
        pert_data.adata = anndata.read_h5ad(file)

    return pert_data

def xu_kinetics_2024(seed, data_dir='data'):
    d = f'{data_dir}/xu_kinetics_2024'
    file = f'{d}/xu_kinetics_2024_{seed}.h5ad'
    Path(d).mkdir(parents=True, exist_ok=True)
    # adata = anndata.read_h5ad(f'{d}/perturb_processed.h5ad')

    # Create split and save data (when GEARS restores saved splits,
    # these are not stored in pert_adata.adata.obs['split'])
    data_name = 'hek293'
    data_path = f'{d}/{data_name}'
    pert_data = PertData(data_path)
    # pert_data.new_data_process(dataset_name='replogle_k562_v2_2022', adata=adata, skip_calc_de = False)  # specific dataset name and adata object
    pert_data.load(data_name=data_name, data_path=data_path)  # load the processed data, the path is saved folder + dataset_name
    pert_data.prepare_split(split='simulation', seed=seed)  # get data split with seed
    pert_data.dataset_name = ''

    if not os.path.exists(file):
        pert_data.adata.write_h5ad(file)
    else:
        pert_data.adata = anndata.read_h5ad(file)

    return pert_data


def frangieh_control_single_2021(seed, data_dir='data', default_pert_graph=False):
    d = f'{data_dir}/frangieh_control_single_2021'
    file = f'{d}/frangieh_control_single_2021_{seed}.h5ad'
    Path(d).mkdir(parents=True, exist_ok=True)
    # adata = anndata.read_h5ad(f'{d}/perturb_processed.h5ad')

    # Create split and save data (when GEARS restores saved splits,
    # these are not stored in pert_adata.adata.obs['split'])
    data_name = 'control'
    data_path = f'{d}/{data_name}'
    pert_data = PertData(data_path)
    # pert_data.new_data_process(dataset_name='replogle_k562_v2_2022', adata=adata, skip_calc_de = False)  # specific dataset name and adata object
    pert_data.load(data_name=data_name, data_path=data_path)  # load the processed data, the path is saved folder + dataset_name
    pert_data.prepare_split(split='simulation', seed=seed)  # get data split with seed
    pert_data.dataset_name = ''

    if not os.path.exists(file):
        pert_data.adata.write_h5ad(file)
    else:
        pert_data.adata = anndata.read_h5ad(file)
    pert_data.adata.obs['control'] = pert_data.adata.obs['sgRNA'].astype(str).str.contains('_SITE_')

    return pert_data

def frangieh_coculture_single_2021(seed, data_dir='data', default_pert_graph=False):
    d = f'{data_dir}/frangieh_coculture_single_2021'
    file = f'{d}/frangieh_coculture_single_2021_{seed}.h5ad'
    Path(d).mkdir(parents=True, exist_ok=True)
    # adata = anndata.read_h5ad(f'{d}/perturb_processed.h5ad')

    # Create split and save data (when GEARS restores saved splits,
    # these are not stored in pert_adata.adata.obs['split'])
    data_name = 'coculture'
    data_path = f'{d}/{data_name}'
    pert_data = PertData(data_path)
    # pert_data.new_data_process(dataset_name='replogle_k562_v2_2022', adata=adata, skip_calc_de = False)  # specific dataset name and adata object
    pert_data.load(data_name=data_name, data_path=data_path)  # load the processed data, the path is saved folder + dataset_name
    pert_data.prepare_split(split='simulation', seed=seed)  # get data split with seed
    pert_data.dataset_name = ''

    if not os.path.exists(file):
        pert_data.adata.write_h5ad(file)
    else:
        pert_data.adata = anndata.read_h5ad(file)
    pert_data.adata.obs['control'] = pert_data.adata.obs['sgRNA'].astype(str).str.contains('_SITE_')

    return pert_data

def frangieh_ifn_single_2021(seed, data_dir='data', default_pert_graph=False):
    d = f'{data_dir}/frangieh_ifn_single_2021'
    file = f'{d}/frangieh_ifn_single_2021_{seed}.h5ad'
    Path(d).mkdir(parents=True, exist_ok=True)
    # adata = anndata.read_h5ad(f'{d}/perturb_processed.h5ad')

    # Create split and save data (when GEARS restores saved splits,
    # these are not stored in pert_adata.adata.obs['split'])
    data_name = 'ifn'
    data_path = f'{d}/{data_name}'
    pert_data = PertData(data_path)
    # pert_data.new_data_process(dataset_name='replogle_k562_v2_2022', adata=adata, skip_calc_de = False)  # specific dataset name and adata object
    pert_data.load(data_name=data_name, data_path=data_path)  # load the processed data, the path is saved folder + dataset_name
    pert_data.prepare_split(split='simulation', seed=seed)  # get data split with seed
    pert_data.dataset_name = ''

    if not os.path.exists(file):
        pert_data.adata.write_h5ad(file)
    else:
        pert_data.adata = anndata.read_h5ad(file)
    pert_data.adata.obs['control'] = pert_data.adata.obs['sgRNA'].astype(str).str.contains('_SITE_')

    return pert_data

def get_pert_data(dataset, seed, **kwargs):
    if dataset == 'Norman2019':
        return norman2019(seed=seed, **kwargs)
    elif dataset == "Adamson2016":
        return adamson2016(seed=seed, **kwargs)
    elif dataset == "Dixit2016":
        return dixit2016(seed=seed, **kwargs)
    elif dataset == "ReplogleK562_gwps":
        return replogle_k562_gwps_2022(seed=seed, **kwargs)
    elif dataset == "ReplogleRPE1_v2":
        return replogle_rpe1_2022_v2(seed=seed, **kwargs)
    elif dataset == "ReplogleRPE1_cc":
        return replogle_rpe1_cc_2022(seed=seed, **kwargs)
    elif dataset == "TianCRISPRa2021":
        return tian_crispra_2021(seed=seed, **kwargs)
    elif dataset == "TianCRISPRi2021":
        return tian_crispri_2021(seed=seed, **kwargs)
    elif dataset == "XuKinetics2024":
        return xu_kinetics_2024(seed=seed, **kwargs)
    elif dataset == "FrangiehControlSingle2021":
        return frangieh_control_single_2021(seed=seed, **kwargs)
    elif dataset == "FrangiehCocultureSingle2021":
        return frangieh_coculture_single_2021(seed=seed, **kwargs)
    elif dataset == "FrangiehIfnSingle2021":
        return frangieh_ifn_single_2021(seed=seed, **kwargs)
    else:
        raise ValueError(f'Dataset {dataset} not supported')