import pandas as pd
import numpy as np
import glob


def read_h5(file):
    with pd.HDFStore(file) as hdf_store:
        metadata = hdf_store.get_storer('data').attrs.metadata
        df_read = hdf_store.get('data')
    df_read = df_read.dropna()
    convert_dict = {'state': int, 'frame': int, 'traj_idx': int}
    df_read = df_read.astype(convert_dict)
    return df_read, metadata


def read_csv(file):
    csv_data = pd.read_csv(file, na_filter=False)
    col_names = ['traj_idx', 'frame', 'x', 'y', 'z', 'state', 'K', 'alpha']
    z = np.empty(len(csv_data.iloc[:, 1]))
    state = np.empty(len(csv_data.iloc[:, 1]))
    K = np.empty(len(csv_data.iloc[:, 1]))
    alpha = np.empty(len(csv_data.iloc[:, 1]))
    z.fill(np.nan)
    state.fill(np.nan)
    K.fill(np.nan)
    alpha.fill(np.nan)
    if np.var(csv_data['z']) < 1e-5:
        csv_data = csv_data.assign(z = z)
    csv_data = csv_data.assign(state = state)
    csv_data = csv_data.assign(K = K)
    csv_data = csv_data.assign(alpha = alpha)
    return csv_data


def read_multiple_h5s(path):
    dfs = []
    meta_info = []
    files_not_same_conditions = []
    prefix = f'_biadd'

    f_list = glob.glob(f'{path}/*{prefix}.h5')
    for f_idx, file in enumerate(f_list):
        df, meta = read_h5(file)
        if f_idx != 0:
            if meta['sample_id'] not in meta_info:
                files_not_same_conditions.append(file)
                continue
            else:
                pure_f_name = file.split('/')[-1].split(f'{prefix}.h5')[0]
                df['filename'] = [pure_f_name] * len(df['traj_idx'])
                traj_indices = df['traj_idx']
                traj_indices = [f'{pure_f_name}_{idx}' for idx in traj_indices]
                df['traj_idx'] = traj_indices
        else:
            meta_info.append(meta['sample_id'])
            pure_f_name = file.split('/')[-1].split(f'{prefix}.h5')[0]
            df['filename'] = [pure_f_name] * len(df['traj_idx'])
            traj_indices = df['traj_idx']
            traj_indices = [f'{pure_f_name}_{idx}' for idx in traj_indices]
            df['traj_idx'] = traj_indices
        dfs.append(df)
    grouped_df = pd.concat(dfs)

    if len(files_not_same_conditions) > 1:
        print('*****************************************************************************************')
        print("Below files are skipped due to their conditions are not same, check metadata of h5 file")
        for ff in files_not_same_conditions:
            print(ff)
        print('*****************************************************************************************')
    return grouped_df


def read_multiple_csv(path):
    dfs = []
    f_list = glob.glob(f'{path}/*_traces.csv')
    for f_idx, file in enumerate(f_list):
        df = read_csv(file)
        if f_idx != 0:
            pure_f_name = file.split('/')[-1].split(f'.csv')[0]
            df['filename'] = [pure_f_name] * len(df['traj_idx'])
            traj_indices = df['traj_idx']
            traj_indices = [f'{pure_f_name}_{idx}' for idx in traj_indices]
            df['traj_idx'] = traj_indices
        else:
            pure_f_name = file.split('/')[-1].split(f'.csv')[0]
            df['filename'] = [pure_f_name] * len(df['traj_idx'])
            traj_indices = df['traj_idx']
            traj_indices = [f'{pure_f_name}_{idx}' for idx in traj_indices]
            df['traj_idx'] = traj_indices
        dfs.append(df)
    grouped_df = pd.concat(dfs) 
    return grouped_df


def andi2_label_parser(path):
    andi_dict = {}
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\n')[0].split(',')
            traj_idx = line[0]
            Ks = []
            alphas = []
            states = []
            cps = []
            for K, alpha, state, cp in np.array(line[1:], dtype=object).reshape(-1, 4):
                Ks.append(float(K))
                alphas.append(float(alpha))
                states.append(int(eval(state)))
                cps.append(int(eval(cp)))
            andi_dict[f'{path.split(".txt")[0].split("/")[-1]}@{traj_idx}'] = np.array([Ks, alphas, states, cps]).T
    return andi_dict


def read_mulitple_andi_labels(path):
    andi_dicts = {}
    prefix = 'fov_*'
    f_list = glob.glob(f'{path}/*{prefix}.txt')
    for f_idx, file in enumerate(f_list):
        andi_dict = andi2_label_parser(file)
        andi_dicts |= andi_dict
    return andi_dicts
