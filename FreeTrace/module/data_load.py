import glob
import sys
import pandas as pd
import numpy as np
from FreeTrace.module.trajectory_object import TrajectoryObj


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


def read_localization(input_file, video=None):
    locals = {}
    locals_info = {}
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
            if len(lines) == 1 or len(lines) == 2:
                raise Exception('Cannot track on zero localization OR single localization.')
            for line in lines[1:]:
                line = line.strip().split('\n')[0].split(',')
                if int(line[0]) not in locals:
                    locals[int(line[0])] = []
                    locals_info[int(line[0])] = []
                pos_line = []
                info_line = []
                for dt in line[1:4]:
                    pos_line.append(np.round(float(dt), 7))
                for dt in line[4:]:
                    info_line.append(np.round(float(dt), 7))
                locals[int(line[0])].append(pos_line)
                locals_info[int(line[0])].append(info_line)
        if video is None:
            max_t = np.max(list(locals.keys()))
        else:
            max_t = len(video)
        for t in np.arange(1, max_t+1):
            if t not in locals:
                locals[t] = [[]]
                locals_info[t] = [[]]

        ret_locals = {}
        ret_locals_info = {}

        for t in locals.keys():
            ret_locals[t] = np.array(locals[t])
            ret_locals_info[t] = np.array(locals_info[t])
        return ret_locals, ret_locals_info
    except Exception as e:
        sys.exit(f'Err msg: {e}')


def read_multiple_locs(input_files: list[str]):
    all_locs = {}
    all_locs_info ={}
    for input_file in input_files:
        loc, loc_info = read_localization(input_file)
        cur_t_steps = list(loc.keys())
        for cur_t in cur_t_steps:
            if cur_t in all_locs:
                all_locs[cur_t].extend(loc[cur_t])
                all_locs_info[cur_t].extend(loc_info[cur_t])
            else:
                all_locs[cur_t] = list(loc[cur_t])
                all_locs_info[cur_t] = list(loc_info[cur_t])
    
    for t in all_locs.keys():
        all_locs[t] = np.array(all_locs[t])
        all_locs_info[t] = np.array(all_locs_info[t])
    return all_locs, all_locs_info


def read_trajectory(file: str, andi_gt=False, pixel_microns=1.0, frame_rate=1.0) -> dict | list:
    """
    @params : filename(String), cutoff value(Integer)
    @return : dictionary of H2B objects(Dict)
    Read a single trajectory file and return the dictionary.
    key is consist of filename@id and the value is H2B object.
    Dict only keeps the histones which have the trajectory length longer than cutoff value.
    """
    filetypes = ['trxyt', 'trx', 'csv']
    # Check filetype.
    assert file.strip().split('.')[-1].lower() in filetypes
    # Read file and store the trajectory and time information in H2B object
    if file.strip().split('.')[-1].lower() in ['trxyt', 'trx']:
        localizations = {}
        tmp = {}
        try:
            with open(file, 'r', encoding="utf-8") as f:
                input = f.read()
            lines = input.strip().split('\n')
            for line in lines:
                temp = line.split('\t')
                x_pos = float(temp[1].strip()) * pixel_microns
                y_pos = float(temp[2].strip()) * pixel_microns
                z_pos = 0. * pixel_microns
                time_step = float(temp[3].strip()) * frame_rate
                if time_step in tmp:
                    tmp[time_step].append([x_pos, y_pos, z_pos])
                else:
                    tmp[time_step] = [[x_pos, y_pos, z_pos]]

            time_steps = np.sort(np.array(list(tmp.keys())))
            first_frame, last_frame = time_steps[0], time_steps[-1]
            steps = np.arange(int(np.round(first_frame * 100)), int(np.round(last_frame * 100)) + 1)
            for step in steps:
                if step/100 in tmp:
                    localizations[step] = tmp[step/100]
                else:
                    localizations[step] = []
            return localizations
        except Exception as e:
            print(f"Unexpected error, check the file: {file}")
            print(e)
    else:
        try:
            trajectory_list = []
            with open(file, 'r', encoding="utf-8") as f:
                input = f.read()
            lines = input.strip().split('\n')
            nb_traj = 0
            old_index = -999
            for line in lines[1:]:
                temp = line.split(',')
                index = int(float(temp[0].strip()))
                frame = int(float(temp[1].strip()))
                x_pos = float(temp[2].strip())
                y_pos = float(temp[3].strip())
                if andi_gt:
                    x_pos = float(temp[3].strip())
                    y_pos = float(temp[2].strip())
                if len(temp) > 4:
                    z_pos = float(temp[4].strip())
                else:
                    z_pos = 0.0

                if index != old_index:
                    nb_traj += 1
                    trajectory_list.append(TrajectoryObj(index=index, max_pause=5))
                    trajectory_list[nb_traj - 1].add_trajectory_position(frame * frame_rate, x_pos * pixel_microns, y_pos * pixel_microns, z_pos * pixel_microns)
                else:
                    trajectory_list[nb_traj - 1].add_trajectory_position(frame * frame_rate, x_pos * pixel_microns, y_pos * pixel_microns, z_pos * pixel_microns)
                old_index = index
            return trajectory_list
        except Exception as e:
            print(f"Unexpected error, check the file: {file}")
            print(e)
