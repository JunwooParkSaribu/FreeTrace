import numpy as np
import pandas as pd
import networkx as nx
from itertools import product
from FreeTrace.module import DataLoad
from tqdm import tqdm


def preprocessing(data, pixelmicrons, framerate, cutoff, tamsd_calcul=True, color=None):
    # load FreeTrace+Bi-ADD data without NaN (NaN where trajectory length is shorter than 5, default in BI-ADD)
    data = data.dropna()
    # using dictionary to convert specific columns
    convert_dict = {'state': int}
    data = data.astype(convert_dict)
    traj_indices = pd.unique(data['traj_idx'])


    # initializations
    dim = 2 # will be changed in future.
    max_frame = data.frame.max()
    total_states = sorted(data['state'].unique())
    product_states = list(product(total_states, repeat=2))
    state_graph = nx.DiGraph()
    state_graph.add_nodes_from(total_states)
    state_graph.add_edges_from(product_states, weight=0)
    state_markov = [[0 for _ in range(len(total_states))] for _ in range(len(total_states))]
    analysis_data1 = {}
    analysis_data1[f'mean_jump_d'] = []
    analysis_data1[f'log10_K'] = []
    analysis_data1[f'alpha'] = []
    analysis_data1[f'state'] = []
    analysis_data1[f'duration'] = []
    analysis_data1[f'traj_id'] = []
    analysis_data1[f'color_code'] = []
    analysis_data2 = {}
    analysis_data2[f'displacements'] = []
    analysis_data2[f'state'] = []
    analysis_data2[f'color_code'] = []
    msd_ragged_ens_trajs = {st:[] for st in total_states}
    tamsd_ragged_ens_trajs = {st:[] for st in total_states}
    msd = {}
    msd[f'mean'] = []
    msd[f'std'] = []
    msd[f'nb_data'] = []
    msd[f'state'] = []
    msd[f'time'] = []
    tamsd = {}
    tamsd[f'mean'] = []
    tamsd[f'std'] = []
    tamsd[f'nb_data'] = []
    tamsd[f'state'] = []
    tamsd[f'time'] = []

    displace_tmp = []
    # get data from trajectories
    if tamsd_calcul:
        print("** Computing of Ensemble-averaged TAMSD takes a few minutes **")
    for traj_idx in tqdm(traj_indices, ncols=120, desc=f'Analysis', unit=f'trajectory'):
        single_traj = data.loc[data['traj_idx'] == traj_idx]
        # calculate state changes inside single trajectory
        before_st = single_traj.state.iloc[0]
        for st in single_traj.state:
            state_graph[before_st][st]['weight'] += 1
            before_st = st

        # chucnk into sub-trajectories
        before_st = single_traj.state.iloc[0]
        chunk_idx = [0, len(single_traj)]
        for st_idx, st in enumerate(single_traj.state):
            if st != before_st:
                chunk_idx.append(st_idx)
            before_st = st
        chunk_idx = sorted(chunk_idx)
        
        for i in range(len(chunk_idx) - 1):
            sub_trajectory = single_traj.iloc[chunk_idx[i]:chunk_idx[i+1]]
            # trajectory length filter condition
            if len(sub_trajectory) >= cutoff:
                # state of trajectory
                state = sub_trajectory.state.iloc[0]
                bi_add_alpha = sub_trajectory.alpha.iloc[0]
                bi_add_K = sub_trajectory.K.iloc[0]

                # convert from pixel-coordinate to micron.
                sub_trajectory.x *= pixelmicrons
                sub_trajectory.y *= pixelmicrons
                sub_trajectory.z *= pixelmicrons 
                bi_add_K *= (pixelmicrons**2/framerate**bi_add_alpha) #TODO: check again

                # coordinate normalize
                sub_trajectory.x -= sub_trajectory.x.iloc[0]
                sub_trajectory.y -= sub_trajectory.y.iloc[0]
                sub_trajectory.z -= sub_trajectory.z.iloc[0]
                
                # calcultae jump distances
                jump_distances = np.sqrt((sub_trajectory.x.iloc[1:].to_numpy() - sub_trajectory.x.iloc[:-1].to_numpy()) ** 2 + (sub_trajectory.y.iloc[1:].to_numpy() - sub_trajectory.y.iloc[:-1].to_numpy()) ** 2)
                displace_tmp.extend(list(jump_distances))
                # MSD
                msd_ragged_ens_trajs[state].append(((sub_trajectory.x.to_numpy())**2 + (sub_trajectory.y.to_numpy())**2) / dim / 2)

                # TAMSD
                if tamsd_calcul:
                    tamsd_tmp = []
                    for lag in range(len(sub_trajectory)):
                        time_averaged = []
                        for pivot in range(len(sub_trajectory) - lag):
                            time_averaged.append(((sub_trajectory.x.iloc[pivot + lag] - sub_trajectory.x.iloc[pivot]) ** 2 + (sub_trajectory.y.iloc[pivot + lag] - sub_trajectory.y.iloc[pivot]) ** 2) / dim / 2)
                        tamsd_tmp.append(np.mean(time_averaged))
                else:
                    tamsd_tmp = [0] * len(sub_trajectory)
                tamsd_ragged_ens_trajs[state].append(tamsd_tmp)


                # add data1 for the visualization
                analysis_data1[f'mean_jump_d'].append(jump_distances.mean())
                analysis_data1[f'log10_K'].append(np.log10(bi_add_K))
                analysis_data1[f'alpha'].append(bi_add_alpha)
                analysis_data1[f'state'].append(state)
                analysis_data1[f'duration'].append((sub_trajectory.frame.iloc[-1] - sub_trajectory.frame.iloc[0] + 1) * framerate)
                analysis_data1[f'traj_id'].append(sub_trajectory.traj_idx.iloc[0])
                analysis_data1[f'color_code'].append(color)
                

                # add data2 for the visualization
                analysis_data2[f'displacements'].extend(list(jump_distances))
                analysis_data2[f'state'].extend([sub_trajectory.state.iloc[0]] * len(list(jump_distances)))
                analysis_data2[f'color_code'].extend([color] * len(list(jump_distances)))

    # calculate average of msd and tamsd for each state
    for state_key in total_states:
        msd_mean = []
        msd_std = []
        msd_nb_data = []
        tamsd_mean = []
        tamsd_std = []
        tamsd_nb_data = []
        for t in range(max_frame):
            msd_nb_ = 0
            tamsd_nb_ = 0
            msd_row_data = []
            tamsd_row_data = []
            for row in range(len(msd_ragged_ens_trajs[state_key])):
                if t < len(msd_ragged_ens_trajs[state_key][row]):
                    msd_row_data.append(msd_ragged_ens_trajs[state_key][row][t])
                    msd_nb_ += 1
            for row in range(len(tamsd_ragged_ens_trajs[state_key])):
                if t < len(tamsd_ragged_ens_trajs[state_key][row]):
                    tamsd_row_data.append(tamsd_ragged_ens_trajs[state_key][row][t])
                    tamsd_nb_ += 1
            msd_mean.append(np.mean(msd_row_data))
            msd_std.append(np.std(msd_row_data))
            msd_nb_data.append(msd_nb_)
            tamsd_mean.append(np.mean(tamsd_row_data))
            tamsd_std.append(np.std(tamsd_row_data))
            tamsd_nb_data.append(tamsd_nb_)

        sts = [state_key] * max_frame
        times = np.arange(0, max_frame) * framerate

        msd[f'mean'].extend(msd_mean)
        msd[f'std'].extend(msd_std)
        msd[f'nb_data'].extend(msd_nb_data)
        msd[f'state'].extend(sts)
        msd[f'time'].extend(times)
        tamsd[f'mean'].extend(tamsd_mean)
        tamsd[f'std'].extend(tamsd_std)
        tamsd[f'nb_data'].extend(tamsd_nb_data)
        tamsd[f'state'].extend(sts)
        tamsd[f'time'].extend(times)

    # normalize markov chain
    for edge in state_graph.edges:
        src, dest = edge
        weight = state_graph[src][dest]["weight"]
        state_markov[src][dest] = weight
    state_markov = np.array(state_markov, dtype=np.float64)
    for idx in range(len(total_states)):
        state_markov[idx] /= np.sum(state_markov[idx])


    analysis_data1 = pd.DataFrame(analysis_data1).astype({'state': int, 'duration': float, 'traj_id':str})
    analysis_data2 = pd.DataFrame(analysis_data2)
    msd = pd.DataFrame(msd)
    tamsd = pd.DataFrame(tamsd)

    print('** preprocessing finished **')
    return analysis_data1, analysis_data2, state_markov, state_graph, msd, tamsd, total_states


def simple_preprocessing(data, pixelmicrons, framerate, cutoff, tamsd_calcul=True, color=None):
    traj_indices = pd.unique(data['traj_idx'])
    # initializations
    dim = 2 # will be changed in future.
    max_frame = data.frame.max()

    analysis_data1 = {}
    analysis_data1[f'mean_jump_d'] = []
    analysis_data1[f'duration'] = []
    analysis_data1[f'traj_id'] = []
    analysis_data1[f'color_code'] = []
    analysis_data2 = {}
    analysis_data2[f'displacements'] = []
    analysis_data2[f'color_code'] = []
    msd = {}
    msd[f'mean'] = []
    msd[f'std'] = []
    msd[f'nb_data'] = []
    msd[f'time'] = []
    tamsd = {}
    tamsd[f'mean'] = []
    tamsd[f'std'] = []
    tamsd[f'nb_data'] = []
    tamsd[f'time'] = []
    msd_ragged_ens_trajs = []
    tamsd_ragged_ens_trajs = []
    displace_tmp = []

    data.x = data.x * pixelmicrons
    data.y = data.y * pixelmicrons
    # get data from trajectories
    if tamsd_calcul:
        print("** Computing of Ensemble-averaged TAMSD takes a few minutes **")
    for traj_idx in tqdm(traj_indices, ncols=120, desc=f'Analysis', unit=f'trajectory'):
        single_traj = data.loc[data['traj_idx'] == traj_idx]

        # trajectory length filter condition
        if len(single_traj) >= cutoff:
            # coordinate normalize
            x_coords = np.array(single_traj.x) - float(single_traj.x.iloc[0])
            y_coords = np.array(single_traj.y) - float(single_traj.y.iloc[0])

            # calcultae jump distances
            jump_distances = np.sqrt((x_coords[1:] - x_coords[:-1]) ** 2 + (y_coords[1:] - y_coords[:-1]) ** 2)
            displace_tmp.extend(list(jump_distances))
            # MSD
            msd_ragged_ens_trajs.append((x_coords**2 + y_coords**2) / dim / 2)

            # TAMSD
            if tamsd_calcul:
                tamsd_tmp = []
                for lag in range(len(single_traj)):
                    time_averaged = []
                    for pivot in range(len(single_traj) - lag):
                        time_averaged.append(((x_coords[pivot + lag] - x_coords[pivot]) ** 2 + (y_coords[pivot + lag] - y_coords[pivot]) ** 2) / dim / 2)
                    tamsd_tmp.append(np.mean(time_averaged))
            else:
                tamsd_tmp = [0] * len(single_traj)
            tamsd_ragged_ens_trajs.append(tamsd_tmp)


            # add data1 for the visualization
            analysis_data1[f'mean_jump_d'].append(jump_distances.mean())
            analysis_data1[f'duration'].append((single_traj.frame.iloc[-1] - single_traj.frame.iloc[0] + 1) * framerate)
            analysis_data1[f'traj_id'].append(single_traj.traj_idx.iloc[0])
            analysis_data1[f'color_code'].append(color)

            # add data2 for the visualization
            analysis_data2[f'displacements'].extend(list(jump_distances))
            analysis_data2[f'color_code'].extend([color] * len(list(jump_distances)))

    msd_mean = []
    msd_std = []
    msd_nb_data = []
    tamsd_mean = []
    tamsd_std = []
    tamsd_nb_data = []
    for t in range(max_frame):
        msd_nb_ = 0
        tamsd_nb_ = 0
        msd_row_data = []
        tamsd_row_data = []
        for row in range(len(msd_ragged_ens_trajs)):
            if t < len(msd_ragged_ens_trajs[row]):
                msd_row_data.append(msd_ragged_ens_trajs[row][t])
                msd_nb_ += 1
        for row in range(len(tamsd_ragged_ens_trajs)):
            if t < len(tamsd_ragged_ens_trajs[row]):
                tamsd_row_data.append(tamsd_ragged_ens_trajs[row][t])
                tamsd_nb_ += 1
        msd_mean.append(np.mean(msd_row_data))
        msd_std.append(np.std(msd_row_data))
        msd_nb_data.append(msd_nb_)
        tamsd_mean.append(np.mean(tamsd_row_data))
        tamsd_std.append(np.std(tamsd_row_data))
        tamsd_nb_data.append(tamsd_nb_)

    times = np.arange(0, max_frame) * framerate
    msd[f'mean'].extend(msd_mean)
    msd[f'std'].extend(msd_std)
    msd[f'nb_data'].extend(msd_nb_data)
    msd[f'time'].extend(times)
    tamsd[f'mean'].extend(tamsd_mean)
    tamsd[f'std'].extend(tamsd_std)
    tamsd[f'nb_data'].extend(tamsd_nb_data)
    tamsd[f'time'].extend(times)
    
    analysis_data1 = pd.DataFrame(analysis_data1).astype({'duration': float, 'traj_id':str})
    analysis_data2 = pd.DataFrame(analysis_data2)
    msd = pd.DataFrame(msd)
    tamsd = pd.DataFrame(tamsd)

    print('** preprocessing finished **')
    return analysis_data1, analysis_data2, msd, tamsd

def post(sin):
    px = []
    for i in range(len(sin)):
        px.append(np.mean(np.array(sin)[np.random.RandomState(i).randint(0, len(sin), size=int(8))]))
    return np.array(px)

def get_groundtruth_with_label(folder, label_folder, pixelmicrons, framerate, cutoff, tamsd_calcul=True):
    # load FreeTrace+Bi-ADD data with NaN since we have ground-truth.
    data = DataLoad.read_multiple_h5s(folder)
    label = DataLoad.read_mulitple_andi_labels(label_folder)
    state_tmp = []
    for label_key in label.keys():
        state_tmp.extend(list(label[label_key][:, 2]))
    state_tmp = np.array(state_tmp, dtype=int)

    # using dictionary to convert specific columns
    convert_dict = {'state': int}
    data = data.astype(convert_dict)
    traj_indices = pd.unique(data['traj_idx'])

    # initializations
    dim = 2 # will be changed in future.
    max_frame = data.frame.max()
    total_states = sorted(np.unique(state_tmp)-1) ####
    state_link_for_gt = {st:idx for idx, st in enumerate(total_states)}
    product_states = list(product(total_states, repeat=2))
    state_graph = nx.DiGraph()
    state_graph.add_nodes_from(total_states)
    state_graph.add_edges_from(product_states, weight=0)
    state_markov = [[0 for _ in range(len(total_states))] for _ in range(len(total_states))]
    analysis_data1 = {}
    analysis_data1[f'mean_jump_d'] = []
    analysis_data1[f'K'] = []
    analysis_data1[f'alpha'] = []
    analysis_data1[f'state'] = []
    analysis_data1[f'length'] = []
    analysis_data1[f'traj_id'] = []
    analysis_data2 = {}
    analysis_data2[f'displacements'] = []
    analysis_data2[f'state'] = []
    msd_ragged_ens_trajs = {st:[] for st in total_states}
    tamsd_ragged_ens_trajs = {st:[] for st in total_states}
    msd = {}
    msd[f'mean'] = []
    msd[f'std'] = []
    msd[f'nb_data'] = []
    msd[f'state'] = []
    msd[f'time'] = []
    tamsd = {}
    tamsd[f'mean'] = []
    tamsd[f'std'] = []
    tamsd[f'nb_data'] = []
    tamsd[f'state'] = []
    tamsd[f'time'] = []


    if tamsd_calcul:
        print("** Computing of Ensemble-averaged TAMSD takes a few minutes **")
    for traj_idx in tqdm(traj_indices, ncols=120, desc=f'Analysis', unit=f'trajectory'):
        # read ground-truth and change values to ground-truth
        corresponding_label_key = f'traj_labs_fov_{traj_idx.split("_")[-2]}@{traj_idx.split("_")[-1]}'
        ground_truth = label[corresponding_label_key]
        single_traj = data.loc[data['traj_idx'] == traj_idx].copy()
        before_cp = 0
        Ks = []
        alphas = []
        states = []
        for K, alpha, state, cp in ground_truth:
            state = int(state)%2 ####
            cp = int(cp)
            padded_K = [K] * (cp - before_cp)
            padded_alpha = [alpha] * (cp - before_cp)
            padded_state = [state] * (cp - before_cp)
            Ks.extend(padded_K)
            alphas.extend(padded_alpha)
            states.extend(padded_state)
            before_cp = cp
        single_traj['state'] = states
        single_traj['alpha'] = alphas
        single_traj['K'] = Ks

        # calculate state changes inside single trajectory
        before_st = single_traj.state.iloc[0]
        for st in single_traj.state:
            state_graph[before_st][st]['weight'] += 1
            before_st = st

        # chucnk into sub-trajectories
        before_st = single_traj.state.iloc[0]
        chunk_idx = [0, len(single_traj)]
        for st_idx, st in enumerate(single_traj.state):
            if st != before_st:
                chunk_idx.append(st_idx)
            before_st = st
        chunk_idx = sorted(chunk_idx)

        for i in range(len(chunk_idx) - 1):
            sub_trajectory = single_traj.iloc[chunk_idx[i]:chunk_idx[i+1]].copy()

            # trajectory length filter condition
            if len(sub_trajectory) >= cutoff:
                # state of trajectory
                state = sub_trajectory.state.iloc[0]
                bi_add_alpha = sub_trajectory.alpha.iloc[0]
                bi_add_K = sub_trajectory.K.iloc[0]

                # convert from pixel-coordinate to micron.
                sub_trajectory.x *= pixelmicrons
                sub_trajectory.y *= pixelmicrons
                sub_trajectory.z *= pixelmicrons ## need to check
                sub_trajectory.K *= (pixelmicrons**2/framerate**bi_add_alpha) #TODO: check again

                # coordinate normalize
                sub_trajectory.x -= sub_trajectory.x.iloc[0]
                sub_trajectory.y -= sub_trajectory.y.iloc[0]
                sub_trajectory.z -= sub_trajectory.z.iloc[0]

                # calcultae jump distances
                jump_distances = np.sqrt((sub_trajectory.x.iloc[1:].to_numpy() - sub_trajectory.x.iloc[:-1].to_numpy()) ** 2 + (sub_trajectory.y.iloc[1:].to_numpy() - sub_trajectory.y.iloc[:-1].to_numpy()) ** 2)


                # MSD
                msd_ragged_ens_trajs[state].append(((sub_trajectory.x.to_numpy())**2 + (sub_trajectory.y.to_numpy())**2) / dim / 2)

                # TAMSD
                if tamsd_calcul:
                    tamsd_tmp = []
                    for lag in range(len(sub_trajectory)):
                        time_averaged = []
                        for pivot in range(len(sub_trajectory) - lag):
                            time_averaged.append(((sub_trajectory.x.iloc[pivot + lag] - sub_trajectory.x.iloc[pivot]) ** 2 + (sub_trajectory.y.iloc[pivot + lag] - sub_trajectory.y.iloc[pivot]) ** 2) / dim / 2)
                        tamsd_tmp.append(np.mean(time_averaged))
                else:
                    tamsd_tmp = [0] * len(sub_trajectory)
                tamsd_ragged_ens_trajs[state].append(tamsd_tmp)


                # add data for the visualization
                analysis_data1[f'mean_jump_d'].append(jump_distances.mean())
                analysis_data1[f'K'].append(bi_add_K)
                analysis_data1[f'alpha'].append(bi_add_alpha)
                analysis_data1[f'state'].append(state)
                analysis_data1[f'length'].append((sub_trajectory.frame.iloc[-1] - sub_trajectory.frame.iloc[0] + 1) * framerate)
                analysis_data1[f'traj_id'].append(sub_trajectory.traj_idx.iloc[0])

                analysis_data2[f'displacements'].extend(list(jump_distances))
                analysis_data2[f'state'].extend([sub_trajectory.state.iloc[0]] * len(list(jump_distances)))

    # calculate average of msd and tamsd for each state
    for state_key in total_states:
        msd_mean = []
        msd_std = []
        msd_nb_data = []
        tamsd_mean = []
        tamsd_std = []
        tamsd_nb_data = []
        for t in range(max_frame):
            msd_nb_ = 0
            tamsd_nb_ = 0
            msd_row_data = []
            tamsd_row_data = []
            for row in range(len(msd_ragged_ens_trajs[state_key])):
                if t < len(msd_ragged_ens_trajs[state_key][row]):
                    msd_row_data.append(msd_ragged_ens_trajs[state_key][row][t])
                    msd_nb_ += 1
            for row in range(len(tamsd_ragged_ens_trajs[state_key])):
                if t < len(tamsd_ragged_ens_trajs[state_key][row]):
                    tamsd_row_data.append(tamsd_ragged_ens_trajs[state_key][row][t])
                    tamsd_nb_ += 1
            msd_mean.append(np.mean(msd_row_data))
            msd_std.append(np.std(msd_row_data))
            msd_nb_data.append(msd_nb_)
            tamsd_mean.append(np.mean(tamsd_row_data))
            tamsd_std.append(np.std(tamsd_row_data))
            tamsd_nb_data.append(tamsd_nb_)

        sts = [state_key] * max_frame
        times = np.arange(0, max_frame) * framerate

        msd[f'mean'].extend(msd_mean)
        msd[f'std'].extend(msd_std)
        msd[f'nb_data'].extend(msd_nb_data)
        msd[f'state'].extend(sts)
        msd[f'time'].extend(times)
        tamsd[f'mean'].extend(tamsd_mean)
        tamsd[f'std'].extend(tamsd_std)
        tamsd[f'nb_data'].extend(tamsd_nb_data)
        tamsd[f'state'].extend(sts)
        tamsd[f'time'].extend(times)

    # normalize markov chain
    for edge in state_graph.edges:
        src, dest = edge
        weight = state_graph[src][dest]["weight"]
        state_markov[state_link_for_gt[src]][state_link_for_gt[dest]] = weight
    state_markov = np.array(state_markov, dtype=np.float64)
    for idx in range(len(total_states)):
        state_markov[idx] /= np.sum(state_markov[idx])

    analysis_data1 = pd.DataFrame(analysis_data1).astype({'state': int, 'length': float, 'traj_id':str})
    analysis_data2 = pd.DataFrame(analysis_data2)
    msd = pd.DataFrame(msd)
    tamsd = pd.DataFrame(tamsd)

    print('** preprocessing finished **')
    return analysis_data1, analysis_data2, state_markov, state_graph, msd, tamsd, total_states
