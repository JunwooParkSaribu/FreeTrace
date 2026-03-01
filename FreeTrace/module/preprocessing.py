import numpy as np
import pandas as pd
import networkx as nx
from itertools import product, permutations
from FreeTrace.module import data_load, data_save
from tqdm import tqdm
import math
import cv2
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib as mpl
import matplotlib.pyplot as plt



def simple_preprocessing(data, pixelmicrons, framerate, cutoff=[3, 99999], selected_state=0, div_time_gap=True, tamsd_calcul=True, color_palette = ['red','cyan','green','blue','gray','pink']):
    # load FreeTrace+Bi-ADD data without NaN (NaN where trajectory length is shorter than 5, default in BI-ADD)
    assert 'yellow' not in color_palette, "Yellow can't be included in the color palette, please exclude yellow."
    data = data.dropna()
    # using dictionary to convert specific columns
    convert_dict = {'state': int}
    data = data.astype(convert_dict)
    traj_indices = pd.unique(data['traj_idx'])
    total_states = sorted(data['state'].unique())

    if len(data) == 0:
        print("*** Data is empty. ***")
        return
    
    if selected_state not in total_states:
        print("*** Selected state is not found in the data. Please try other state numbers. ***")
        return
    
    if len(total_states) == 1:
        color_palette = ['red']

    # state re-ordering w.r.t. K
    avg_ks = []
    for st in total_states:
        avg_ks.append(data['K'][data['state']==st].mean())
    avg_ks = np.array(avg_ks)
    prev_states = np.argsort(avg_ks)
    state_reorder = {st:idx for idx, st in enumerate(prev_states)}
    ordered_states = np.empty(len(data['state']), dtype=np.uint8)
    for st_idx in range(len(ordered_states)):
        ordered_states[st_idx] = state_reorder[data['state'].iloc[st_idx]]
    data['state'] = ordered_states

    # initializations
    dim = 2 # will be changed in future.
    max_frame = data.frame.max()

    product_states = list(product(total_states, repeat=2))
    state_graph = nx.DiGraph()
    state_graph.add_nodes_from(total_states)
    state_graph.add_edges_from(product_states, count=0, freq=0)
    
    if len(total_states) > 1:
        state_changing_duration = {tuple(state): [] for state in product_states}
    else:
        state_changing_duration = {tuple([total_states[0], total_states[0]]):[]}

    state_markov = [[0 for _ in range(len(total_states))] for _ in range(len(total_states))]
    analysis_data1 = {}
    analysis_data1[f'mean_jump_d'] = []
    analysis_data1[f'K'] = []
    analysis_data1[f'H'] = []
    analysis_data1[f'state'] = []
    analysis_data1[f'duration'] = []
    analysis_data1[f'traj_id'] = []
    analysis_data1[f'color'] = []

    analysis_data2 = {}
    analysis_data2[f'2d_displacement'] = []
    analysis_data2[f'state'] = []

    analysis_data3 = {}
    analysis_data3[f'angle'] = []
    analysis_data3[f'state'] = []

    analysis_data4 = {}
    for st in total_states:
        analysis_data4[st] = [[] for _ in range(9999)]
        analysis_data4[st][0].append(0)

    analysis_data5 = {}
    analysis_data5[f'1d_ratio'] = []
    analysis_data5[f'state'] = []

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

    # get data from trajectories
    if tamsd_calcul:
        print("Computing TAMSD... This takes while depending on the length and number of trajectories.")
        
    for traj_idx in tqdm(traj_indices, ncols=120, desc=f'Analysis', unit=f'trajectory'):
        single_traj = data.loc[data['traj_idx'] == traj_idx].copy()
        single_traj = single_traj.sort_values(by=['frame'])

        # chunk into sub-trajectories by frame gap
        before_st = single_traj.state.iloc[0]
        frame_gaps = np.diff(single_traj.frame.to_numpy())
        time_chunk_idx = [0, len(single_traj)]
        if div_time_gap:
            for f_idx, frame_gap in enumerate(frame_gaps):
                if frame_gap > 1.0:
                    time_chunk_idx.append(f_idx+1)
            time_chunk_idx = sorted(time_chunk_idx)

        for i in range(len(time_chunk_idx) - 1):
            time_chunked_single_traj = single_traj.iloc[time_chunk_idx[i]:time_chunk_idx[i+1]].copy()

            # chunk into sub-trajectories by state
            before_st = time_chunked_single_traj.state.iloc[0]
            state_chunk_idx = [0, len(time_chunked_single_traj)]
            for st_idx, st in enumerate(time_chunked_single_traj.state):
                if st != before_st:
                    state_chunk_idx.append(st_idx)
                before_st = st
            state_chunk_idx = sorted(state_chunk_idx)

            for i in range(len(state_chunk_idx) - 1):
                sub_trajectory = time_chunked_single_traj.iloc[state_chunk_idx[i]:state_chunk_idx[i+1]].copy()
                # trajectory length filter condition
                if cutoff[0] <= len(sub_trajectory) <= cutoff[1]:
                    # state of trajectory
                    state = sub_trajectory.state.iloc[0]
                    fr_H = sub_trajectory.H.iloc[0]
                    fr_K = sub_trajectory.K.iloc[0]

                    # convert from pixel-coordinate to micron.
                    sub_trajectory.x *= pixelmicrons
                    sub_trajectory.y *= pixelmicrons
                    sub_trajectory.z *= pixelmicrons 
                    #fr_K *= ((pixelmicrons**2)) #/ (framerate**fr_H))

                    frame_diffs = sub_trajectory.frame.iloc[1:].to_numpy() - sub_trajectory.frame.iloc[:-1].to_numpy()
                    duration = np.sum(frame_diffs) * framerate

                    if len(state_chunk_idx) == 2:
                        state_graph[state][state]['count'] += 1
                    else:
                        if i >= 1:
                            state_graph[prev_state][state]['count'] += 1
                            state_changing_duration[tuple([prev_state, state])].append(prev_duration)
                        if i == len(state_chunk_idx) - 2:  # last state of chunked trjectory which is considered as non-state changing stae.
                            state_graph[state][state]['count'] += 1 
                    prev_state = state
                    prev_duration = duration
                
                    # coordinate rescale
                    sub_trajectory.x -= sub_trajectory.x.iloc[0]
                    sub_trajectory.y -= sub_trajectory.y.iloc[0]
                    sub_trajectory.z -= sub_trajectory.z.iloc[0]

                    # calcultae jump distances (assumption of Brownian particle if the trajectory is not divided by the gap in frames)               
                    jump_distances = (np.sqrt(((sub_trajectory.x.iloc[1:].to_numpy() - sub_trajectory.x.iloc[:-1].to_numpy()) ** 2) / (sub_trajectory.frame.iloc[1:].to_numpy() - sub_trajectory.frame.iloc[:-1].to_numpy())
                                            + ((sub_trajectory.y.iloc[1:].to_numpy() - sub_trajectory.y.iloc[:-1].to_numpy()) ** 2) / (sub_trajectory.frame.iloc[1:].to_numpy() - sub_trajectory.frame.iloc[:-1].to_numpy()) )) 
                
                    # angles
                    x_vec = (sub_trajectory.x.iloc[1:].to_numpy() - sub_trajectory.x.iloc[:-1].to_numpy())
                    y_vec = (sub_trajectory.y.iloc[1:].to_numpy() - sub_trajectory.y.iloc[:-1].to_numpy())
                    vecs = np.vstack([x_vec, y_vec]).T
                    angles = [dot_product_angle(vecs[vec_idx], vecs[vec_idx+1]) for vec_idx in range(len(vecs) - 1)]
                    ratios = np.array([])
                    for l in range(1, 2):
                        r_tmp = np.concatenate([(x_vec[l::l] / x_vec[:-l:l]), (y_vec[l::l] / y_vec[:-l:l])])
                        ratios = np.concatenate((ratios, r_tmp))
                    #ratios = [0 if math.isnan(val) or math.isinf(val) else val for val in ratios]

                    # Ensemble averaged SD
                    copy_frames = sub_trajectory.frame.to_numpy()
                    copy_frames = copy_frames - copy_frames[0]
                    tmp_msd = []
                    for frame, sq_disp in zip(np.arange(0, copy_frames[-1], 1), ((sub_trajectory.x.to_numpy())**2 + (sub_trajectory.y.to_numpy())**2) / dim / 2):
                        if frame in copy_frames:
                            tmp_msd.append(sq_disp)
                        else:
                            tmp_msd.append(None)
                    msd_ragged_ens_trajs[state].append(tmp_msd)


                    # TAMSD
                    if tamsd_calcul:
                        tamsd_tmp = []
                        for lag in range(len(sub_trajectory)):
                            if lag == 0:
                                tamsd_tmp.append(0)
                            else:
                                time_averaged = []
                                for pivot in range(len(sub_trajectory) - lag):
                                    if lag == (sub_trajectory.frame.iloc[pivot + lag] - sub_trajectory.frame.iloc[pivot]):
                                        time_averaged.append(((sub_trajectory.x.iloc[pivot + lag] - sub_trajectory.x.iloc[pivot]) ** 2 + (sub_trajectory.y.iloc[pivot + lag] - sub_trajectory.y.iloc[pivot]) ** 2) / dim / 2)
                                        analysis_data4[state][lag].append((sub_trajectory.x.iloc[pivot + lag] - sub_trajectory.x.iloc[pivot]))
                                        analysis_data4[state][lag].append((sub_trajectory.y.iloc[pivot + lag] - sub_trajectory.y.iloc[pivot]))
                                if len(time_averaged) > 0:
                                    tamsd_tmp.append(np.mean(time_averaged))
                                else:
                                    tamsd_tmp.append(None)
                    else:
                        tamsd_tmp = [0] * len(sub_trajectory)
                    tamsd_ragged_ens_trajs[state].append(tamsd_tmp)


                    if len(state_chunk_idx) > 2:
                        analysis_data1[f'color'].append("yellow")
                    else:
                        analysis_data1[f'color'].append(color_palette[state])

                    # add data1 for the visualization
                    analysis_data1[f'mean_jump_d'].append(jump_distances.mean())
                    analysis_data1[f'K'].append(fr_K)
                    analysis_data1[f'H'].append(fr_H)
                    analysis_data1[f'state'].append(state)
                    analysis_data1[f'duration'].append(duration)
                    analysis_data1[f'traj_id'].append(sub_trajectory.traj_idx.iloc[0])


                    # add data2 for the visualization
                    analysis_data2[f'2d_displacement'].extend(list(jump_distances))
                    analysis_data2[f'state'].extend([sub_trajectory.state.iloc[0]] * len(list(jump_distances)))

                    # add data3 for angles
                    analysis_data3[f'angle'].extend(list(angles))
                    analysis_data3[f'state'].extend([sub_trajectory.state.iloc[0]] * len(list(angles)))

                    # add data5 for ratio (consecutive displacements)
                    analysis_data5[f'1d_ratio'].extend(list(ratios))
                    analysis_data5[f'state'].extend([sub_trajectory.state.iloc[0]] * len(list(ratios)))



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
                if t < len(msd_ragged_ens_trajs[state_key][row]) and msd_ragged_ens_trajs[state_key][row][t] is not None:
                    msd_row_data.append(msd_ragged_ens_trajs[state_key][row][t])
                    msd_nb_ += 1
            for row in range(len(tamsd_ragged_ens_trajs[state_key])):
                if t < len(tamsd_ragged_ens_trajs[state_key][row]) and tamsd_ragged_ens_trajs[state_key][row][t] is not None:
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
        weight = state_graph[src][dest]["count"]
        state_markov[src][dest] = weight
    state_markov = np.array(state_markov, dtype=np.float64)
    for idx in range(len(total_states)):
        state_markov[idx] /= np.sum(state_markov[idx])
    for edge in state_graph.edges:
        src, dest = edge
        state_graph[src][dest]['freq'] = np.round(state_markov[src][dest], 3)


    analysis_data1 = pd.DataFrame(analysis_data1).astype({'state': int, 'duration': float, 'traj_id':str})
    analysis_data2 = pd.DataFrame(analysis_data2)
    analysis_data3 = pd.DataFrame(analysis_data3)
    analysis_data5 = pd.DataFrame(analysis_data5)
    msd = pd.DataFrame(msd)
    tamsd = pd.DataFrame(tamsd)

    if tamsd_calcul == False:
        tamsd = None
        
    print('** preprocessing finished **')
    return analysis_data1, analysis_data2, analysis_data3, analysis_data4, analysis_data5, state_markov, state_graph, msd, tamsd, total_states, state_changing_duration


def count_cumul_trajs_with_roi(data:pd.DataFrame|str, roi_file:str|None, start_frame=1, end_frame=100, cutoff=5):
    from roifile import ImagejRoi
    """
    Cropping trajectory result with ROI(region of interest) or frames.
    It returns a list which contains the number of accumulated trajectories, only considering the first positions and time for each trajectory.

    data: .h5 or equivalent format of trajectory result file.
    roi_file: region_of_interest.roi which containes the roi information in pixel.
    start_frame: start frame to crop the trajectory result with frame.
    end_frame: end frame to crop the trajectory result with frame.
    """
    assert "_traces.csv" in data or type(data) is pd.DataFrame or ".h5" in data, "The input filename should be .h5 or traces.csv which are the results of FreeTrace or BI-ADD or equilvalent format."
    assert end_frame > start_frame, "The number of end frame must be greater than start frame."
    

    if roi_file is None:
        contours = None
    elif type(roi_file) is str and len(roi_file) == 0:
        contours = None
    else:
        contours = ImagejRoi.fromfile(roi_file).coordinates().astype(np.int32)
    
    trajectory_counts = []
    cumul=1
    coords = {t:[] for t in np.arange(start_frame, end_frame+1)}
    time_steps = np.arange(start_frame, end_frame, 1)
    observed_max_t = -1
    observed_min_t = 99999999

    if type(data) is str and '.h5' in data:
        data = data_load.read_multiple_h5s(path=data)
        traj_indices = data['traj_idx'].unique()
        for traj_idx in traj_indices:
            single_traj = data[data['traj_idx'] == traj_idx]
            # considers only the first position.
            x = single_traj['x'].iloc[0]
            y = single_traj['y'].iloc[0]
            t = single_traj['frame'].iloc[0]
            if t in coords and len(single_traj) >= cutoff:
                coords[t].append([x, y])
            observed_max_t = max(t, observed_max_t)
            observed_min_t = min(t, observed_min_t)

    elif type(data) is str and '_traces.csv' in data:
        trajectory_list = data_load.read_trajectory(data)
        for trajectory in trajectory_list:
            xyz = trajectory.get_positions()
            x = xyz[:, 0][0]
            y = xyz[:, 1][0]
            t = int(trajectory.get_times()[0])
            if t in coords and len(xyz) >= cutoff:
                coords[t].append([x, y])
            observed_max_t = max(t, observed_max_t)
            observed_min_t = min(t, observed_min_t)
    else:
        traj_indices = data['traj_idx'].unique()
        for traj_idx in traj_indices:
            single_traj = data[data['traj_idx'] == traj_idx]
            # considers only the first position.
            x = single_traj['x'].iloc[0]
            y = single_traj['y'].iloc[0]
            t = single_traj['frame'].iloc[0]
            if t in coords and len(single_traj) >= cutoff:
                coords[t].append([x, y])
            observed_max_t = max(t, observed_max_t)
            observed_min_t = min(t, observed_min_t)
                

    for t in tqdm(time_steps, ncols=120, desc=f'Accumulation', unit=f'frame'):
        if t == start_frame:
            st_tmp = []
            for stack_t in range(t, t+cumul):
                time_st = []
                if stack_t in time_steps:
                    for stack_coord in coords[stack_t]:
                        if roi_file is not None:
                            x, y = stack_coord
                            masked = cv2.pointPolygonTest(contours, (x, y), False)
                            if masked == 1:
                                time_st.append(stack_coord)
                        else:
                            time_st.append(stack_coord)
                st_tmp.append(time_st)
            traj_count = np.sum([len(x) for x in st_tmp])
            trajectory_counts.append(traj_count)
            prev_tmps=st_tmp
        else:
            stack_t = t+cumul-1
            time_st = []
            if stack_t in time_steps:
                for stack_coord in coords[stack_t]:

                    if roi_file is not None:
                        x, y = stack_coord
                        masked = cv2.pointPolygonTest(contours, (x, y), False)
                        if masked == 1:
                            time_st.append(stack_coord)
                    else:
                        time_st.append(stack_coord)

            st_tmp = prev_tmps[1:]
            st_tmp.append(time_st)
            traj_count = np.sum([len(x) for x in st_tmp])
            trajectory_counts.append(traj_count)
            prev_tmps = st_tmp

    return trajectory_counts, np.cumsum(trajectory_counts)


def check_roi_passing_traces(h5_file:str, roi_file:str|None):
    assert ".h5" in h5_file, "Wrong trajectory file format, .h5 extendsion is needed."
    assert roi_file is not None, "This needs ROI file."

    from roifile import ImagejRoi
    contours = ImagejRoi.fromfile(roi_file).coordinates().astype(np.int32)

    stay_inside_trajectories = []
    out_to_in_trajectories = []
    in_to_out_trajectories = []
    complex_trajectories = []
    df = data_load.read_h5(h5_file)[0]
    traj_indices = df['traj_idx'].unique()

    for traj_idx in traj_indices:
        single_traj = df[df['traj_idx'] == traj_idx]
        xs = single_traj['x'].to_numpy()
        ys = single_traj['y'].to_numpy()
        masks = []

        for x, y in zip(xs, ys):
            masked = int(cv2.pointPolygonTest(contours, (x, y), False))
            masks.append(masked)
        
        if -1 in masks:
            if masks[0] == -1 and masks[-1] == 1:
                out_to_in_trajectories.append(traj_idx)
            elif masks[0] == 1 and masks[-1] == -1:
                in_to_out_trajectories.append(traj_idx)
            else:
                complex_trajectories.append(traj_idx)
        else:
            stay_inside_trajectories.append(traj_idx)
        
    print(f"Filename: {h5_file}\nROIname: {roi_file}")
    print(f'Nb of total trajectories: {len(traj_indices)}')
    print(f'Nb of Out-to-In trajectories: {len(out_to_in_trajectories)},  ratio: {np.round(len(out_to_in_trajectories) / len(traj_indices),3)}')
    print(f'Nb of In-to-Out trajectories: {len(in_to_out_trajectories)},  ratio: {np.round(len(in_to_out_trajectories) / len(traj_indices),3)}')
    print(f'Nb of Complex trajectories: {len(complex_trajectories)},  ratio: {np.round(len(complex_trajectories) / len(traj_indices),3)}')
    print(f'')


def linear_fit(msd:pd.DataFrame, timepoints:dict, states:list):
    assert len(timepoints) == len(states), "The number of timepoints and states should be same."
    slopes = {state:[None, None] for state in states}
    for state in states:
        timepoint = timepoints[state]
        times = msd[msd['state']==state]['time'].to_numpy()
        means = msd[msd['state']==state]['mean'].to_numpy()
        selected_times = times[times <= timepoint]
        selected_means = means[times <= timepoint]
        slope, b = np.polyfit(selected_times, selected_means, 1)
        slopes[state] = [slope, b]
    return slopes


def diffusion_coefficient(msd:pd.DataFrame, timepoints:dict, states:list):
    assert len(timepoints) == len(states), "The number of timepoints and states should be same."
    diff_coef_state = {state:None for state in states}
    for state in states:
        timepoint = timepoints[state]
        diff_coef = []
        times = msd[msd['state']==state]['time'].to_numpy()
        means = msd[msd['state']==state]['mean'].to_numpy()
        selected_times = times[times <= timepoint]
        selected_means = means[times <= timepoint]
        for lag in range(1, len(selected_means)):
            time_lag = selected_times[lag]
            diff_coef_lag = []
            for pivot in range(0, len(selected_means) - lag):
                diff_coef_lag.append(abs(selected_means[lag+pivot] - selected_means[pivot]))
            diff_coef.append(np.mean(diff_coef_lag) / time_lag)
        diff_coef_state[state] = np.round(np.mean(diff_coef), 4)
    return diff_coef_state


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


def dot_product_angle(v1, v2):
    ang = angle_between(v1, v2)
    if ang == np.inf or ang == np.nan or math.isnan(ang):
        return 0
    return 180 - ang


def pdf_cauchy(u, h, t=1, s=1):
    assert 0 < h < 1
    return 1/(np.pi * np.sqrt(1-(2**(2*h-1)-1)**2)) * 1 / ( (u - (2**(2*h-1) - 1)*(t/s)**h)**2 / ((1 - (2**(2*h-1)-1)**2)*np.sqrt((t/s)**(2*h))) + (np.sqrt((t/s)**(2*h))))


def cdf_cauchy(u, h, t=1, s=1):
    assert 0 < h < 1
    val = np.arctan( (2*(s**h)*u + (t**h)*(2 - 2**(2*h))) / ((t**h) * np.sqrt(2**(2*h+2) - 2**(4*h))) ) / np.pi + 0.5
    return val


def inv_cdf_cauchy(proba, h, t=1, s=1):
    assert 0 < h < 1
    denom = ((t**h) * np.sqrt(2**(2*h+2) - 2**(4*h)))
    a = (2*(s**h)) / denom
    return (np.tan(np.pi * proba - np.pi * 0.5) / a) - ((t**h)*(2 - 2**(2*h)) / 2*(s**h))


def pdf_cauchy_2mixture(x, h1, h2, alpha, beta):
    return beta*(alpha*pdf_cauchy(x, h1) + (1-alpha)*pdf_cauchy(x, h2))


def cdf_cauchy_2mixture(x, h1, h2, alpha, beta):
    return beta*(alpha*cdf_cauchy(x, h1) + (1-alpha)*cdf_cauchy(x, h2))


def pdf_cauchy_3mixture(x, h1, h2, h3, alpha1, alpha2, alpha3, beta):
    return beta*(alpha1*pdf_cauchy(x, h1) + (alpha2)*pdf_cauchy(x, h2) + (alpha3)*pdf_cauchy(x, h3))


def cdf_cauchy_1mixture(x, h1, alpha):
    return alpha*cdf_cauchy(x, h1)


def pdf_cauchy_1mixture(x, h1, alpha):
    return alpha*pdf_cauchy(x, h1)


def cauchy_location(h1, t=1, s=1):
    return (2**(2*h1-1) - 1)*(t/s)**h1


def func_to_fit(func, x, params):
    return func(x, *params)


def func_to_minimise(params, func, x, y):
    y_pred = func_to_fit(func, x, params)
    return np.sum(abs(y_pred - y))


def trajectory_visualization(original_data:pd.DataFrame, analysis_data1:pd.DataFrame, 
                             cutoff:list, pixelmicron:float, resolution_multiplier=80, 
                             roi='', scalebar=True, arrow=False, color_for_roi=False, thickness = 3) -> np.ndarray:
    print("** visualizing trajectories... **")
    scale = resolution_multiplier
    color_maps = {}
    color_maps_plot = {}
    roi_center = None
    angle_circle_radius = 10

    min_x = original_data['x'].min()
    min_y = original_data['y'].min()
    max_x = original_data['x'].max()
    max_y = original_data['y'].max()
    x_width = int(((max_x - min_x) * scale))
    y_width = int(((max_y - min_y) * scale))
    traj_image = np.ones((y_width, x_width, 4)).astype(np.uint8)
    angle_image = np.ones((y_width, x_width, 4)).astype(np.uint8)
    angle_cmap = plt.get_cmap('jet')

    if len(roi) > 4:
        from roifile import ImagejRoi
        contours = ImagejRoi.fromfile(roi).coordinates().astype(np.int32)
        roi_center = (np.array([contours[0][0] - min_x, contours[0][1] - min_y])*scale).astype(int)
        for i in range(len(contours) - 1):
            prev_pt = (np.array([contours[i][0] - min_x, contours[i][1] - min_y])*scale).astype(int)
            next_pt = (np.array([contours[i+1][0] - min_x, contours[i+1][1] - min_y])*scale).astype(int)
            cv2.circle(traj_image, next_pt, thickness+1, (128, 128, 128, 255), -1)
            cv2.circle(angle_image, next_pt, thickness+1, (128, 128, 128, 255), -1)
            roi_center += next_pt
        roi_center = roi_center.astype(float)
        roi_center /= len(contours)

    for traj_idx in tqdm(original_data['traj_idx'].unique(), ncols=120, desc=f'Visualization', unit='trajectory'):
        single_traj = original_data[original_data['traj_idx'] == traj_idx]
        corresponding_ids = analysis_data1['traj_id'] == single_traj['traj_idx'].iloc[0]
        if np.sum(corresponding_ids) > 0:
            traj_color = analysis_data1[analysis_data1['traj_id'] == single_traj['traj_idx'].iloc[0]]['color'].iloc[0]
            state = analysis_data1[analysis_data1['traj_id'] == single_traj['traj_idx'].iloc[0]]['state'].iloc[0]
            if traj_color not in color_maps and traj_color != 'yellow':
                color_maps[traj_color] = str(state)
                color_maps_plot[state] = traj_color
            if cutoff[0] <= len(single_traj) <= cutoff[1]:
                traj_color = mcolors.to_rgb(traj_color)
                traj_color = (int(traj_color[2]*255), int(traj_color[1]*255), int(traj_color[0]*255), 255)  # BGR color for cv2
                pts = np.array([[int((x - min_x) * scale), int((y - min_y) * scale)] for x, y in zip(single_traj['x'], single_traj['y'])], np.int32)
                for i in range(len(pts)-1):
                    prev_pt = pts[i]
                    next_pt = pts[i+1]
                    if arrow:
                        if color_for_roi and roi_center is not None:
                            if np.sqrt(np.sum((next_pt - roi_center)**2)) < np.sqrt(np.sum((prev_pt - roi_center)**2)):
                                cv2.arrowedLine(traj_image, prev_pt, next_pt, (0, 0, 255, 255), thickness)
                            else:
                                cv2.arrowedLine(traj_image, prev_pt, next_pt, (0, 255, 255, 255), thickness)
                            """
                            from module.preprocessing import dot_product_angle
                            angle = dot_product_angle(next_pt - prev_pt, roi_center - prev_pt)
                            if 0 < angle < 90 or 270 < angle < 360:
                                cv2.arrowedLine(image, prev_pt, next_pt, (0, 0, 255), thickness)
                            else:
                                cv2.arrowedLine(image, prev_pt, next_pt, (0, 255, 255), thickness)
                            """
                        else:
                            cv2.arrowedLine(traj_image, prev_pt, next_pt, traj_color, thickness)
                    else:
                        cv2.line(traj_image, prev_pt, next_pt, traj_color, thickness)

                for i in range(len(pts)-2):
                    prev_vec = pts[i+1] - pts[i]
                    next_vec = pts[i+2] - pts[i+1]
                    angle_degree = dot_product_angle(prev_vec, next_vec)
                    angle_color = angle_cmap(angle_degree / 180)
                    cv2.circle(angle_image, center=(pts[i+1][0], pts[i+1][1]), radius=angle_circle_radius,
                               color=(int(angle_color[2]*255), int(angle_color[1]*255), int(angle_color[0]*255), 200), thickness=-1)
                    
    if scalebar:
        cv2.line(traj_image, [int(max(0, x_width - 2*scale - int(scale/pixelmicron))), int(max(0, y_width - 2*scale))], [int(max(0, x_width - 2*scale)) , int(max(0, y_width - 2*scale))], (255, 255, 255, 255), 6)
        cv2.line(angle_image, [int(max(0, x_width - 2*scale - int(scale/pixelmicron))), int(max(0, y_width - 2*scale))], [int(max(0, x_width - 2*scale)) , int(max(0, y_width - 2*scale))], (255, 255, 255, 255), 6)
            
    traj_image = cv2.cvtColor(traj_image, cv2.COLOR_BGRA2RGBA)
    color_maps['yellow'] = 'transitioning'
    color_maps_plot['transitioning'] = 'yellow'
    patches = [mpatches.Patch(color=c,label=color_maps[c]) for c in color_maps]
    return traj_image, angle_image, patches, color_maps, color_maps_plot