import os
import sys
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import lru_cache
from itertools import product
import networkx as nx
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.model_selection import GridSearchCV
from FreeTrace.module import cost_function
from FreeTrace.module.trajectory_object import TrajectoryObj
from FreeTrace.module.image_module import read_tif, make_image_seqs, make_whole_img
from FreeTrace.module.data_save import write_trajectory
from FreeTrace.module.data_load import read_localization
from FreeTrace.module.auxiliary import initialization
from FreeTrace.module.xml_module import write_xml


@lru_cache
def pdf_mu_measure(alpha):
    idx = int((alpha / POLY_FIT_DATA['alpha'][-1]) * (len(POLY_FIT_DATA['alpha']) - 1))
    return POLY_FIT_DATA['mu'][idx]


@lru_cache
def indice_fetch(alpha, k):
    alpha_index = len(STD_FIT_DATA['alpha_space']) - 1
    k_index = len(STD_FIT_DATA['logD_space']) -1
    for alpha_i, reg_alpha in enumerate(STD_FIT_DATA['alpha_space']):
        if alpha < reg_alpha:
            alpha_index = alpha_i
            break
    for k_i, reg_k in enumerate(STD_FIT_DATA['logD_space']):
        if k < reg_k:
            k_index = k_i
            break
    return alpha_index, k_index


@lru_cache
def std_fetch(alpha_i, k_i):
    return STD_FIT_DATA['std_grid'][alpha_i, k_i]


def predict_multinormal(relativ_coord, alpha, k, lag):
    sigma_ = 4.5
    abnormal = False
    alpha = 1.0
    k=1.0
    alpha_index, k_index = indice_fetch(alpha, k)
    mean_std = std_fetch(alpha_index, k_index)

    if np.sqrt(np.sum(relativ_coord**2)) > sigma_*mean_std:
        abnormal = True
    
    relativ_coord = relativ_coord[:DIMENSION]
    log_pdf = np.sum(np.log(1/(np.sqrt(2*np.pi) * mean_std) * np.exp(-1./2 * (relativ_coord / mean_std)**2)))
    return log_pdf, abnormal


"""
def predict_cauchy(next_vec, prev_vec, alpha, lag, precision, dimension):
    log_pdf = 0
    abnormal = False
    delta_t = lag + 1

    for vec1, vec2 in zip(next_vec[:DIMENSION], prev_vec[:DIMENSION]):
        if vec2 < 0:
            vec2 -= precision
        else:
            vec2 += precision
        coord_ratio = vec1 / vec2

        if 0.95 < alpha < 1.05:
            if abs(coord_ratio) > 10: ## TODO
                abnormal = True
            log_pdf += math.log( 1/math.pi * 1/((coord_ratio)**2 + 1) )

        else:
            rho = 1/2. * ((delta_t-1)**alpha - 2*delta_t**alpha + (delta_t+1)**alpha)
            relativ_cov = 1/2. * ((delta_t+1)**alpha - (delta_t)**alpha - (1)**alpha)
            scale = math.sqrt(abs(1-rho**2)) ## TODO
            if abs(coord_ratio-relativ_cov) > 10: ## TODO
                abnormal = True
            log_pdf += math.log( 1/(math.pi * scale) * 1 / ( ((coord_ratio - relativ_cov)/scale)**2 * (rho/relativ_cov) + (relativ_cov/rho) ) )

    return log_pdf, abnormal
"""


def greedy_shortest(srcs, dests):
    srcs = np.array(srcs)
    dests = np.array(dests)
    x_distribution = []
    y_distribution = []
    z_distribution = []
    superposed_locals = dests
    superposed_len = len(superposed_locals)
    linked_src = [False] * len(srcs)
    linked_dest = [False] * superposed_len
    linkage = [[0 for _ in range(superposed_len)] for _ in range(len(srcs))]
    combs = list(product(np.arange(len(srcs)), np.arange(len(superposed_locals))))
    euclid_tmp0 = []
    euclid_tmp1 = []
    for i, dest in combs:
        euclid_tmp0.append(srcs[i])
        euclid_tmp1.append(superposed_locals[dest])
    euclid_tmp0 = np.array(euclid_tmp0)
    euclid_tmp1 = np.array(euclid_tmp1)

    segment_lengths = euclidean_displacement(euclid_tmp0, euclid_tmp1)
    x_diff = euclid_tmp0[:, 0] - euclid_tmp1[:, 0]
    y_diff = euclid_tmp0[:, 1] - euclid_tmp1[:, 1]
    z_diff = euclid_tmp0[:, 2] - euclid_tmp1[:, 2]

    if segment_lengths is not None:
        for (i, dest), segment_length, x_, y_, z_ in zip(combs, segment_lengths, x_diff, y_diff, z_diff):
            if segment_length is not None:
                linkage[i][dest] = segment_length
    minargs = np.argsort(np.array(linkage).flatten())

    for minarg in minargs:
        src = minarg // superposed_len
        dest = minarg % superposed_len
        if linked_dest[dest] or linked_src[src]:
            continue
        else:
            linked_dest[dest] = True
            linked_src[src] = True
            x_distribution.append(x_diff[minarg])
            y_distribution.append(y_diff[minarg])
            z_distribution.append(z_diff[minarg]) 
    return x_distribution[:-1], y_distribution[:-1], z_distribution[:-1]


def segmentation(localization: dict, time_steps: np.ndarray, lag=2):
    lag = 0
    dist_x_all = []
    dist_y_all = []
    dist_z_all = []

    for i, time_step in enumerate(time_steps[:-1]):
        dests = [[] for _ in range(lag + 1)]
        srcs = localization[time_step]
        for j in range(i+1, i+lag+2):
            dest = localization[time_steps[j]]
            dests[j - i - 1].extend(dest)
        for dest in dests:
            if srcs[0].shape[0] > 1 and dest[0].shape[0] > 1:
                dist_x, dist_y, dist_z = greedy_shortest(srcs=srcs, dests=dest)
                dist_x_all.extend(dist_x)
                dist_y_all.extend(dist_y)
                dist_z_all.extend(dist_z)


    ndim = 2 if np.var(dist_z_all) < 1e-5 else 3
    diffraction_light_limit = 10  #TODO:diffraction light limit

    for _ in range(2):
        filtered_x = []
        filtered_y = []
        filtered_z = []
        if ndim == 2:
            estim_limit = 4 * np.mean([np.std(dist_x_all[:-1]), np.std(dist_y_all[:-1])])
        else:
            estim_limit = 4 * np.mean([np.std(dist_x_all[:-1]), np.std(dist_y_all[:-1]), np.std(dist_z_all[:-1])])
        filter_min = max(estim_limit, diffraction_light_limit)
        for x, y, z in zip(dist_x_all[:-1], dist_y_all[:-1], dist_z_all[:-1]):
            if abs(x) < filter_min and abs(y) < filter_min and abs(z) < filter_min:
                filtered_x.append(x)
                filtered_y.append(y)
                filtered_z.append(z)
        dist_x_all = filtered_x
        dist_y_all = filtered_y
        dist_z_all = filtered_z

    filtered_x = []
    filtered_y = []
    filtered_z = []
    for x, y, z in zip(dist_x_all[:-1], dist_y_all[:-1], dist_z_all[:-1]):
        if abs(x) < diffraction_light_limit and abs(y) < diffraction_light_limit and abs(z) < diffraction_light_limit:
            filtered_x.append(x)
            filtered_y.append(y)
            filtered_z.append(z)
    dist_x_all = filtered_x
    dist_y_all = filtered_y
    dist_z_all = filtered_z
    return np.array([dist_x_all, dist_y_all, dist_z_all])


def count_localizations(localization):
    nb = 0
    xyz_min = np.array([1e5, 1e5, 1e5])
    xyz_max = np.array([-1e5, -1e5, -1e5])
    time_steps = np.sort(list(localization.keys()))
    for t in time_steps:
        loc = localization[t]
        if loc.shape[1] > 0:
            x_ = loc[:, 0]
            y_ = loc[:, 1]
            z_ = loc[:, 2]
            xyz_min = [min(xyz_min[0], np.min(x_)), min(xyz_min[1], np.min(y_)), min(xyz_min[2], np.min(z_))]
            xyz_max = [max(xyz_max[0], np.max(x_)), max(xyz_max[1], np.max(y_)), max(xyz_max[2], np.max(z_))]
            nb += len(loc)
    nb_per_time = nb / len(time_steps)
    return np.array(time_steps), nb_per_time, np.array(xyz_min), np.array(xyz_max)


def euclidean_displacement(pos1, pos2):
    assert type(pos1) == type(pos2)
    if type(pos1) is not np.ndarray and type(pos1) is not list:
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2)
    if pos1.ndim == 2:
        if len(pos1[0]) == 0 or len(pos2[0]) == 0:
            return None
    if type(pos1) != np.ndarray and type(pos1) == list:
        return [math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2)]
    elif type(pos1) == np.ndarray and pos1.ndim == 1:
        return [math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2)]
    elif type(pos1) == np.ndarray and pos1.shape[0] >= 1 and pos1.shape[1] == 3:
        return np.sqrt((pos1[:, 0] - pos2[:, 0])**2 + (pos1[:, 1] - pos2[:, 1])**2 + (pos1[:, 2] - pos2[:, 2])**2)
    elif len(pos1[0]) == 0 or len(pos2[0]) == 0:
        return None
    else:
        raise Exception


def gmm_bic_score(estimator, x):
    return -estimator.bic(x)


def approx_gauss(distributions):
    #resample_nb = 3000
    #resampled = distribution[np.random.randint(0, len(distribution), min(resample_nb, len(distribution)))]
    max_xyz = []
    max_euclid = 0
    min_euclid = 6.5   #TODO increase over time? well...

    qt_distrbutions = []
    for distribution in distributions:
        if np.var(distribution) > 1e-5:
            distribution = np.array(distribution)
            quantile = np.quantile(distribution, [0.025, 0.975])
            qt_distrbutions.append(distribution[(distribution > quantile[0]) * (distribution < quantile[1])])
    distributions = qt_distrbutions

    for distribution in distributions:
        if np.var(distribution) > 1e-5:
            selec_mean = []
            selec_var = []
            param_grid = [
                {
                "n_components": [1],
                "means_init": [[[0]]]
                },
                {
                "n_components": [2],
                "means_init": [[[0], [0]]]
                },
                {
                "n_components": [3],
                "means_init": [[[0], [0], [0]]]
                }
                ]
            grid_search = GridSearchCV(
                GaussianMixture(max_iter=100, n_init=3, covariance_type='full'),
                param_grid=param_grid,
                scoring=gmm_bic_score, verbose=0
            )
            grid_search.fit(distribution.reshape(-1, 1))
            cluster_df = pd.DataFrame(grid_search.cv_results_)[
                ["param_n_components", "mean_test_score"]
            ]
            cluster_df["mean_test_score"] = -cluster_df["mean_test_score"]
            cluster_df = cluster_df.rename(
                columns={
                    "param_n_components": "Number of components",
                    "mean_test_score": "BIC score",
                }
            )
            opt_nb_component = np.argmin(cluster_df["BIC score"]) + 1
            cluster = BayesianGaussianMixture(n_components=opt_nb_component, max_iter=100, n_init=3,
                                            mean_prior=[0], mean_precision_prior=1e7, covariance_type='full').fit(distribution.reshape(-1, 1))

            for mean_, cov_, weight_ in zip(cluster.means_.flatten(), cluster.covariances_.flatten(), cluster.weights_.flatten()):
                if -1 < mean_ < 1 and weight_ > 0.05:
                    selec_mean.append(mean_)
                    selec_var.append(cov_)
            max_arg = np.argsort(selec_var)[::-1][0]
            max_var = selec_var[max_arg]
            max_xyz.append(math.sqrt(max_var) * 2.5)
            
    system_dim = len(max_xyz)
    for i in range(system_dim):
        max_euclid += max_xyz[i]**2
    max_euclid = max(math.sqrt(max_euclid), min_euclid)
    return max_euclid


def approximation(real_distribution, time_forecast, jump_threshold=float|None):
    approx = {}
    if jump_threshold is None:
        max_euclid = approx_gauss(real_distribution)
        for t in range(time_forecast+1):
            approx[t] = max_euclid 
    else:
        for t in range(time_forecast+1):
            approx[t] = jump_threshold
    return approx


def metropolis_hastings(pdf, n_iter, burn=0.25):
    i = 0
    u = np.random.uniform(0, 1, size=n_iter)
    current_x = np.argmax(pdf)
    samples = []
    acceptance_ratio = np.array([0, 0])
    while True:
        next_x = int(np.round(np.random.normal(current_x, 1)))
        next_x = max(0, min(next_x, len(pdf) - 1))
        proposal1 = 1  # g(current|next)
        proposal2 = 1  # g(next|current)
        target1 = pdf[next_x]
        target2 = pdf[current_x]
        accept_proba = min(1, (target1 * proposal1) / (target2 * proposal2))
        if u[i] <= accept_proba:
            samples.append(next_x)
            current_x = next_x
            acceptance_ratio[1] += 1
        else:
            acceptance_ratio[0] += 1
        i += 1
        if i == n_iter:
            break
    return np.array(samples)[int(len(samples)*burn):]


def find_paths_as_iter(G, source=(0, 0), path=None, seen=None):
    if path is None:
        path = [tuple(source)]
    if seen is None:
        seen = {source}
    desc = nx.descendants_at_distance(G, source, 1)
    if not desc:
        yield path
    else:
        for n in desc:
            if n in seen:
                yield path 
            else:
                yield from find_paths_as_iter(G, n, path+[tuple(n)], seen.union([n]))


def find_paths_as_list(G, source=(0, 0), path=None, seen=None):
    return_path_list = []
    path_list = list(find_paths_as_iter(G, source, path, seen))
    for path in path_list:
        if len(path) > 1:
            return_path_list.append(tuple(path))
    return return_path_list


def predict_alphas(x, y):
    pred_alpha = REG_MODEL.alpha_predict(np.array([x, y]))
    return pred_alpha


def predict_ks(x, y):
    pred_logk = REG_MODEL.k_predict([np.array([x, y])])
    return pred_logk[0]


def predict_long_seq(next_path, trajectories_costs, localizations, prev_alpha, prev_k, next_times, prev_path=None, start_indice=None, last_time=-1, jump_threshold=20, selected_graph=None, final_graph=None):
    traj_cost = []
    ab_index = []
    abnormal = False
    abnormal_penalty = 1000
    time_penalty = abnormal_penalty / TIME_FORECAST
    time_score = 0
    abnomral_jump_score = 0
    time_gaps = 0
    cutting_threshold = 2 * abnormal_penalty
    initial_cost = cutting_threshold - 100

    if prev_path is not None:
        prev_path = list(prev_path)
    
    if trajectories_costs[next_path] is not None or len(next_path) == 1:
        return ab_index, None

    terminal = is_terminal(next_path[-1], localizations, jump_threshold, selected_graph, final_graph)

    for idx in range(1, len(next_path) - 1):
        if (next_path[idx+1][0] - next_path[idx][0]) - 1 > TIME_FORECAST:
            trajectories_costs[next_path] = initial_cost
            return [idx], terminal

    if len(next_path) <= 1:
        print(next_path)
        raise Exception
    elif len(next_path) == 2:
        trajectories_costs[next_path] = initial_cost + abnormal_penalty
    elif len(next_path) == 3:
        before_node = next_path[1]
        next_node = next_path[2]
        time_gap = next_node[0] - before_node[0] - 1
        next_coord = localizations[next_node[0]][next_node[1]]
        cur_coord = localizations[before_node[0]][before_node[1]]
        input_mu = next_coord - cur_coord
        log_p0, abnormal = predict_multinormal(input_mu, prev_alpha, prev_k, time_gap)
        trajectories_costs[next_path] = initial_cost + abs(log_p0) 
    else:
        before_node = next_path[1]
        next_node = next_path[2]
        time_gap = next_node[0] - before_node[0] - 1
        next_coord = localizations[next_node[0]][next_node[1]]
        cur_coord = localizations[before_node[0]][before_node[1]]
        input_mu = next_coord - cur_coord
        log_p0, abnormal1 = predict_multinormal(input_mu, prev_alpha, prev_k, time_gap)
        if abnormal1:
            abnomral_jump_score += abnormal_penalty
            ab_index.append(1)
        traj_cost.append(abs(log_p0))
        
        for edge_index in range(3, len(next_path)):
            if abnormal:
                prev_alpha = 1.0
                prev_k = 2.0
                if prev_path is not None:
                    prev_path = [prev_path[0]]
            
            bebefore_node = next_path[edge_index - 2]
            before_node = next_path[edge_index - 1]
            next_node = next_path[edge_index]
            time_gap = next_node[0] - before_node[0] - 1
            next_coord = localizations[next_node[0]][next_node[1]]
            cur_coord = localizations[before_node[0]][before_node[1]]
            before_coord = localizations[bebefore_node[0]][bebefore_node[1]]

            if prev_path is not None and not abnormal:
                prev_path.append(before_node)
                if edge_index % ALPHA_MODULO == 0:
                    prev_xys = np.array([localizations[txy[0]][txy[1]][:2] for txy in prev_path[1:]])[-ALPHA_MAX_LENGTH:]
                    prev_alpha = predict_alphas(prev_xys[:,0], prev_xys[:,1])

            log_p0, abnormal2 = cost_function.predict_cauchy((next_coord - cur_coord), (cur_coord- before_coord), prev_alpha, time_gap, 1.0, DIMENSION)

            if terminal:
                if abnormal2:
                    abnomral_jump_score += abnormal_penalty
                    ab_index.append(edge_index - 1)
            else:
                if abnormal2 and edge_index != len(next_path) - 1:
                    abnomral_jump_score += abnormal_penalty
                    ab_index.append(edge_index - 1)
            traj_cost.append(abs(log_p0))

        for node_idx in range(1, len(next_path)-1):
            time_gaps += (next_path[node_idx+1][0] - next_path[node_idx][0]) - 1
        time_score += time_gaps * time_penalty
        traj_cost = np.array(traj_cost)

        if len(traj_cost) > 1:
            if not terminal:
                final_score = np.mean(traj_cost[:-1]) + abnomral_jump_score + time_score# + np.std(traj_cost[:-1])
            else:
                final_score = np.mean(traj_cost) + abnomral_jump_score + time_score# + np.std(traj_cost)
        else:
            final_score = abnormal_penalty + time_score + traj_cost[0]

        trajectories_costs[next_path] = final_score
        #print(trajectories_costs[next_path], traj_cost, abnomral_jump_score, time_score, ab_index, next_path, prev_alpha, prev_k, pdf_mu_measure(prev_alpha), np.std(traj_cost[:-1]))

    if trajectories_costs[next_path] > cutting_threshold:
        return ab_index, terminal
    else:
        return [], terminal
    

def generate_next_paths(next_graph:nx.graph, init_graph:nx.graph, localizations, next_times, distribution, source_node):
    while True:
        start_g_len = len(next_graph.nodes)
        index = 0
        cumulative_last_nodes = []
        while True:
            last_nodes = list([nodes[-1] for nodes in find_paths_as_iter(next_graph, source=source_node)])
            for last_node in last_nodes:
                if last_node not in cumulative_last_nodes:
                    cumulative_last_nodes.append(last_node)

            for last_node in cumulative_last_nodes:
                for cur_time in next_times[index:index+1]:
                    if last_node[0] < cur_time and last_node != source_node:
                        jump_d_pos1 = []
                        jump_d_pos2 = []
                        node_loc = localizations[last_node[0]][last_node[1]]
                        for next_idx, loc in enumerate(localizations[cur_time]):
                            if len(loc) == 3 and len(node_loc) == 3:
                                jump_d_pos1.append([loc[0], loc[1], loc[2]])
                                jump_d_pos2.append([node_loc[0], node_loc[1], node_loc[2]])
                        jump_d_pos1 = np.array(jump_d_pos1)
                        jump_d_pos2 = np.array(jump_d_pos2)
                        if jump_d_pos1.shape[0] > 0:
                            jump_d_mat = euclidean_displacement(jump_d_pos1, jump_d_pos2)
                            local_idx = 0
                            for next_idx, loc in enumerate(localizations[cur_time]):
                                if len(loc) == 3 and len(node_loc) == 3:
                                    jump_d = jump_d_mat[local_idx]
                                    local_idx += 1
                                    time_gap = cur_time - last_node[0] - 1
                                    if time_gap in distribution:
                                        threshold = distribution[time_gap]
                                        if jump_d < threshold:
                                            next_node = (cur_time, next_idx)
                                            if next_node not in init_graph.nodes:
                                                next_graph.add_edge(last_node, next_node, jump_d=jump_d)

            for cur_time in next_times[index:index+1]:
                for idx in range(len(localizations[cur_time])):
                    if (cur_time, idx) not in next_graph and (cur_time, idx) not in init_graph and len(localizations[cur_time][0]) == 3:
                        next_graph.add_edge((0, 0), (cur_time, idx), jump_d=-1)

            index += 1
            if index == len(next_times):
                break
        end_g_len = len(next_graph.nodes)
        if start_g_len == end_g_len:
            break
    return next_graph, cumulative_last_nodes


def match_prev_next(prev_paths, next_path, hashed_prev_next):
    if next_path not in hashed_prev_next:
        for prev_path in prev_paths:
            if len(prev_path) > 1:
                if prev_path[-1] in next_path:
                    hashed_prev_next[next_path] = prev_path
                    return hashed_prev_next[next_path]
        hashed_prev_next[next_path] = None
        return hashed_prev_next[next_path]
    else:
        return hashed_prev_next[next_path]
    

def is_terminal(node, localizations, max_jump_d, selected_graph, final_graph):
    node_t = node[0]
    node_loc_idx = node[1]
    node_loc = localizations[node_t][node_loc_idx]

    for next_t in range(node_t + 1, node_t + TIME_FORECAST + 1):
        if next_t in localizations:
            for next_node_idx, search_loc in enumerate(localizations[next_t]):
                next_node = tuple([next_t, next_node_idx])
                if len(search_loc) == 3 and next_node not in selected_graph.nodes and next_node not in final_graph.nodes:
                    jump_d = euclidean_displacement(node_loc, search_loc)[0]
                    if jump_d < max_jump_d:
                        return False
    return True


def select_opt_graph2(final_graph:nx.graph, saved_graph:nx.graph, next_graph:nx.graph, localizations, next_times, distribution, first_step, last_time):
    selected_graph = nx.DiGraph()
    source_node = (0, 0)
    selected_graph.add_node(source_node)
    alpha_values = {}
    k_values = {}
    start_indice = {}
    hashed_prev_next = {}
    init_graph = final_graph.copy()
    prev_lowest = [source_node]

    if not first_step:
        prev_paths = find_paths_as_list(saved_graph, source=source_node)
        if TF:
            for path_idx in range(len(prev_paths)):
                prev_xys = np.array([localizations[txy[0]][txy[1]][:2] for txy in prev_paths[path_idx][1:]])[-ALPHA_MAX_LENGTH:]
                if len(prev_xys) > 0:
                    prev_x_pos = prev_xys[:, 0]
                    prev_y_pos = prev_xys[:, 1]
                    prev_alpha = predict_alphas(prev_x_pos, prev_y_pos)
                    prev_k = predict_ks(prev_x_pos, prev_y_pos)
                    alpha_values[tuple(prev_paths[path_idx])] = prev_alpha
                    k_values[tuple(prev_paths[path_idx])] = prev_k
                    prev_paths[path_idx] = tuple(prev_paths[path_idx])
        else:
            for path_idx in range(len(prev_paths)):
                alpha_values[tuple(prev_paths[path_idx])] = 1.0
                k_values[tuple(prev_paths[path_idx])] = 2.0
                prev_paths[path_idx] = tuple(prev_paths[path_idx])

    # Generate next graph
    next_graph, last_nodes = generate_next_paths(next_graph, init_graph, localizations, next_times, distribution, source_node)
    trajectories_costs = {tuple(next_path):None for next_path in find_paths_as_iter(next_graph, source=source_node)}
    ab_indice = {}
    is_terminals = {}
    orphans = []

    while True:
        cost_copy = {}
        next_paths = find_paths_as_list(next_graph, source=source_node)

        if len(next_paths) <= 0:
            break
        
        for path_idx in range(len(next_paths)):
            next_path = tuple(next_paths[path_idx])
            index_ind = 0
            for next_node in next_path:
                if next_node in init_graph.nodes:
                    index_ind += 1
            start_indice[tuple(next_path)] = index_ind
            if next_path in trajectories_costs:
                cost_copy[next_path] = trajectories_costs[next_path]
            next_paths[path_idx] = tuple(next_paths[path_idx])
        trajectories_costs = cost_copy
        
        # Calculate cost
        if first_step:
            for next_path in next_paths:
                ab_index, term = predict_long_seq(next_path, trajectories_costs, localizations, 1.0, 2.0, next_times, start_indice=start_indice, last_time=last_time, jump_threshold=distribution[0], selected_graph=selected_graph, final_graph=final_graph)
                if term is not None:
                    is_terminals[next_path] = term
                if len(ab_index) > 0:
                    ab_indice[next_path] = ab_index 

        else:
            for next_path in next_paths:
                prev_path = match_prev_next(prev_paths, next_path, hashed_prev_next)
                if prev_path is None:
                    ab_index, term = predict_long_seq(next_path, trajectories_costs, localizations, 1.0, 2.0, next_times, start_indice=start_indice, last_time=last_time, jump_threshold=distribution[0], selected_graph=selected_graph, final_graph=final_graph)
                else:
                    prev_alpha = alpha_values[prev_path]
                    prev_k = k_values[prev_path]
                    ab_index, term = predict_long_seq(next_path, trajectories_costs, localizations, prev_alpha, prev_k, next_times, prev_path, start_indice=start_indice, last_time=last_time, jump_threshold=distribution[0], selected_graph=selected_graph, final_graph=final_graph)
                if len(ab_index) > 0:
                    ab_indice[next_path] = ab_index 
                if term is not None:
                    is_terminals[next_path] = term

        # Cost sorting
        trajs = [path for path in trajectories_costs.keys()]
        costs = [trajectories_costs[path] for path in trajectories_costs.keys()]
        low_cost_args = np.argsort(costs)
        next_trajectories = np.array(trajs, dtype=object)[low_cost_args]
        lowest_cost_traj = list(next_trajectories[0])
        for i in range(len(lowest_cost_traj)):
            lowest_cost_traj[i] = tuple(lowest_cost_traj[i])
        lowest_cost_traj = tuple(lowest_cost_traj)
        
        """
        print('##################################################################')
        for cost, traj in zip(np.array(costs)[low_cost_args][::-1][-30:], next_trajectories[::-1][-30:]):
            traj = tuple([tuple(x) for x in traj])
            print(f'{traj} -> {cost}', is_terminals[traj], lowest_cost_traj)
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        """

        # Abnormal trajectory cutting
        if lowest_cost_traj in ab_indice:
            for ab_i in ab_indice[tuple(lowest_cost_traj)][:1]:
                if (lowest_cost_traj[ab_i], lowest_cost_traj[ab_i+1]) in next_graph.edges:
                    next_graph.remove_edge(lowest_cost_traj[ab_i], lowest_cost_traj[ab_i+1])
                if (source_node, lowest_cost_traj[ab_i+1]) not in next_graph.edges:
                    next_graph.add_edge(source_node, lowest_cost_traj[ab_i+1])
                added_path = [source_node]
                for path in lowest_cost_traj[ab_i+1:]:
                    added_path.append(path)
                added_path = tuple(added_path)
                trajectories_costs[added_path] = None
                added_path = [source_node]
                for path in lowest_cost_traj[1:ab_i+1]:
                    added_path.append(path)
                added_path = tuple(added_path)
                trajectories_costs[added_path] = None

            next_paths = find_paths_as_list(next_graph, source=source_node)
            for path_idx in range(len(next_paths)):
                next_paths[path_idx] = tuple(next_paths[path_idx])
            for next_path in next_paths:
                if next_path not in trajectories_costs:
                    trajectories_costs[next_path] = None
            continue


        # Prune the graph
        while 1:
            before_pruning = len(next_graph)
            for rm_node in lowest_cost_traj[1:]:
                predcessors = list(next_graph.predecessors(rm_node)).copy()
                sucessors = list(next_graph.successors(rm_node)).copy()
                next_graph_copy = next_graph.copy()
                next_graph_copy.remove_node(rm_node)
                for pred in predcessors:
                    for suc in sucessors:
                        if pred not in init_graph.nodes and suc not in init_graph.nodes:
                            if pred != source_node and (pred, suc) not in next_graph.edges:
                                pred_loc = localizations[pred[0]][pred[1]]
                                suc_loc = localizations[suc[0]][suc[1]]
                                jump_d = euclidean_displacement(pred_loc, suc_loc)[0]
                                time_gap = suc[0] - pred[0] - 1
                                if time_gap in distribution:
                                    threshold = distribution[time_gap]
                                    if jump_d < threshold:
                                        next_graph.add_edge(pred, suc, jump_d=jump_d)
            after_pruning = len(next_graph)
            if before_pruning == after_pruning:
                break
        

        if is_terminals[lowest_cost_traj]:
            for del_node in lowest_cost_traj[1:]:
                next_graph.remove_node(del_node)
        else:
            if len(lowest_cost_traj) == 2:
                next_graph.remove_node(lowest_cost_traj[-1])
            else:
                for del_node in lowest_cost_traj[1:-1]:
                    next_graph.remove_node(del_node)
        pop_cost = trajectories_costs.pop(lowest_cost_traj)


        # selected graph update
        terminal_lowest_cost = is_terminals[lowest_cost_traj]
        for edge_index in range(1, len(lowest_cost_traj)):
            before_node = lowest_cost_traj[edge_index - 1]
            next_node = lowest_cost_traj[edge_index]
            selected_graph.add_edge(before_node, next_node, terminal=terminal_lowest_cost)

        #print(f'LEN:{len(next_graph), next_graph.edges, next_graph.nodes, len(last_nodes), last_nodes},  Neighbor of source : ',list(next_graph.neighbors(source_node)))
        if len(list(next_graph.neighbors(source_node))) ==0 or tuple(lowest_cost_traj) == tuple(prev_lowest):
            break
            
        # newborn cost update
        for next_path in find_paths_as_iter(next_graph, source=source_node):
            next_path = tuple(next_path)
            if next_path not in trajectories_costs:
                trajectories_costs[next_path] = None
            
        prev_lowest = list(lowest_cost_traj).copy()
        prev_lowest = tuple(prev_lowest)

    for time in next_times:
        for node_idx in range(len(localizations[time])):
            cur_node = tuple([time, node_idx])
            if len(localizations[time][node_idx]) == 3 and cur_node not in selected_graph.nodes and cur_node not in final_graph.nodes:
                orphans.append(cur_node)

    return selected_graph, len(orphans) > 0


def forecast(localization: dict, t_avail_steps, distribution, image_length, realtime_visualization):
    first_construction = True
    last_time = image_length
    source_node = (0, 0)
    time_forecast = TIME_FORECAST
    final_graph = nx.DiGraph()
    light_prev_graph = nx.DiGraph()
    next_graph = nx.DiGraph()
    final_graph.add_node(source_node)
    light_prev_graph.add_node(source_node)
    next_graph.add_node(source_node)
    next_graph.add_edges_from([((0, 0), (t_avail_steps[0], index), {'jump_d':-1}) for index in range(len(localization[t_avail_steps[0]]))])
    selected_time_steps = np.arange(t_avail_steps[0], min(t_avail_steps[0] + 1 + time_forecast, t_avail_steps[-1] + 1))
    saved_time_steps = 1
    mysum = 0
    return_graphs = []


    realtime_obj = None
    if realtime_visualization:
        from FreeTrace.module.image_module import RealTimePlot
        realtime_obj = RealTimePlot(f'Tracking : {VIDEO_PATH}', job_type='track', show_frame=True)
        realtime_obj.turn_on()


    while True:
        node_pairs = []
        start_time = selected_time_steps[-1]
        if VERBOSE:
            pbar_update = selected_time_steps[0] - saved_time_steps -1 + len(selected_time_steps)
            mysum += pbar_update
            PBAR.update(pbar_update)

        if len(set(selected_time_steps).intersection(set(t_avail_steps))) != 0:
            selected_sub_graph, has_orphan = select_opt_graph2(final_graph, light_prev_graph, next_graph, localization, selected_time_steps, distribution, first_construction, last_time)
        else:
            selected_sub_graph = nx.DiGraph()
            selected_sub_graph.add_node(source_node)

        first_construction = False
        light_prev_graph = nx.DiGraph()
        light_prev_graph.add_node(source_node)

        selected_paths = find_paths_as_list(selected_sub_graph, source=source_node)
        if len(selected_sub_graph.nodes) <= 1 and last_time in selected_time_steps:
            if VERBOSE:
                    PBAR.update(image_length - mysum)
            break
        else:
            if last_time in selected_time_steps and not has_orphan:
                if VERBOSE:
                    PBAR.update(image_length - mysum)
                for path in selected_paths:
                    without_source_path = path[1:]
                    if len(without_source_path) == 1:
                        if without_source_path[0] not in final_graph.nodes:
                            final_graph.add_edge(source_node, tuple(without_source_path[0]))
                    else:
                        for idx in range(len(without_source_path) - 1):
                            before_node = tuple(without_source_path[idx])
                            next_node = tuple(without_source_path[idx+1])
                            if next_node not in final_graph.nodes:
                                final_graph.add_edge(before_node, next_node)
                    if not nx.has_path(final_graph, source_node, tuple(without_source_path[0])):
                        final_graph.add_edge(source_node, tuple(without_source_path[0]))
                break
            else:
                for path in selected_paths:
                    terminal = selected_sub_graph.get_edge_data(path[0], path[1])['terminal']
                    if len(path) == 2:
                        if terminal and path[-1] not in final_graph.nodes:
                            final_graph.add_edge(source_node, path[-1])
                        else:
                            start_time = min(start_time, path[-1][0])
                    else:
                        if not terminal:
                            start_time = min(start_time, path[-2][0])
                            if len(path) == 3:
                                before_node = path[1]
                                if before_node not in final_graph.nodes:
                                    final_graph.add_edge(source_node, before_node)
                                node_pairs.append([path[1]])            
                            elif len(path) > 3:
                                for edge_index in range(2, len(path) - 1):
                                    before_node = path[edge_index - 1]
                                    next_node = path[edge_index]
                                    if before_node in final_graph.nodes:
                                        if (before_node, next_node) not in final_graph.edges:
                                            final_graph.add_edge(before_node, next_node)
                                    else:
                                        if (source_node, before_node) not in final_graph.edges:
                                            final_graph.add_edge(source_node, before_node)
                                        if (before_node, next_node) not in final_graph.edges:
                                            final_graph.add_edge(before_node, next_node)

                                node_pairs.append([path[-3], path[-2]])
                                ancestors = list(nx.ancestors(final_graph, path[-2]))
                                sorted_ancestors = sorted(ancestors, key=lambda tup: tup[0], reverse=True)
                                if len(sorted_ancestors) > 1:
                                    for idx in range(len(sorted_ancestors[:ALPHA_MAX_LENGTH+3]) - 1):
                                        light_prev_graph.add_edge(sorted_ancestors[idx+1], sorted_ancestors[idx])
                                    if sorted_ancestors[idx+1] != source_node:
                                        light_prev_graph.add_edge(source_node, sorted_ancestors[idx+1]) 
                        else:
                            for edge_index in range(2, len(path)):
                                before_node = path[edge_index - 1]
                                next_node = path[edge_index]
                                if before_node in final_graph.nodes:
                                    if (before_node, next_node) not in final_graph.edges:
                                        final_graph.add_edge(before_node, next_node)
                                else:
                                    if (source_node, before_node) not in final_graph.edges:
                                        final_graph.add_edge(source_node, before_node)
                                    if (before_node, next_node) not in final_graph.edges:
                                        final_graph.add_edge(before_node, next_node)                    

        ## start time -> node min not in final graph from selected time steps
        for time in selected_time_steps:
            for node_idx in range(len(localization[time])):
                node = tuple([time, node_idx])
                if len(localization[time][node_idx]) == 3 and node not in final_graph.nodes:
                    start_time = min(start_time, node[0])

        if realtime_visualization:
            realtime_obj.put_into_queue((IMAGES, find_paths_as_iter(final_graph, source=source_node), selected_time_steps[:-1], localization), mod_n=1)
        
        saved_time_steps = selected_time_steps[-1]
        next_first_time = selected_time_steps[-1] + 1
        next_graph = nx.DiGraph()
        next_graph.add_node(source_node)

        selected_time_steps = [t for t in range(start_time, min(last_time + 1, next_first_time + time_forecast))]
        for node_pair in node_pairs:
            if len(node_pair) == 1:
                next_graph.add_edge(source_node, node_pair[0], jump_d=-1)
            else:
                last_xyz = localization[node_pair[-1][0]][node_pair[-1][1]]
                second_last_xyz = localization[node_pair[0][0]][node_pair[0][1]]
                next_graph.add_edge(source_node, node_pair[0], jump_d=-1)
                next_graph.add_edge(node_pair[0], node_pair[-1], jump_d=math.sqrt((last_xyz[0] - second_last_xyz[0])**2 + (last_xyz[1] - second_last_xyz[1])**2))

    all_nodes_ = []
    for t in list(localization.keys()):
        for nb_sample in range(len(localization[t])):
            if len(localization[t][nb_sample]) == 3:
                all_nodes_.append((t, nb_sample))
    for node_ in all_nodes_:
        if node_ not in final_graph:
            print('Dropped node: ', node_)

    if realtime_obj is not None:
        realtime_obj.turn_off()

    trajectory_list = []
    traj_idx = 0
    for path in find_paths_as_iter(final_graph, source=source_node):
        if len(path) >= CUTOFF + 1:
            traj = TrajectoryObj(index=traj_idx, localizations=localization)
            for node in path[1:]:
                traj.add_trajectory_tuple(node[0], node[1])
            trajectory_list.append(traj)
            traj_idx += 1
    return trajectory_list


def trajectory_inference(localization: dict, time_steps: np.ndarray, distribution: dict, image_length=None, realtime_visualization=False):
    t_avail_steps = []
    for time in np.sort(time_steps):
        if len(localization[time][0]) == 3:
            t_avail_steps.append(time)
    trajectory_list = forecast(localization, t_avail_steps, distribution, image_length, realtime_visualization=realtime_visualization)
    return trajectory_list


def run(input_video_path:str, output_path:str, time_forecast=2, cutoff=0, jump_threshold=None, gpu_on=True, save_video=False, verbose=False, batch=False, realtime_visualization=False, return_state=0):
    global IMAGES
    global VERBOSE
    global BATCH
    global CUTOFF
    global GPU_AVAIL
    global REG_LEGNTHS
    global ALPHA_MAX_LENGTH
    global CUDA
    global TF
    global POLY_FIT_DATA
    global STD_FIT_DATA 
    global TIME_FORECAST
    global PBAR
    global REG_MODEL
    global JUMP_THRESHOLD
    global ALPHA_MODULO
    global VIDEO_PATH
    global DIMENSION


    VERBOSE = verbose
    BATCH = batch
    TIME_FORECAST = max(1, min(5, time_forecast))
    CUTOFF = cutoff
    GPU_AVAIL = gpu_on
    REG_LEGNTHS = [3, 5, 8]
    ALPHA_MAX_LENGTH = 10
    ALPHA_MODULO = 3
    DIMENSION = 2
    JUMP_THRESHOLD = jump_threshold
    VIDEO_PATH = input_video_path
    CUDA, TF = initialization(GPU_AVAIL, REG_LEGNTHS, ptype=1, verbose=VERBOSE, batch=BATCH)
    POLY_FIT_DATA = np.load(f'{__file__.split("/Tracking.py")[0]}/models/theta_hat.npz')
    STD_FIT_DATA = np.load(f'{__file__.split("/Tracking.py")[0]}/models/std_sets.npz')


    output_xml = f'{output_path}/{input_video_path.split("/")[-1].split(".tif")[0]}_traces.xml'
    output_trj = f'{output_path}/{input_video_path.split("/")[-1].split(".tif")[0]}_traces.csv'
    output_trxyt = f'{output_path}/{input_video_path.split("/")[-1].split(".tif")[0]}_traces.trxyt'
    output_imgstack = f'{output_path}/{input_video_path.split("/")[-1].split(".tif")[0]}_traces.tiff'
    output_img = f'{output_path}/{input_video_path.split("/")[-1].split(".tif")[0]}_traces.png'


    final_trajectories = []
    images = read_tif(input_video_path)
    IMAGES = images
    if images.shape[0] <= 1:
        sys.exit('Image squence length error: Cannot track on a single image.')
    loc, loc_infos = read_localization(f'{output_path}/{input_video_path.split("/")[-1].split(".tif")[0]}_loc.csv', images)


    if TF:
        if VERBOSE:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        else:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
        from FreeTrace.module.load_models import RegModel
        REG_MODEL = RegModel(REG_LEGNTHS)


    t_steps, mean_nb_per_time, xyz_min, xyz_max = count_localizations(loc)
    raw_distributions = segmentation(loc, time_steps=t_steps, lag=time_forecast)
    max_jumps = approximation(raw_distributions, time_forecast=time_forecast, jump_threshold=JUMP_THRESHOLD)


    if VERBOSE:
        print(f'Mean nb of particles per frame: {mean_nb_per_time:.2f} particles/frame')
        PBAR = tqdm(total=t_steps[-1], desc="Tracking", unit="frame", ncols=120)


    try:
        final_trajectories = trajectory_inference(localization=loc, time_steps=t_steps,
                                                  distribution=max_jumps, image_length=images.shape[0], realtime_visualization=realtime_visualization)
    except Exception as e:
        print("\nInference ERR: ", e)
        return_state.value = 0
        if VERBOSE:
            PBAR.close()
        sys.exit(0)


    if VERBOSE:
        PBAR.close()


    #write_xml(output_file=output_xml, trajectory_list=final_trajectories, snr='7', density='low', scenario='Vesicle', cutoff=CUTOFF)
    write_trajectory(output_trj, final_trajectories)
    make_whole_img(final_trajectories, output_dir=output_img, img_stacks=images)
    if save_video:
        print(f'Visualizing trajectories...')
        make_image_seqs(final_trajectories, output_dir=output_imgstack, img_stacks=images, time_steps=t_steps)
    

    if return_state != 0:
        return_state.value = 1
    return True


def run_process(input_video_path:str, output_path:str, time_forecast=5, cutoff=2, jump_threshold=None|float,
                gpu_on=True, save_video=False, verbose=False, batch=False, realtime_visualization=False) -> bool:
    """
    Create a process to run the tracking of particles to reconstruct the trajectories from localized molecules.
    This function reads both the video.tiff and the video_loc.csv which was generated with Localization process.
    Thus, the localization of particles is mandatory before performing the reconstruction of trajectories. 

    @params
        input_video_path:
        Path of video (video.tiff)

        output_path:
        Path of outputs (video_traces.csv and supplementary outputs depending on the visualization options)
        
        time_forecast(frame):
        Amount of frames to consider for the reconstruction of most probable trajectories for each calculation. 
        
        cutoff(frame):
        Minimum length of trajectory to consider.

        jump_threshold(pixel): 
        Maximum jump length of particles. If it is set to None, FreeTrace infers its maximum length with GMM, otherwise this value is fixed to the given value.
        The inferred maximum jump length is limited under diffraction light limit of particles in SPT, if you use FreeTrace for non-SPT particles, please set this value manually.
        
        gpu_on:
        Perform neural network enhanced trajectory inference assuming fractional Brownian motion. With False, FreeTrace infers the trajectory assuming standard Brownian motion.
        
        save_video:
        Save and visualize the reconstructed trajectories. (video_traces.tiff)
        
        verbose:
        Print the process.
        
        realtime_visualization:
        Real time visualization of process.

    @return
        return:
        It returns True if the tracking of particles is finished succesfully, False otherwise.
        The unit of saved particle coordinate is pixel.
    """

    from multiprocessing import Process, Value
    return_state = Value('b', 0)
    options = {
        'time_forecast': time_forecast,
        'cutoff': cutoff,
        'jump_threshold': jump_threshold,
        'gpu_on': gpu_on,
        'save_video': save_video,
        'verbose': verbose,
        'batch': batch,
        'return_state': return_state,
        'realtime_visualization': realtime_visualization
    }
    
    p = Process(target=run, args=(input_video_path, output_path),  kwargs=options)
    try:
        p.start()
        p.join()
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating childs")
        p.terminate()
        p.join()
    finally:
        p.close()
    return return_state.value
