import os
import sys
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import lru_cache
from itertools import product
import networkx as nx
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.model_selection import GridSearchCV
from scipy.stats import multivariate_normal
from scipy.stats import rv_histogram
from FreeTrace.module.TrajectoryObject import TrajectoryObj
from FreeTrace.module.ImageModule import read_tif, make_image_seqs, make_whole_img
from FreeTrace.module.XmlModule import write_xml
from FreeTrace.module.FileIO import write_trajectory, read_localization, read_parameters, write_trxyt, initialization
from timeit import default_timer as timer


@lru_cache
def pdf_mu_measure(alpha):
    idx = int((alpha / POLY_FIT_DATA['alpha'][-1]) * (len(POLY_FIT_DATA['alpha']) - 1))
    return POLY_FIT_DATA['mu'][idx]


@lru_cache
def pdf_std_measure(alpha):
    idx = int((alpha / POLY_FIT_DATA['alpha'][-1]) * (len(POLY_FIT_DATA['alpha']) - 1))
    return POLY_FIT_DATA['std'][idx]


def log_p_multi(relativ_coord, alpha, lag):
    idx = int((alpha / POLY_FIT_DATA['alpha'][-1]) * (len(POLY_FIT_DATA['alpha']) - 1))
    return MULTI_NORMALS[lag][idx].logpdf(relativ_coord)


def greedy_shortest(srcs, dests, lag):
    srcs = np.array(srcs)
    dests = np.array(dests)
    distribution = []
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
    segment_lengths = euclidian_displacement(euclid_tmp0, euclid_tmp1)
    if segment_lengths is not None:
        for (i, dest), segment_length in zip(combs, segment_lengths):
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
            distribution.append(linkage[src][dest])

    diffraction_light_limit = 15.0 * np.power(lag + 1, (1/4))  #TODO: diffraction light limit
    filtered_distrib = []
    if len(distribution) > 2:
        for jump_d in distribution[:-1]:
            if jump_d < diffraction_light_limit:
                filtered_distrib.append(jump_d)
    else:
        for jump_d in distribution:
            if jump_d < diffraction_light_limit:
                filtered_distrib.append(jump_d)
    return filtered_distrib


def segmentation(localization: dict, time_steps: np.ndarray, lag=2):
    lag = min(lag, len(time_steps) - 2)
    seg_distribution = {}
    for i in range(lag + 1):
        seg_distribution[i] = []
    for i, time_step in enumerate(time_steps[:-lag-1:1]):
        dests = [[] for _ in range(lag + 1)]
        srcs = localization[time_step]
        for j in range(i+1, i+lag+2):
            dest = localization[time_steps[j]]
            dests[j - i - 1].extend(dest)
        for l, dest in enumerate(dests):
            dist = greedy_shortest(srcs=srcs, dests=dest, lag=l)
            seg_distribution[l].extend(dist)
    return seg_distribution


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


def euclidian_displacement(pos1, pos2):
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


def approx_cdf(distribution, conf, bin_size, approx, n_iter, burn):
    #resample_nb = 3000
    #resampled = distribution[np.random.randint(0, len(distribution), min(resample_nb, len(distribution)))]
    resampled = distribution

    bin_size *= 2
    length_max_val = np.max(resampled)
    bins = np.arange(0, length_max_val + bin_size, bin_size)
    kdes = []
    param_grid = {
        "n_components": [1, 2, 3],
    }
    grid_search = GridSearchCV(
        GaussianMixture(max_iter=1000, n_init=10, covariance_type='diag'), param_grid=param_grid,
        scoring=gmm_bic_score
    )
    grid_search.fit(resampled.reshape(-1, 1))
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
    opt_nb_component = np.argmin(cluster_df["BIC score"]) + param_grid['n_components'][0]
    cluster = BayesianGaussianMixture(n_components=opt_nb_component, max_iter=1000, n_init=20,
                                      mean_precision_prior=1e-7,
                                      covariance_type='diag').fit(resampled.reshape(-1, 1))
    #print('MEANS: ', cluster.means_, '\nCOVS: ', cluster.covariances_, '\nWEIGHTS: ', cluster.weights_)

    max_diffusive = 0
    for mean, cov, weight in zip(cluster.means_.flatten(), cluster.covariances_.flatten(), cluster.weights_.flatten()):
        sample = np.random.normal(loc=mean, scale=cov, size=10000)
        kde = KernelDensity(kernel="gaussian", bandwidth=0.75).fit(sample.reshape(-1, 1))
        kdes.append(kde)
        if cov > 2.5:  #TODO: diffraction light limit
            continue

        if weight > 0.1:
            max_diffusive = max(max_diffusive, mean + 2.5*cov)

    hist = np.histogram(resampled, bins=bins)
    hist_dist = rv_histogram(hist)
    pdf = hist[0] / np.sum(hist[0])
    bin_edges = hist[1]
    pdf = np.where(pdf > 0.0005, pdf, 0)
    pdf = pdf / np.sum(pdf)

    if approx == 'metropolis_hastings':
        resampled = metropolis_hastings(pdf, n_iter=n_iter, burn=burn) * bin_size
        reduced_bins = np.arange(0, length_max_val + bin_size, bin_size)
        hist = np.histogram(resampled, bins=reduced_bins)
        hist_dist = rv_histogram(hist)
        pdf = hist[0] / np.sum(hist[0])
        bin_edges = hist[1]
    return max_diffusive, pdf, bin_edges, hist_dist.cdf, resampled, kdes, cluster


def approximation(real_distribution, conf, bin_size, approx='metropolis_hastings', n_iter=1e6, burn=0, thresholds=None):
    for lag_key in real_distribution:
        real_distribution[lag_key] = np.array(real_distribution[lag_key])
    approx_distribution = {}
    n_iter = int(n_iter)
    for index, lag in enumerate(real_distribution.keys()):
        seg_len_obv, pdf_obv, bins_obv, cdf_obv, distrib, kdes, cluster = (
            approx_cdf(distribution=real_distribution[lag],
                        conf=conf, bin_size=bin_size, approx=approx, n_iter=n_iter, burn=burn))
        if thresholds is not None:
            approx_distribution[lag] = [thresholds[index], pdf_obv, bins_obv, cdf_obv, distrib, kdes, cluster]
        else:
            approx_distribution[lag] = [seg_len_obv, pdf_obv, bins_obv, cdf_obv, distrib, kdes, cluster]

    #########################
    if thresholds == None:
        max_length_0 = approx_distribution[0][0]
        if max_length_0 <= 0:
            max_length_0 = 1
        alpha = min(4, 4 / max_length_0)  # TODO: consideration
        for index, lag in enumerate(real_distribution.keys()):
            approx_distribution[lag][0] = max_length_0 * np.power(index + 1, (1/3)) + alpha  # TODO: consideration
    #########################

    bin_max = -1
    for lag in real_distribution.keys():
        bin_max = max(bin_max, len(approx_distribution[lag][1]))
    for lag in real_distribution.keys():
        for index in [1, 2]:
            if index == 1:
                tmp = np.zeros(bin_max)
                tmp[:len(approx_distribution[lag][index]) - index] = approx_distribution[lag][index][:-1 - index + 1]
            else:
                tmp = np.arange(0, 1000, approx_distribution[lag][index][1] - approx_distribution[lag][index][0])[:bin_max]
                tmp[:len(approx_distribution[lag][index]) - index + 1] = approx_distribution[lag][index][:-1 - index + 2]
            approx_distribution[lag][index] = tmp
    return approx_distribution


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


def dfs_edges(G, source=None, depth_limit=None):
    paths = []
    if source is None:
        nodes = G
    else:
        nodes = [source]
    visited = set()
    if depth_limit is None:
        depth_limit = len(G)
    for start in nodes:
        if start in visited:
            continue
        visited.add(start)
        stack = [(start, depth_limit, iter(G[start]))]
        while stack:
            parent, depth_now, children = stack[-1]
            try:
                child = next(children)
                flag = True
                if depth_now > 1:
                    stack.append((child, depth_now - 1, iter(G[child])))
            except StopIteration:
                if flag:
                    path_list = [node[0] for node in stack]
                    if len(path_list) >= 2:
                        paths.append(path_list)
                flag = False
                stack.pop()
    return paths


def paths_dfs(G, source=None):
    paths = []
    node_tuple_gen = nx.edge_dfs(G, source=source)
    first_tuple = next(node_tuple_gen)
    path = [first_tuple[0], first_tuple[1]]
    while True:
        try:
            node_tuple = next(node_tuple_gen)
            if node_tuple[0] == source:
                paths.append(path)
                path = [node_tuple[0], node_tuple[1]]
            else:
                if path[-1] == node_tuple[0]:
                    path.append(node_tuple[1])
                else:
                    paths.append(path)
                    path_copy = path.copy()
                    path = []
                    for node in path_copy:
                        if node == node_tuple[0]:
                            path.append(node)
                            break
                        else:
                            path.append(node)
                    path.append(node_tuple[1])
        except StopIteration:
            paths.append(path)
            break
    return paths



def predict_alphas(x, y):
    pred_alpha = REG_MODEL.alpha_predict(np.array([x, y]))
    return pred_alpha


def predict_long_seq(next_path, trajectories_costs, localizations, prev_alpha, initial_cost, next_times):
    if next_path in trajectories_costs and trajectories_costs[next_path] < initial_cost - 1:
        return
    else:
        if len(next_path) <= 1:
            raise Exception
        elif len(next_path) == 2:
            trajectories_costs[next_path] = initial_cost
        elif len(next_path) == 3 and next_path[2][0] in next_times:
            before_node = next_path[1]
            next_node = next_path[2]
            time_gap = next_node[0] - before_node[0] - 1
            next_coord = localizations[next_node[0]][next_node[1]]
            cur_coord = localizations[before_node[0]][before_node[1]]
            dir_vec_before = np.array([0, 0, 0])
            estim_mu = (time_gap + 1) * pdf_mu_measure(prev_alpha) * dir_vec_before + cur_coord
            input_mu = next_coord - estim_mu
            log_p0 = log_p_multi(input_mu, prev_alpha, time_gap)
            log_p0 = abs(log_p0)
            trajectories_costs[next_path] = log_p0
        elif len(next_path) == 3 and next_path[2][0] not in next_times:
            trajectories_costs[next_path] = initial_cost
        else:
            if len(next_path) >= 4:
                traj_cost = []
                for edge_index in range(3, len(next_path)):
                    bebefore_node = next_path[edge_index - 2]
                    before_node = next_path[edge_index - 1]
                    next_node = next_path[edge_index]
                    time_gap = next_node[0] - before_node[0] - 1
                    next_coord = localizations[next_node[0]][next_node[1]]
                    cur_coord = localizations[before_node[0]][before_node[1]]
                    before_coord = localizations[bebefore_node[0]][bebefore_node[1]]
                    dir_vec_before = cur_coord - before_coord
                    estim_mu = (time_gap + 1) * pdf_mu_measure(prev_alpha) * dir_vec_before + cur_coord
                    input_mu = next_coord - estim_mu
                    log_p0 = log_p_multi(input_mu, prev_alpha, time_gap)
                    log_p0 = abs(log_p0)
                    traj_cost.append(log_p0)
                traj_cost = np.mean(traj_cost)
                trajectories_costs[next_path] = traj_cost
            else:
                sys.exit("Untreated exception, check trajectory inference method again.")


def select_opt_graph(saved_graph:nx.graph, next_graph:nx.graph, localizations, next_times, distribution, first_step, most_probable_jump_d):
    initial_cost = 1000
    selected_graph = nx.DiGraph()
    source_node = (0, 0)
    alpha_values = {}
    
    if not first_step:
        prev_paths = paths_dfs(saved_graph, source=source_node)
        if TF:
            for path_idx in range(len(prev_paths)):
                prev_xys = np.array([localizations[txy[0]][txy[1]][:2] for txy in prev_paths[path_idx][1:]])[-ALPHA_MAX_LENGTH:]
                prev_x_pos = prev_xys[:, 0]
                prev_y_pos = prev_xys[:, 1]
                prev_alpha = predict_alphas(prev_x_pos, prev_y_pos)
                alpha_values[tuple(prev_paths[path_idx])] = prev_alpha
                prev_paths[path_idx] = tuple(prev_paths[path_idx])
        else:
            for path_idx in range(len(prev_paths)):
                alpha_values[tuple(prev_paths[path_idx])] = 1.0
                prev_paths[path_idx] = tuple(prev_paths[path_idx])

    while True:
        start_g_len = len(next_graph)
        index = 0
        while True:
            last_nodes = list([nodes[-1] for nodes in paths_dfs(next_graph, source=source_node)])
            for last_node in last_nodes:
                for cur_time in next_times[index:index+1]:
                    if last_node[0] < cur_time:
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
                            jump_d_mat = euclidian_displacement(jump_d_pos1, jump_d_pos2)
                            local_idx = 0
                            for next_idx, loc in enumerate(localizations[cur_time]):
                                if len(loc) == 3 and len(node_loc) == 3:
                                    jump_d = jump_d_mat[local_idx]
                                    local_idx += 1
                                    time_gap = cur_time - last_node[0] - 1
                                    if time_gap in distribution:
                                        threshold = distribution[time_gap][0]
                                        if jump_d < threshold:
                                            next_node = (cur_time, next_idx)
                                            next_graph.add_edge(last_node, next_node, jump_d=jump_d)
            for cur_time in next_times[index:index+1]:
                for idx in range(len(localizations[cur_time])):
                    if (cur_time, idx) not in next_graph and len(localizations[cur_time][0]) == 3:
                        next_graph.add_edge((0, 0), (cur_time, idx), jump_d=most_probable_jump_d)
            index += 1
            if index == len(next_times):
                break
        end_g_len = len(next_graph)
        if start_g_len == end_g_len:
            break

    t_time = timer()
    trajectories_costs = {tuple(next_path):initial_cost for next_path in paths_dfs(next_graph, source=source_node)}
    while True:
        cost_copy = {}
        next_paths = paths_dfs(next_graph, source=source_node)
        for path_idx in range(len(next_paths)):
            next_paths[path_idx] = tuple(next_paths[path_idx])
        for next_path in next_paths:
            if next_path in trajectories_costs:
                cost_copy[next_path] = trajectories_costs[next_path]
        trajectories_costs = cost_copy

        if first_step:
            for next_path in next_paths:
                predict_long_seq(next_path, trajectories_costs, localizations, 1.0, initial_cost, next_times)
        else:
            for prev_path in prev_paths:
                prev_alpha = alpha_values[prev_path]
                for next_path in next_paths:
                    if prev_path[-1] in next_path:
                        predict_long_seq(next_path, trajectories_costs, localizations, prev_alpha, initial_cost, next_times)

        trajs = [path for path in trajectories_costs.keys()]
        costs = [trajectories_costs[path] for path in trajectories_costs.keys()]
        low_cost_args = np.argsort(costs)
        next_trajectories = np.array(trajs, dtype=object)[low_cost_args]
        lowest_cost_traj = list(next_trajectories[0])
        for i in range(len(lowest_cost_traj)):
            lowest_cost_traj[i] = tuple(lowest_cost_traj[i])

        for rm_node in lowest_cost_traj[1:]:
            predcessors = list(next_graph.predecessors(rm_node)).copy()
            sucessors = list(next_graph.successors(rm_node)).copy()
            next_graph_copy = next_graph.copy()
            next_graph_copy.remove_node(rm_node)
            for pred in predcessors:
                for suc in sucessors:
                    if (pred, rm_node) in next_graph.edges and (rm_node, suc) in next_graph.edges and not nx.has_path(next_graph_copy, pred, suc):
                        if pred == source_node and not nx.has_path(next_graph_copy, source_node, suc):
                            next_graph.add_edge(pred, suc, jump_d=most_probable_jump_d)
                        else:
                            if pred in next_graph and suc in next_graph and pred != (0, 0):
                                pred_loc = localizations[pred[0]][pred[1]]
                                suc_loc = localizations[suc[0]][suc[1]]
                                jump_d = euclidian_displacement(pred_loc, suc_loc)
                                time_gap = suc[0] - pred[0] - 1
                                if time_gap in distribution:
                                    threshold = distribution[time_gap][0]
                                    if jump_d < threshold:
                                        next_graph.add_edge(pred, suc, jump_d=jump_d)

        next_graph.remove_nodes_from(lowest_cost_traj[1:])
        trajectories_costs.pop(tuple(lowest_cost_traj))

        # selected graph update
        for edge_index in range(1, len(lowest_cost_traj)):
            before_node = lowest_cost_traj[edge_index - 1]
            next_node = lowest_cost_traj[edge_index]
            selected_graph.add_edge(before_node, next_node)

        # escape loop
        if len(next_graph) == 1:
            break

        # newborn cost update
        for next_path in paths_dfs(next_graph, source=source_node):
            next_path = tuple(next_path)
            if next_path not in trajectories_costs:
                trajectories_costs[next_path] = initial_cost

    #print(f'\n{"cost calcul":<35}:{(timer() - t_time):.2f}s')
    return selected_graph


def forecast(localization: dict, t_avail_steps, distribution, blink_lag):
    first_construction = True
    last_time = t_avail_steps[-1]
    source_node = (0, 0)
    time_forecast = TIME_FORECAST
    max_pause_time = blink_lag + 1
    final_graph = nx.DiGraph()
    light_prev_graph = nx.DiGraph()
    next_graph = nx.DiGraph()
    final_graph.add_node(source_node)
    light_prev_graph.add_node(source_node)
    next_graph.add_node(source_node)
    initial_time_gap = 0
    most_probable_jump_d = distribution[initial_time_gap][6].means_[np.argmax(distribution[initial_time_gap][6].weights_)][0]
    next_graph.add_edges_from([((0, 0), (t_avail_steps[0], index), {'jump_d':most_probable_jump_d}) for index in range(len(localization[t_avail_steps[0]]))])
    selected_time_steps = np.arange(t_avail_steps[0] + 1, t_avail_steps[0] + 1 + time_forecast)
    saved_time_steps = 1
    mysum = 0

    while True:
        if VERBOSE:
            pbar_update = selected_time_steps[0] - saved_time_steps -1 + len(selected_time_steps)
            mysum += pbar_update
            PBAR.update(pbar_update)

        selected_sub_graph = select_opt_graph(light_prev_graph, next_graph, localization, selected_time_steps, distribution, first_construction, most_probable_jump_d)
        light_prev_graph = nx.DiGraph()
        light_prev_graph.add_node(source_node)
        last_nodes = []
        second_last_nodes = []
        if first_construction:
            start_index = 1
        else:
            start_index = 3
        for path in paths_dfs(selected_sub_graph, source=source_node):
            if len(path) <= 3:
                for i in range(1, len(path)):
                    before_node = path[i-1]
                    next_node = path[i]
                    if next_node not in final_graph.nodes:
                        final_graph.add_edge(before_node, next_node)
            else:
                for edge_index in range(start_index, len(path)):
                    before_node = path[edge_index - 1]
                    next_node = path[edge_index]
                    if before_node in final_graph:
                        final_graph.add_edge(before_node, next_node)
                    else:
                        final_graph.add_edge(source_node, before_node)
                        final_graph.add_edge(before_node, next_node)
            if selected_time_steps[-1] - path[-1][0] < max_pause_time: 
                last_nodes.append(path[-1])
                second_last_nodes.append(path[-2])

                ancestors = list(nx.ancestors(final_graph, path[-1]))
                sorted_ancestors = sorted(ancestors, key=lambda tup: tup[0], reverse=True)
                light_prev_graph.add_edge(sorted_ancestors[0], path[-1])
                if len(sorted_ancestors) > 1:
                    for idx in range(len(sorted_ancestors[:ALPHA_MAX_LENGTH+3]) - 1):
                        light_prev_graph.add_edge(sorted_ancestors[idx+1], sorted_ancestors[idx])
                    if sorted_ancestors[idx+1] != source_node:
                        light_prev_graph.add_edge(source_node, sorted_ancestors[idx+1])

        if last_time in selected_time_steps:
            if VERBOSE:
                PBAR.update(TIME_STEPS[-1] - mysum)
            break

        saved_time_steps = selected_time_steps[-1]
        next_first_time = selected_time_steps[-1] + 1
        next_graph = nx.DiGraph()
        next_graph.add_node(source_node)
        if len(last_nodes) == 0:
            first_construction = True
            while True:
                if next_first_time in t_avail_steps:
                    break
                next_first_time += 1
            selected_time_steps = [t for t in range(next_first_time, min(last_time + 1, next_first_time + time_forecast))]
            next_graph.add_edges_from([(source_node, (next_first_time, index), {'jump_d':most_probable_jump_d}) for index in range(len(localization[next_first_time]))])
        else:
            first_construction = False
            selected_time_steps = [t for t in range(next_first_time, min(last_time + 1, next_first_time + time_forecast))]
            for last_node, second_last_node in zip(last_nodes, second_last_nodes):
                if second_last_node == source_node:
                    next_graph.add_edge(source_node, last_node, jump_d=most_probable_jump_d)
                else:
                    last_xyz = localization[last_node[0]][last_node[1]]
                    second_last_xyz = localization[second_last_node[0]][second_last_node[1]]
                    next_graph.add_edge(source_node, second_last_node, jump_d=most_probable_jump_d)
                    next_graph.add_edge(second_last_node, last_node, jump_d=math.sqrt((last_xyz[0] - second_last_xyz[0])**2 + (last_xyz[1] - second_last_xyz[1])**2))

    all_nodes_ = []
    for t in list(localization.keys()):
        for nb_sample in range(len(localization[t])):
            if len(localization[t][nb_sample]) == 3:
                all_nodes_.append((t, nb_sample))
    
    for node_ in all_nodes_:
        if node_ not in final_graph:
            print('missing node: ', node_, ' possible errors on tracking.')

    trajectory_list = []
    for traj_idx, path in enumerate(paths_dfs(final_graph, source=source_node)):
        traj = TrajectoryObj(index=traj_idx, localizations=localization)
        for node in path[1:]:
            traj.add_trajectory_tuple(node[0], node[1])
        trajectory_list.append(traj)
    return trajectory_list


def trajectory_inference(localization: dict, time_steps: np.ndarray, distribution: dict, blink_lag=2):
    t_avail_steps = []
    for time in np.sort(time_steps):
        if len(localization[time][0]) == 3:
            t_avail_steps.append(time)
    trajectory_list = forecast(localization, t_avail_steps, distribution, blink_lag)
    return trajectory_list


def run(input_video, outpur_dir, blink_lag=1, cutoff=0, pixel_microns=1, frame_rate=1, gpu_on=True, visualization=False, verbose=False, batch=False):
    global VERBOSE
    global BATCH
    global INPUT_TIFF 
    global OUTPUT_DIR 
    global BLINK_LAG 
    global CUTOFF
    global VISUALIZATION 
    global PIXEL_MICRONS 
    global FRAME_RATE 
    global GPU_AVAIL
    global REG_LEGNTHS
    global ALPHA_MAX_LENGTH
    global CUDA
    global TF
    global POLY_FIT_DATA 
    global TIME_FORECAST
    global THRESHOLDS 
    global TIME_STEPS
    global PBAR
    global REG_MODEL
    global MULTI_NORMALS

    VERBOSE = verbose
    BATCH = batch
    INPUT_TIFF = input_video
    OUTPUT_DIR = outpur_dir
    BLINK_LAG = blink_lag
    CUTOFF = cutoff
    VISUALIZATION = visualization
    PIXEL_MICRONS = pixel_microns
    FRAME_RATE = frame_rate
    GPU_AVAIL = gpu_on
    REG_LEGNTHS = [5, 8, 12]
    ALPHA_MAX_LENGTH = 8
    CUDA, TF = initialization(GPU_AVAIL, REG_LEGNTHS, ptype=1, verbose=VERBOSE, batch=BATCH)
    POLY_FIT_DATA = np.load(f'{__file__.split('/Tracking.py')[0]}/models/theta_hat.npz')

    output_xml = f'{OUTPUT_DIR}/{INPUT_TIFF.split("/")[-1].split(".tif")[0]}_traces.xml'
    output_trj = f'{OUTPUT_DIR}/{INPUT_TIFF.split("/")[-1].split(".tif")[0]}_traces.csv'
    output_trxyt = f'{OUTPUT_DIR}/{INPUT_TIFF.split("/")[-1].split(".tif")[0]}_traces.trxyt'
    output_imgstack = f'{OUTPUT_DIR}/{INPUT_TIFF.split("/")[-1].split(".tif")[0]}_traces.tiff'
    output_img = f'{OUTPUT_DIR}/{INPUT_TIFF.split("/")[-1].split(".tif")[0]}_traces.png'

    MULTI_NORMALS = {}
    for lag in range(BLINK_LAG+1):
        MULTI_NORMALS[lag] = [multivariate_normal(mean=[0, 0, 0], cov=[[pdf_std_measure(alpha)*(lag+1), 0, 0],
                                                                       [0, pdf_std_measure(alpha)*(lag+1), 0],
                                                                       [0, 0, pdf_std_measure(alpha)*(lag+1)]], allow_singular=False) for alpha in POLY_FIT_DATA['alpha']]
    final_trajectories = []
    confidence = 0.95
    TIME_FORECAST = 1
    THRESHOLDS = None

    images = read_tif(INPUT_TIFF)
    if images.shape[0] <= 1:
        sys.exit('Image squence length error: Cannot track on a single image.')
    loc, loc_infos = read_localization(f'{OUTPUT_DIR}/{INPUT_TIFF.split("/")[-1].split(".tif")[0]}_loc.csv', images)

    if TF:
        if VERBOSE:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        else:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
        from FreeTrace.module.load_models import RegModel
        REG_MODEL = RegModel(REG_LEGNTHS)

    TIME_STEPS, mean_nb_per_time, xyz_min, xyz_max = count_localizations(loc)
    raw_jump_distribution = segmentation(loc, time_steps=TIME_STEPS, lag=BLINK_LAG)
    bin_size = np.mean(xyz_max - xyz_min) / 5000. 
    jump_distribution = approximation(raw_jump_distribution, confidence, bin_size, n_iter=1e3, burn=0, approx=None, thresholds=THRESHOLDS)

    if VERBOSE:
        print(f'Mean nb of molecules per frame: {mean_nb_per_time:.2f} molecules/frame')
        for lag in jump_distribution.keys():
            print(f'{lag}_limit_length: {jump_distribution[lag][0]}')
        PBAR = tqdm(total=TIME_STEPS[-1], desc="Tracking", unit="frame", ncols=120)

    """
    fig, axs = plt.subplots((BLINK_LAG + 1), 2, figsize=(20, 10))
    show_x_max = 20
    show_y_max = 0.30
    for lag in jump_distribution.keys():
        raw_segs_hist, bin_edges = np.histogram(raw_jump_distribution[lag], bins=np.arange(0, show_x_max, bin_size * 2))
        mcmc_segs_hist, _ = np.histogram(jump_distribution[lag][4], bins=bin_edges)
        axs[lag][1].hist(bin_edges[:-1], bin_edges, weights=raw_segs_hist / np.sum(raw_segs_hist), alpha=0.5)
        axs[lag][0].hist(bin_edges[:-1], bin_edges, weights=mcmc_segs_hist / np.sum(mcmc_segs_hist), alpha=0.5)
        for i in range(len(jump_distribution[lag][6].weights_)):
            axs[lag][0].plot(jump_distribution[lag][2], np.exp(jump_distribution[lag][5][i].score_samples(jump_distribution[lag][2].reshape(-1, 1))), label=f'{lag}_PDF')
        axs[lag][0].vlines(jump_distribution[lag][0], ymin=0, ymax=.14, alpha=0.6, colors='r', label=f'{lag}_limit')
        axs[lag][0].legend()
        axs[lag][1].legend()
        axs[lag][0].set_xlim([0, show_x_max])
        axs[lag][1].set_xlim([0, show_x_max])
        axs[lag][0].set_ylim([0, show_y_max])
        axs[lag][1].set_ylim([0, show_y_max])
    plt.show()
    """

    trajectory_list = trajectory_inference(localization=loc, time_steps=TIME_STEPS,
                                           distribution=jump_distribution, blink_lag=BLINK_LAG)
    for trajectory in trajectory_list:
        if not trajectory.delete(cutoff=CUTOFF):
            final_trajectories.append(trajectory)

    if VERBOSE:
        PBAR.close()

    #write_xml(output_file=output_xml, trajectory_list=final_trajectories, snr='7', density='low', scenario='Vesicle', cutoff=CUTOFF)
    write_trajectory(output_trj, final_trajectories)
    #write_trxyt(output_trxyt, final_trajectories, PIXEL_MICRONS, FRAME_RATE)
    make_whole_img(final_trajectories, output_dir=output_img, img_stacks=images)
    if VISUALIZATION:
        print(f'Visualizing trajectories...')
        make_image_seqs(final_trajectories, output_dir=output_imgstack, img_stacks=images, time_steps=TIME_STEPS, cutoff=CUTOFF,
                        add_index=False, local_img=None, gt_trajectory=None)
        
    return True
