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
from FreeTrace.module.trajectory_object import TrajectoryObj
from FreeTrace.module.image_module import read_tif, make_image_seqs, make_whole_img
from FreeTrace.module.xml_module import write_xml
from FreeTrace.module.data_load import read_localization
from FreeTrace.module.data_save import write_trajectory, write_trxyt
from FreeTrace.module.auxiliary import initialization
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


def select_opt_graph2(saved_graph:nx.graph, next_graph:nx.graph, localizations, next_times, distribution, first_step, most_probable_jump_d):
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
    incr = 0
    while True:
        if VERBOSE:
            pbar_update = selected_time_steps[0] - saved_time_steps -1 + len(selected_time_steps)
            mysum += pbar_update
            PBAR.update(pbar_update)
        """
        print('------------- next graph ---------------')
        for path in paths_dfs(next_graph, source=source_node):
            print(path)
        """
        selected_sub_graph = select_opt_graph(light_prev_graph, next_graph, localization, selected_time_steps, distribution, first_construction, most_probable_jump_d)
        """
        print('@@@@@@@@@@@@@@ selected graph @@@@@@@@@@@@@@@')
        for path in paths_dfs(selected_sub_graph, source=source_node):
            print(path)
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        if incr == 1:
            exit()
        incr += 1
        """
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


def run(input_video, outpur_dir, blink_lag=1, cutoff=0, pixel_microns=1, frame_rate=1, gpu_on=True, save_video=False, verbose=False, batch=False, return_state=0):
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
    VISUALIZATION = save_video
    PIXEL_MICRONS = pixel_microns
    FRAME_RATE = frame_rate
    GPU_AVAIL = gpu_on
    REG_LEGNTHS = [5, 8, 12]
    ALPHA_MAX_LENGTH = 8
    CUDA, TF = initialization(GPU_AVAIL, REG_LEGNTHS, ptype=1, verbose=VERBOSE, batch=BATCH)
    POLY_FIT_DATA = np.load(f'{__file__.split("/Tracking.py")[0]}/models/theta_hat.npz')

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
    
    if return_state != 0:
        return_state.value = 1
    return True


def run_process(input_video, outpur_dir, blink_lag=1, cutoff=0, pixel_microns=1, frame_rate=1, gpu_on=True, save_video=False, verbose=False, batch=False):
    from multiprocessing import Process, Value
    return_state = Value('b', 0)
    options = {
        'blink_lag': blink_lag,
        'cutoff': cutoff,
        'pixel_microns': pixel_microns,
        'frame_rate': frame_rate,
        'gpu_on': gpu_on,
        'save_video': save_video,
        'verbose': verbose,
        'batch': batch,
        'return_state': return_state
    }
    
    p = Process(target=run, args=(input_video, outpur_dir),  kwargs=options)
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


##########################################################################################################################################################################

"""
def make_image(trajectory_list, output, cutoff=2, scale_factor=3, margins=(200, 200), color_seq=None, thickness=5, scale_bar_in_log10=None, background='black'):
    if scale_bar_in_log10 is None:
        scale_bar_in_log10 = [-3, 0]
    scale_factor = min(scale_factor, 3)
    factor = int(10**scale_factor)
    #factor = 20
    x_min = 99999
    y_min = 99999
    x_max = -1
    y_max = -1
    for traj in trajectory_list:
        xys = traj.get_positions()
        xmin = np.min(xys[:, 0])
        ymin = np.min(xys[:, 1])
        xmax = np.max(xys[:, 0])
        ymax = np.max(xys[:, 1])
        x_min = min(x_min, xmin)
        y_min = min(y_min, ymin)
        x_max = max(x_max, xmax)
        y_max = max(y_max, ymax)

    x_width = int(((x_max) * factor) + 1) + margins[0] * 2
    y_width = int(((y_max) * factor) + 1) + margins[1] * 2
    if background=='white':
        img = np.ones((y_width, x_width, 3)).astype(np.uint8) * 255
    else:
        img = np.ones((y_width, x_width, 3)).astype(np.uint8)
    print(f'Image pixel size :({x_width}x{y_width}) = ({np.round(x_width / factor, 3)}x{np.round(y_width / factor, 3)}) in micrometer')

    for traj in trajectory_list:
        if len(traj.get_positions()) >= cutoff:
            times = traj.get_times()
            indices = [i for i, time in enumerate(times)]
            pts = np.array([[int(x * factor) + int(margins[0]/2), int(y * factor) + int(margins[0]/2)] for x, y, _ in traj.get_positions()[indices]], np.int32)
            log_diff_coefs = np.log10(traj.get_inst_diffusion_coefs(1, t_range=None))
            for i in range(len(pts)-1):
                prev_pt = pts[i]
                next_pt = pts[i+1]
                log_diff_coef = log_diff_coefs[i]
                color = color_seq[int(((min(max(scale_bar_in_log10[0], log_diff_coef), - scale_bar_in_log10[-1]) - scale_bar_in_log10[0]) / (scale_bar_in_log10[-1] - scale_bar_in_log10[0])) * (len(color_seq) - 1))]
                color = (int(color[2]*255), int(color[1]*255), int(color[0]*255))  # BGR
                cv2.line(img, prev_pt, next_pt, color, thickness)

    cv2.imwrite(output, img) 
   

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) < 3:
        print('example of use:')
        print('python diffusion_image.py filename.csv pixelmicrons framerate')
        exit(1)

    if '.csv' not in args[0]:
        print(args)
        print('input trajectory file extension must be filename_traces.csv')
        exit(1)
    args[1] = float(args[1])
    args[2] = float(args[2])

    trajectory_length_cutoff = 0
    scale_factor = 2 # decide the resolution of image. (higher value produce better image). Max value must be lower than 3, if you don't have good RAMs.
    background_color = 'black'
    colormap = 'jet'  # matplotlib colormap
    thickness = 1  # thickness of line
    scale_bar_in_log10 = [-1.25, 0.25]   # linear color mapping of log10(diffusion coefficient) in range[-3, 0] micrometer^2/second, if log_diff_coef is < than -3, set it to the first color of cmap, if log_diff_coef is > than 0, set it to the last color of cmap
    margins = (1 * 10**scale_factor, 1 * 10**scale_factor)  # image margin in pixel

    mycmap = plt.get_cmap(colormap, lut=None)
    color_seq = [mycmap(i)[:3] for i in range(mycmap.N)][::-1]
    trajectory_list = read_trajectory(args[0], pixel_microns=args[1], frame_rate=args[2])

    make_image(trajectory_list, f'{args[0].split(".csv")[0]}_diffusion.png', cutoff=trajectory_length_cutoff, margins=margins, scale_factor=scale_factor, color_seq=color_seq, thickness=thickness, scale_bar_in_log10=scale_bar_in_log10, background=background_color)
"""




#################################################################################################################################################


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
from scipy.stats import multivariate_normal
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
    sigma_ = 4
    log_pdf = -1000
    abnormal = False
    multinormal_hash = {}
    alpha_index, k_index = indice_fetch(alpha, k)
    mean_std = std_fetch(alpha_index, k_index)
    if np.sqrt(np.sum(relativ_coord**2)) > sigma_*mean_std:
        abnormal = True
    if (mean_std, lag) not in multinormal_hash:
        multinormal_hash[(mean_std, lag)] = multivariate_normal(mean=[0, 0, 0], cov=[[mean_std*math.sqrt((lag+1)), 0, 0],
                                                                                     [0, mean_std*math.sqrt((lag+1)), 0],
                                                                                     [0, 0, mean_std*math.sqrt((lag+1))]], allow_singular=False)
    log_pdf = multinormal_hash[(mean_std, lag)].logpdf(relativ_coord)
    return log_pdf, abnormal


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
    min_euclid = 5.0

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
            approx[t] = max_euclid  #TODO increase over time? well...
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


def find_paths(G, source=(0, 0), path=None, seen=None):
    if path is None:
        path = [source]
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
                yield from find_paths(G, n, path+[n], seen.union([n]))


def predict_alphas(x, y):
    pred_alpha = REG_MODEL.alpha_predict(np.array([x, y]))
    return pred_alpha


def predict_ks(x, y):
    pred_logk = REG_MODEL.k_predict([np.array([x, y])])
    return pred_logk[0]


def predict_long_seq(next_path, trajectories_costs, localizations, prev_alpha, prev_k, next_times, prev_path=None, start_indice=None):
    abnormal = False
    abnormal_penalty = 1000
    time_penalty = abnormal_penalty / TIME_FORECAST
    time_score = 0
    abnomral_jump_score = 0
    traj_cost = []
    ab_index = []
    if 0.9999 < prev_alpha < 1.0001 and 1.9999 < prev_k < 2.0001:
        abnomral_jump_score += abnormal_penalty

    cutting_threshold = 2 * abnormal_penalty
    initial_cost = cutting_threshold - 10
    
    if trajectories_costs[next_path] is not None:
        return ab_index

    for idx in range(1, len(next_path) - 1):
        if (next_path[idx+1][0] - next_path[idx][0]) - 1 > TIME_FORECAST:
            trajectories_costs[next_path] = initial_cost
            return [idx]

    if len(next_path) <= 1:
        raise Exception
    elif len(next_path) == 2:
        trajectories_costs[next_path] = initial_cost
    else:
        start_index = start_indice[next_path]
        if start_index == 1 or start_index == 2:
            if abnormal:
                prev_alpha = 1.0
                prev_k = 2.0
            before_node = next_path[1]
            next_node = next_path[2]
            time_gap = next_node[0] - before_node[0] - 1
            next_coord = localizations[next_node[0]][next_node[1]]
            cur_coord = localizations[before_node[0]][before_node[1]]
            dir_vec_before = np.array([1, 1, 1])
            estim_mu = (time_gap + 1) * pdf_mu_measure(prev_alpha) * dir_vec_before + cur_coord
            input_mu = next_coord - estim_mu
            log_p0, abnormal = predict_multinormal(input_mu, prev_alpha, prev_k, time_gap)

            if abnormal:
                abnomral_jump_score += abnormal_penalty
                ab_index.append(1)
            time_score += time_gap * time_penalty
            traj_cost.append(abs(log_p0))

            for edge_index in range(3, len(next_path)):
                if abnormal:
                    prev_alpha = 1.0
                    prev_k = 2.0
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
                log_p0, abnormal = predict_multinormal(input_mu, prev_alpha, prev_k, time_gap)
                
                if abnormal:
                    abnomral_jump_score += abnormal_penalty
                    ab_index.append(edge_index - 1)
                time_score += time_gap * time_penalty
                traj_cost.append(abs(log_p0))

        elif start_index >= 3:
            for edge_index in range(3, len(next_path)):
                if abnormal:
                    prev_alpha = 1.0
                    prev_k = 2.0
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
                log_p0, abnormal = predict_multinormal(input_mu, prev_alpha, prev_k, time_gap)
                
                if abnormal:
                    abnomral_jump_score += abnormal_penalty
                    ab_index.append(edge_index - 1)
                time_score += time_gap * time_penalty
                traj_cost.append(abs(log_p0))
        else:
            sys.exit("Untreated exception, check trajectory inference method again.")

        if len(traj_cost) > 1:
            final_score = np.mean(traj_cost[:-1]) + abnomral_jump_score + time_score
        else:
            final_score = abnomral_jump_score + time_score
        trajectories_costs[next_path] = final_score
        #print(trajectories_costs[next_path], traj_cost, abnomral_jump_score, time_score, ab_index, next_path, prev_alpha, prev_k, pdf_mu_measure(prev_alpha), start_index)

    if trajectories_costs[next_path] > cutting_threshold:
        return ab_index
    else:
        return []
    

def generate_next_paths(next_graph:nx.graph, init_graph:nx.graph, localizations, next_times, distribution, source_node):
    while True:
        start_g_len = len(next_graph.nodes)
        index = 0
        cumulative_last_nodes = []
        while True:
            last_nodes = list([nodes[-1] for nodes in find_paths(next_graph, source=source_node)])
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
    return next_graph


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


def select_opt_graph2(final_graph:nx.graph, saved_graph:nx.graph, next_graph:nx.graph, localizations, next_times, distribution, first_step):
    selected_graph = nx.DiGraph()
    source_node = (0, 0)
    selected_graph.add_node(source_node)
    alpha_values = {}
    k_values = {}
    start_indice = {}
    hashed_prev_next = {}
    init_graph = final_graph.copy()

    if not first_step:
        prev_paths = list(find_paths(saved_graph, source=source_node))
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
    next_graph = generate_next_paths(next_graph, init_graph, localizations, next_times, distribution, source_node)
    trajectories_costs = {tuple(next_path):None for next_path in find_paths(next_graph, source=source_node)}

    while True:
        orphans = []
        ab_indice = {}
        cost_copy = {}

        # Cost copy, conversion to tuple, start_index write.
        next_paths = list(find_paths(next_graph, source=source_node))
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
                ab_index = predict_long_seq(next_path, trajectories_costs, localizations, 1.0, 2.0, next_times, start_indice=start_indice)
                if len(ab_index) > 0:
                    ab_indice[next_path] = ab_index 

        else:
            for next_path in next_paths:
                prev_path = match_prev_next(prev_paths, next_path, hashed_prev_next)
                if prev_path is None:
                    ab_index = predict_long_seq(next_path, trajectories_costs, localizations, 1.0, 2.0, next_times, start_indice=start_indice)
                else:
                    prev_alpha = alpha_values[prev_path]
                    prev_k = k_values[prev_path]
                    ab_index = predict_long_seq(next_path, trajectories_costs, localizations, prev_alpha, prev_k, next_times, prev_path, start_indice=start_indice)
                if len(ab_index) > 0:
                    ab_indice[next_path] = ab_index 

        # Cost sorting
        trajs = [path for path in trajectories_costs.keys()]
        costs = [trajectories_costs[path] for path in trajectories_costs.keys()]
        low_cost_args = np.argsort(costs)
        next_trajectories = np.array(trajs, dtype=object)[low_cost_args]
        lowest_cost_traj = list(next_trajectories[0])
        for i in range(len(lowest_cost_traj)):
            lowest_cost_traj[i] = tuple(lowest_cost_traj[i])

        #for cost, traj in zip(np.array(costs)[low_cost_args][::-1], next_trajectories[::-1]):
        #    print(f'{traj} -> {cost}')

        # Abnormal trajectory cutting
        if tuple(lowest_cost_traj) in ab_indice:
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

            next_paths = list(find_paths(next_graph, source=source_node))
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
        
        # add edges from source to orphans
        for rm_node in lowest_cost_traj[1:]:
            for neighbor in next_graph.neighbors(rm_node):
                if neighbor not in lowest_cost_traj:
                    orphans.append(neighbor)
        next_graph.remove_nodes_from(lowest_cost_traj[1:])
        for orphan_node in orphans:
            if not nx.has_path(next_graph, source_node, orphan_node):
                next_graph.add_edge(source_node, orphan_node)
        pop_cost = trajectories_costs.pop(tuple(lowest_cost_traj))

        # selected graph update
        for edge_index in range(1, len(lowest_cost_traj)):
            before_node = lowest_cost_traj[edge_index - 1]
            next_node = lowest_cost_traj[edge_index]
            selected_graph.add_edge(before_node, next_node)

        # escape loop
        if len(next_graph) == 1:
            break
            
        # newborn cost update
        for next_path in find_paths(next_graph, source=source_node):
            next_path = tuple(next_path)
            if next_path not in trajectories_costs:
                trajectories_costs[next_path] = None

    return selected_graph


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
    selected_time_steps = np.arange(t_avail_steps[0] + 1, t_avail_steps[0] + 1 + time_forecast)
    saved_time_steps = 1
    mysum = 0

    realtime_obj = None
    if realtime_visualization:
        from FreeTrace.module.image_module import RealTimePlot
        realtime_obj = RealTimePlot('Tracking', job_type='track', show_frame=True)
        realtime_obj.turn_on()

    while True:
        node_pairs = []
        start_time = selected_time_steps[-1]
        if VERBOSE:
            pbar_update = selected_time_steps[0] - saved_time_steps -1 + len(selected_time_steps)
            mysum += pbar_update
            PBAR.update(pbar_update)

        if len(set(selected_time_steps).intersection(set(t_avail_steps))) != 0:
            selected_sub_graph = select_opt_graph2(final_graph, light_prev_graph, next_graph, localization, selected_time_steps, distribution, first_construction)
        else:
            selected_sub_graph = nx.DiGraph()
            selected_sub_graph.add_node(source_node)

        first_construction = False
        light_prev_graph = nx.DiGraph()
        light_prev_graph.add_node(source_node)
        if len(selected_sub_graph.nodes) > 1:
            if last_time in selected_time_steps:
                if VERBOSE:
                    PBAR.update(image_length - mysum)
                for path in find_paths(selected_sub_graph, source=source_node):
                    without_source_path = path[1:]
                    if len(without_source_path) == 1:
                        if without_source_path[0] not in final_graph.nodes:
                            final_graph.add_edge(source_node, without_source_path[0])
                    else:
                        for idx in range(len(without_source_path) - 1):
                            before_node = without_source_path[idx]
                            next_node = without_source_path[idx+1]
                            if (before_node, next_node) not in final_graph.edges:
                                final_graph.add_edge(before_node, next_node)
                break
            else:
                for path in find_paths(selected_sub_graph, source=source_node):
                    if len(path) == 2:
                        if path[-1][0] < selected_time_steps[-1] - TIME_FORECAST:
                            final_graph.add_edge(source_node, path[-1])
                        else:
                            start_time = min(start_time, path[-1][0])
                    else:
                        if path[-2][0] >= selected_time_steps[-1] - TIME_FORECAST:
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
        
        if last_time in selected_time_steps:
            if VERBOSE:
                PBAR.update(image_length - mysum)
            break

        if realtime_visualization:
            realtime_obj.put_into_queue((IMAGES, list(find_paths(final_graph, source=source_node)), selected_time_steps[:-1], localization), mod_n=1)
        
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
            print('Missing node: ', node_, ' possible errors on tracking.')

    if realtime_obj is not None:
        realtime_obj.turn_off()

    trajectory_list = []
    traj_idx = 0
    for path in find_paths(final_graph, source=source_node):
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


    VERBOSE = verbose
    BATCH = batch
    TIME_FORECAST = max(1, min(10, time_forecast))
    CUTOFF = cutoff
    GPU_AVAIL = gpu_on
    REG_LEGNTHS = [3, 5, 8]
    ALPHA_MAX_LENGTH = 10
    JUMP_THRESHOLD = jump_threshold
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

    final_trajectories = trajectory_inference(localization=loc, time_steps=t_steps,
                                              distribution=max_jumps, image_length=images.shape[0], realtime_visualization=realtime_visualization)
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

