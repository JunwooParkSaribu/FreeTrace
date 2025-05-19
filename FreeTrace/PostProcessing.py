import numpy as np
import itertools
from sklearn.mixture import GaussianMixture
from FreeTrace.module.trajectory_object import TrajectoryObj


def post_processing(trajectory_list, cutoff, verbose=0):
    gap_distrib = []
    filtered_traj_list = []
    post_processed_count = 0
    traj_index = 0
    
    for traj in trajectory_list:
        pos = traj.get_positions()
        if len(pos) >= cutoff:
            frames = traj.get_times()
            xs = pos[:, 0]
            ys = pos[:, 1]

            frame_sorted_args = np.argsort(frames)
            xs = xs[frame_sorted_args]
            ys = ys[frame_sorted_args]
            frames = frames[frame_sorted_args]

            pos = np.vstack([xs, ys]).T
            predicted_false_change_pair = []

            if len(xs) >= 5:
                #disps = np.sqrt((xs[1:] - xs[:-1])**2 + (ys[1:] - ys[:-1])**2)
                if len(xs) <= 10:
                    n_comp = 2
                elif len(xs) < 100:
                    n_comp = 3
                else:
                    n_comp = 4

                gm = GaussianMixture(n_components=n_comp, max_iter=1000, init_params='k-means++', covariance_type='spherical')
                gm = gm.fit(pos)
                labels = gm.predict(pos)
                change_label={}
                pairs = list(itertools.combinations(range(n_comp), r=2))
                for pair in pairs:
                    pair_distance = np.sqrt(np.sum((gm.means_[pair[0]] - gm.means_[pair[1]])**2))
                    if pair_distance < 1.5:
                        change_label[pair[1]] = pair[0]
                for key in sorted(list(change_label.keys()))[::-1]:
                    for idx in range(len(labels)):
                        if labels[idx] == key:
                            labels[idx] = change_label[key]
                means_for_labels = np.array([gm.means_[label] for label in labels])
                
                disps_for_means = np.sqrt(np.sum((pos - means_for_labels)**2, axis=1))

                gaps = []
                for label in sorted(np.unique(labels)):
                    check = np.array([idx for idx, lb in enumerate(labels) if lb==label])
                    gaps.append(np.mean(disps_for_means[check]))
                #gap /= len(disps_for_means)
                gap_distrib.append(np.max(gaps))
    gap_distrib = np.array(gap_distrib)
    if len(gap_distrib[gap_distrib < 1.0])/len(gap_distrib) > 0.8:
        seuil = 1.0
    else:
        seuil = 0.0

    for traj in trajectory_list:
        pos = traj.get_positions()
        if len(pos) >= cutoff:
            frames = traj.get_times()
            xs = pos[:, 0]
            ys = pos[:, 1]

            frame_sorted_args = np.argsort(frames)
            xs = xs[frame_sorted_args]
            ys = ys[frame_sorted_args]
            frames = frames[frame_sorted_args]

            pos = np.vstack([xs, ys]).T
            predicted_false_change_pair = []

            if len(xs) >= 5:
                #disps = np.sqrt((xs[1:] - xs[:-1])**2 + (ys[1:] - ys[:-1])**2)
                if len(xs) <= 10:
                    n_comp = 2
                elif len(xs) < 100:
                    n_comp = 3
                else:
                    n_comp = 4

                gm = GaussianMixture(n_components=n_comp, max_iter=1000, init_params='k-means++', covariance_type='spherical')
                gm = gm.fit(pos)
                labels = gm.predict(pos)
                change_label={}
                pairs = list(itertools.combinations(range(n_comp), r=2))
                for pair in pairs:
                    pair_distance = np.sqrt(np.sum((gm.means_[pair[0]] - gm.means_[pair[1]])**2))
                    if pair_distance < 1.5:
                        change_label[pair[1]] = pair[0]
                for key in sorted(list(change_label.keys()))[::-1]:
                    for idx in range(len(labels)):
                        if labels[idx] == key:
                            labels[idx] = change_label[key]
                means_for_labels = np.array([gm.means_[label] for label in labels])
                
                disps_for_means = np.sqrt(np.sum((pos - means_for_labels)**2, axis=1))

                gap = 0
                for label in sorted(np.unique(labels)):
                    check = np.array([idx for idx, lb in enumerate(labels) if lb==label])
                    gap += np.sum(disps_for_means[check])
                gap /= len(disps_for_means)

                unique_labels = sorted(np.unique(labels))
                pairs = list(itertools.combinations(unique_labels, r=2))
                change_label_distances = {(pair[0], pair[1]):[] for pair in pairs}
                prev_label = labels[0]
                for idx in range(1, len(pos)):
                    cur_label = labels[idx]
                    if cur_label != prev_label:
                        a, b = min(prev_label, cur_label), max(prev_label, cur_label)
                        change_label_distances[(a, b)].append(np.sqrt(np.sum((pos[idx] - pos[idx-1])**2)))
                    prev_label = cur_label
    
                for pair in pairs:
                    changing_distances = change_label_distances[(pair[0], pair[1])]
                    if gap < seuil:
                        predicted_false_change_pair.append(pair)


            if len(predicted_false_change_pair) > 0:
                post_processed_count += 1
                prev_label = labels[0]
                new_traj = TrajectoryObj(index=traj_index)
                new_traj.add_trajectory_position(frames[0], xs[0], ys[0], 0)
                for lb_idx in range(1, len(labels)):
                    cur_label = labels[lb_idx]
                    if (min(prev_label, cur_label), max(prev_label, cur_label)) in predicted_false_change_pair:
                        if len(new_traj.get_positions()) >= cutoff:
                            filtered_traj_list.append(new_traj)
                            traj_index += 1
                        new_traj = TrajectoryObj(index=traj_index)
                    new_traj.add_trajectory_position(frames[lb_idx], xs[lb_idx], ys[lb_idx], 0)
                    prev_label = cur_label
                if len(new_traj.get_positions()) >= cutoff:
                    filtered_traj_list.append(new_traj)
                    traj_index += 1
            else:
                new_traj = TrajectoryObj(index=traj_index)
                for idx in range(len(xs)):
                    new_traj.add_trajectory_position(frames[idx], xs[idx], ys[idx], 0)
                if len(new_traj.get_positions()) >= cutoff:
                    filtered_traj_list.append(new_traj)
                    traj_index += 1


    if verbose != 0:
        print(f'Post processed nb of trajectories: {post_processed_count} with ratio: {len(gap_distrib[gap_distrib < 1.0])/len(gap_distrib)}, seuil: {seuil}')
    return filtered_traj_list


def post_processing2(trajectory_list, cutoff, verbose=0):
    gap_distrib = []
    filtered_traj_list = []
    post_processed_count = 0
    traj_index = 0
    
    for traj in trajectory_list:
        pos = traj.get_positions()
        if len(pos) >= cutoff:
            frames = traj.get_times()
            xs = pos[:, 0]
            ys = pos[:, 1]

            frame_sorted_args = np.argsort(frames)
            xs = xs[frame_sorted_args]
            ys = ys[frame_sorted_args]
            frames = frames[frame_sorted_args]

            pos = np.vstack([xs, ys]).T
            predicted_false_change_pair = []

            if len(xs) >= 5:
                #disps = np.sqrt((xs[1:] - xs[:-1])**2 + (ys[1:] - ys[:-1])**2)
                if len(xs) <= 10:
                    n_comp = 2
                elif len(xs) < 100:
                    n_comp = 3
                else:
                    n_comp = 4

                gm = GaussianMixture(n_components=n_comp, max_iter=1000, init_params='k-means++', covariance_type='spherical')
                gm = gm.fit(pos)
                labels = gm.predict(pos)
                change_label={}
                pairs = list(itertools.combinations(range(n_comp), r=2))
                for pair in pairs:
                    pair_distance = np.sqrt(np.sum((gm.means_[pair[0]] - gm.means_[pair[1]])**2))
                    if pair_distance < 1.5:
                        change_label[pair[1]] = pair[0]
                for key in sorted(list(change_label.keys()))[::-1]:
                    for idx in range(len(labels)):
                        if labels[idx] == key:
                            labels[idx] = change_label[key]
                means_for_labels = np.array([gm.means_[label] for label in labels])
                
                disps_for_means = np.sqrt(np.sum((pos - means_for_labels)**2, axis=1))

                gaps = []
                for label in sorted(np.unique(labels)):
                    check = np.array([idx for idx, lb in enumerate(labels) if lb==label])
                    gaps.append(np.mean(disps_for_means[check]))
                #gap /= len(disps_for_means)
                gap_distrib.append(np.max(gaps))
    gap_distrib = np.array(gap_distrib)
    if len(gap_distrib[gap_distrib < 1.0])/len(gap_distrib) > 0.8:
        seuil = 1.0
    else:
        seuil = 0.0

    for traj in trajectory_list:
        pos = traj.get_positions()
        if len(pos) >= cutoff:
            frames = traj.get_times()
            xs = pos[:, 0]
            ys = pos[:, 1]

            frame_sorted_args = np.argsort(frames)
            xs = xs[frame_sorted_args]
            ys = ys[frame_sorted_args]
            frames = frames[frame_sorted_args]

            pos = np.vstack([xs, ys]).T
            predicted_false_change_pair = []

            if len(xs) >= 5:
                disps = np.sqrt((xs[1:] - xs[:-1])**2 + (ys[1:] - ys[:-1])**2)
                if len(xs) <= 10:
                    n_comp = 5
                elif len(xs) < 100:
                    n_comp = 8
                else:
                    n_comp = 10

                gm = GaussianMixture(n_components=n_comp, n_init=3, max_iter=1000, init_params='k-means++', covariance_type='tied')
                gm = gm.fit(pos)
                labels = gm.predict(pos)
                change_label={}
                pairs = list(itertools.combinations(range(n_comp), r=2))
                for pair in pairs:
                    pair_distance = np.sqrt(np.sum((gm.means_[pair[0]] - gm.means_[pair[1]])**2))
                    if pair_distance < 0.5:
                        change_label[pair[1]] = pair[0]
                for key in sorted(list(change_label.keys()))[::-1]:
                    for idx in range(len(labels)):
                        if labels[idx] == key:
                            labels[idx] = change_label[key]
                means_for_labels = np.array([gm.means_[label] for label in labels])
                
                disps_for_means = np.sqrt(np.sum((pos - means_for_labels)**2, axis=1))

                gap = 0
                gaps = []
                for label in sorted(np.unique(labels)):
                    check = np.array([idx for idx, lb in enumerate(labels) if lb==label])
                    gap += np.sum(disps_for_means[check])
                    gaps.append(np.mean(disps_for_means[check]))
                gap /= len(disps_for_means)

                unique_labels = sorted(np.unique(labels))
                pairs = list(itertools.combinations(unique_labels, r=2))
                change_label_distances = {(pair[0], pair[1]):[] for pair in pairs}
                prev_label = labels[0]

                fig, axs = plt.subplots(1, 4, figsize=(16, 4))
                for idx in range(1, len(pos)):
                    cur_label = labels[idx]
                    if cur_label != prev_label:
                        a, b = min(prev_label, cur_label), max(prev_label, cur_label)
                        change_label_distances[(a, b)].append(np.sqrt(np.sum((pos[idx] - pos[idx-1])**2)))
                    prev_label = cur_label
                    axs[0].plot([pos[idx-1][0], pos[idx][0]], [pos[idx-1][1], pos[idx][1]])
    
                for pair in pairs:
                    changing_distances = change_label_distances[(pair[0], pair[1])]
                    if gap < seuil:
                        predicted_false_change_pair.append(pair)
                
                print('GAP: ', gap, gaps)
                print(labels)
                axs[0].scatter(pos[:,0], pos[:,1])
                axs[0].scatter(gm.means_[:,0], gm.means_[:,1], c='red')
                window_width = max(np.max(pos[:, 0]) - np.min(pos[:, 0]), np.max(pos[:, 0]) - np.min(pos[:, 0])) + 4
                xcenter = np.mean(pos[:, 0])
                ycenter = np.mean(pos[:, 1])
                axs[0].set_xlim([xcenter - window_width//2, xcenter + window_width//2])
                axs[0].set_ylim([ycenter - window_width//2, ycenter + window_width//2])
                axs[1].hist(disps, bins=np.linspace(0, 10, 80))

                clf = LocalOutlierFactor(n_neighbors=5)
                clf.fit_predict(pos)
                print(clf.negative_outlier_factor_)
                print(disps)
                axs[2].hist(clf.negative_outlier_factor_, bins=np.linspace(min(-10, np.min(clf.negative_outlier_factor_)), 0, 20))

                """
                G = nx.Graph()
                for i in range(0, len(pos)):
                    G.add_node(i, position=pos[i])
                for i in range(0, len(pos)):
                    for j in range(i+1, len(pos)):
                        #G.add_edge(i, j, distance=1 / np.sqrt(np.sum((pos[i] - pos[j])**2))**2)
                        G.add_edge(i, j, distance=1)
                    #G.add_edge(idx-1, idx)
                communs = nx.community.greedy_modularity_communities(G, weight='distance', resolution=1, best_n=3, cutoff=3)
                for commun, color in zip(communs, ['red', 'orange', 'yellow', 'green', 'blue', 'cyan', 'purple', 'gray', 'black']):
                    for node in commun:
                        x, y = G.nodes[node]['position']
                        axs[3].scatter(x, y, c=color)
                """

                plt.show()

            if len(predicted_false_change_pair) > 0:
                post_processed_count += 1
                prev_label = labels[0]
                new_traj = TrajectoryObj(index=traj_index)
                new_traj.add_trajectory_position(frames[0], xs[0], ys[0], 0)
                for lb_idx in range(1, len(labels)):
                    cur_label = labels[lb_idx]
                    if (min(prev_label, cur_label), max(prev_label, cur_label)) in predicted_false_change_pair:
                        if len(new_traj.get_positions()) >= cutoff:
                            filtered_traj_list.append(new_traj)
                            traj_index += 1
                        new_traj = TrajectoryObj(index=traj_index)
                    new_traj.add_trajectory_position(frames[lb_idx], xs[lb_idx], ys[lb_idx], 0)
                    prev_label = cur_label
                if len(new_traj.get_positions()) >= cutoff:
                    filtered_traj_list.append(new_traj)
                    traj_index += 1
            else:
                new_traj = TrajectoryObj(index=traj_index)
                for idx in range(len(xs)):
                    new_traj.add_trajectory_position(frames[idx], xs[idx], ys[idx], 0)
                if len(new_traj.get_positions()) >= cutoff:
                    filtered_traj_list.append(new_traj)
                    traj_index += 1


    if verbose != 0:
        print(f'Post processed nb of trajectories: {post_processed_count} with ratio: {len(gap_distrib[gap_distrib < 1.0])/len(gap_distrib)}, seuil: {seuil}')
    return filtered_traj_list


"""
import networkx as nx
from sklearn.neighbors import LocalOutlierFactor
from FreeTrace.module.data_load import read_csv
import matplotlib.pyplot as plt
my_traj_list = []
#df = read_csv('outputs/20250318-RPE-1-FUS-1C9-Wang4-100uM-JF200nM-Hoetchst03-4H-cell24-IR10min_immobile_dense_traces_postoff_0.5.csv')
#df = read_csv('outputs/20231129-cell6-ir_whole_traces_postoff_0.5.csv')
#df = read_csv('outputs/20231129-cell6-whole-noIR_traces_postoff_0.5.csv')
df = read_csv('outputs/alpha_test0_traces_postoff_0.5.csv')
#df = read_csv('outputs/alpha_test1_traces_postoff_0.5.csv')
#df = read_csv('outputs/sample0_traces_postoff_0.5.csv')
#df = read_csv('outputs/20250219-RPE-1-WT-1C9-JF200nM-Hoetchst03-cell3-IR10min_traces_postoff_0.5.csv')
#df = read_csv('outputs/20250219-RPE-1-WT-1C9-JF200nM-Hoetchst03-cell2_traces_postoff_0.5.csv')
traj_indices = df['traj_idx'].unique()
for ix, traj_idx in enumerate(traj_indices):
    single_traj = df[df['traj_idx'] == traj_idx]
    xs = single_traj['x'].to_numpy()
    ys = single_traj['y'].to_numpy()
    frames = single_traj['frame'].to_numpy()
    traj = TrajectoryObj(index=ix)
    for x, y, frame in zip(xs, ys, frames):
        traj.add_trajectory_position(frame, x, y, 0)
    my_traj_list.append(traj)
post_processing2(my_traj_list, 3, 1)
exit()
"""


"""
def post_processing(trajectory_list, cutoff):
    length_check = 3
    std_expectation_cut = 0.075

    filtered_pos = []
    filtered_frames = []
    for traj in trajectory_list:
        start_n_comp = 3
        delete_idx = []

        pos = traj.get_positions()
        frames = traj.get_times()
        xs = pos[:, 0]
        ys = pos[:, 1]
        zs = pos[:, 2]

        frame_sorted_args = np.argsort(frames)
        xs = xs[frame_sorted_args]
        ys = ys[frame_sorted_args]
        zs = zs[frame_sorted_args]
        frames = frames[frame_sorted_args]
        
        if len(xs) >= 5:
            disps = np.sqrt((xs[1:] - xs[:-1])**2 + (ys[1:] - ys[:-1])**2).reshape((len(xs) - 1), 1)
            if np.std(disps) < 1e-6:
                continue
            while True:
                if start_n_comp == 1:
                    break
                flag = 0
                gm = GaussianMixture(start_n_comp, max_iter=100, init_params='k-means++').fit(disps / np.max(disps))
                means_ = gm.means_.flatten()
                stds_ = gm.covariances_.flatten()
                pairs = itertools.combinations(range(start_n_comp), r=2)
                
                for pair in pairs:
                    if abs(means_[pair[0]] - means_[pair[1]]) < 0.5 \
                        or (means_[pair[1]] - 4*stds_[pair[1]] < means_[pair[0]] + 4*stds_[pair[0]] < means_[pair[1]] + 4*stds_[pair[1]]) \
                        and (means_[pair[0]] - 4*stds_[pair[0]] < means_[pair[1]] + 4*stds_[pair[1]] < means_[pair[0]] + 4*stds_[pair[0]]):
                        start_n_comp -= 1
                        flag = 1
                        break
                if flag == 0:
                    break
            
            if start_n_comp >= 2:         
                labels = gm.predict(disps / np.max(disps))
                expects = [means_[label] for label in labels]
                std_for_class = ((disps / np.max(disps)).flatten() - expects)**2
                std_expect = np.sqrt(np.mean(std_for_class))
                if std_expect < std_expectation_cut:
                    label_count = [0 for _ in np.unique(labels)]
                    i = 1
                    before_label = labels[0]
                    cuts = []

                    while True:
                        cur_label = labels[i]
                        if before_label != cur_label:
                            cuts.append(i)
                        i+=1
                        if i == len(labels):
                            break

                    before_label = labels[0]
                    chunk_idx = [0, len(labels)]
                    for lb_idx, label in enumerate(labels):
                        if label != before_label:
                            chunk_idx.append(lb_idx)
                        before_label = label
                    chunk_idx = sorted(chunk_idx)
                    for label in labels:
                        label_count[label] += 1
                    max_label = np.argmax(label_count)

                    for idx in range(len(chunk_idx) - 1):
                        if (chunk_idx[idx+1] - chunk_idx[idx]) <= length_check and labels[chunk_idx[idx]] != max_label:
                            delete_idx.extend(list(range(chunk_idx[idx], chunk_idx[idx+1])))
                print(delete_idx, disps, std_expect)
        if len(delete_idx) > 0:
            new_xs = []
            new_ys = []
            new_zs = []
            new_frames = []

            delete_idx = np.array(delete_idx)
            for i in range(len(xs)):
                if i in delete_idx:
                    new_xs.append(xs[i])
                    new_ys.append(ys[i])
                    new_zs.append(zs[i])
                    new_frames.append(frames[i])
                    if len(new_xs) >= 2:
                        filtered_pos.append([new_xs, new_ys, new_zs])
                        filtered_frames.append(new_frames)
                    new_xs = []
                    new_ys = []
                    new_zs = []
                    new_frames = []
                else:
                    new_xs.append(xs[i])
                    new_ys.append(ys[i])
                    new_zs.append(zs[i])
                    new_frames.append(frames[i])
        else:
            filtered_pos.append([xs, ys, zs])
            filtered_frames.append(frames)
            
    filtered_trajectory_list = []
    traj_idx = 0
    for path, frames in zip(filtered_pos, filtered_frames):
        if len(path) >= cutoff:
            traj = TrajectoryObj(index=traj_idx)
            for node_idx in range(len(frames)):
                x = path[0][node_idx]
                y = path[1][node_idx]
                z = path[2][node_idx]
                frame = frames[node_idx]
                traj.add_trajectory_position(frame, x, y, z)
            filtered_trajectory_list.append(traj)
            traj_idx += 1
    return filtered_trajectory_list
"""
