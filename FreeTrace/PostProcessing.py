import numpy as np
import itertools
from sklearn.mixture import GaussianMixture
from FreeTrace.module.trajectory_object import TrajectoryObj


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
