import cupyx.scipy.spatial.distance
import numpy as np
import sys
import matplotlib.pyplot as plt
import tifffile
import itertools
import gc
from scipy.spatial import distance
from scipy import stats
from FreeTrace.module.ImageModule import read_tif
import os
import cv2


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


def quantification(window_size):
    x = np.arange(-(window_size-1)/2, (window_size+1)/2)
    y = np.arange(-(window_size-1)/2, (window_size+1)/2)
    xv, yv = np.meshgrid(x, y, sparse=True)
    grid = np.stack(np.meshgrid(xv, yv), -1).reshape(window_size * window_size, 2)
    return grid.astype(np.float32)


def make_loc_depth_image(output_dir, coords, multiplier=16, winsize=7, resolution=2, dim=2):
    resolution = int(max(1, min(3, resolution)))  # resolution in [1, 2, 3]
    amp = 1
    multiplier = multiplier - 1 if multiplier % 2 == 1 else multiplier
    winsize += multiplier * resolution
    cov_std = multiplier * resolution
    amp_ = 10**amp
    margin_pixel = 2
    margin_pixel *= 10*amp_
    amp_*= resolution

    time_steps = np.array(list(coords.keys()))
    all_coords = []
    for t in time_steps:
        for coord in coords[t]:
            if len(coord) == 3:
                all_coords.append(coord)
    all_coords = np.array(all_coords)
    if len(all_coords) == 0:
        return

    x_min = np.min(all_coords[:, 0])
    x_max = np.max(all_coords[:, 0])
    y_min = np.min(all_coords[:, 1])
    y_max = np.max(all_coords[:, 1])
    z_min = np.min(all_coords[:, 2])
    z_max = np.max(all_coords[:, 2])
    z_min, z_max = np.quantile(all_coords[:, 2], [0.01, 0.99])
    all_coords[:, 1] -= x_min
    all_coords[:, 0] -= y_min

    if dim == 2:
        mycmap = plt.get_cmap('jet', lut=None)
        color_seq = [mycmap(i)[:3] for i in range(mycmap.N)]
        image = np.zeros((int((y_max - y_min)*amp_ + margin_pixel), int((x_max - x_min)*amp_ + margin_pixel)), dtype=np.float32)
        all_coords = np.round(all_coords * amp_)
        template = np.ones((1, (winsize)**2, 2), dtype=np.float32) * quantification(winsize)
        template = (np.exp((-1./2) * np.sum(template @ np.linalg.inv([[cov_std, 0], [0, cov_std]]) * template, axis=2))).reshape([winsize, winsize])

        for roundup_coord in all_coords:
            coord_col = int(roundup_coord[0] + margin_pixel//2)
            coord_row = int(roundup_coord[1] + margin_pixel//2)
            row = min(max(0, coord_row), image.shape[0])
            col = min(max(0, coord_col), image.shape[1])
            image[row - winsize//2: row + winsize//2 + 1, col - winsize//2: col + winsize//2 + 1] += template
        
        img_min, img_max = np.quantile(image, [0.01, 0.995])
        image = np.minimum(image, np.ones_like(image) * img_max)
        image = image / np.max(image)
        plt.figure('Localization density', dpi=512)
        plt.imshow(image, cmap=mycmap, origin='upper')
        plt.axis('off')
        plt.savefig(f'{output_dir}_loc_{dim}d_density.png', bbox_inches='tight')
        plt.close('all')
    else:
        z_coords = np.maximum((all_coords[:, 2] - z_min), np.zeros_like(all_coords[:, 2]))
        z_coords = np.minimum(z_coords, np.ones_like(z_coords) * (z_max - z_min))
        z_coords = z_coords / (z_max - z_min)
        mycmap = plt.get_cmap('jet', lut=None)
        color_seq = [mycmap(i)[:3] for i in range(mycmap.N)]
        
        image = np.zeros((int((y_max - y_min)*amp_ + margin_pixel), int((x_max - x_min)*amp_ + margin_pixel), 3), dtype=np.float32)
        all_coords = np.round(all_coords * amp_)
        template = np.ones((1, (winsize)**2, 2), dtype=np.float32) * quantification(winsize)
        template = (np.exp((-1./2) * np.sum(template @ np.linalg.inv([[cov_std, 0], [0, cov_std]]) * template, axis=2))).reshape([winsize, winsize])

        for idx, (roundup_coord, z_coord) in enumerate(zip(all_coords, z_coords)):
            coord_col = int(roundup_coord[0] + margin_pixel//2)
            coord_row = int(roundup_coord[1] + margin_pixel//2)
            color_z = color_seq[min(int(np.round(len(color_seq) * z_coord)), len(color_seq)-1)]
            row = min(max(0, coord_row), image.shape[0])
            col = min(max(0, coord_col), image.shape[1])
            image[row - winsize//2: row + winsize//2 + 1, col - winsize//2: col + winsize//2 + 1, 0] += template * color_z[0]
            image[row - winsize//2: row + winsize//2 + 1, col - winsize//2: col + winsize//2 + 1, 1] += template * color_z[1]
            image[row - winsize//2: row + winsize//2 + 1, col - winsize//2: col + winsize//2 + 1, 2] += template * color_z[2]
        
        img_min, img_max = np.quantile(image, [0.01, 0.995])
        image = np.minimum(image, np.ones_like(image) * img_max)
        image = image / np.max(image)
        plt.figure('Localization density', dpi=512)
        plt.imshow(image, origin='upper')
        plt.axis('off')
        plt.savefig(f'{output_dir}_loc_{dim}d_densitymap.png', bbox_inches='tight')
        plt.close('all')


def make_loc_radius_video2(output_dir, coords, frame_cumul=100, radius=[1, 10], start_frame=1, end_frame=10000, gauss=True, gpu=False):
    if gpu:
        import cupy as cp
        from cuvs.distance import pairwise_distance
        mempool = cp.get_default_memory_pool()
        mempool.set_limit(fraction=0.8)

    dim = 2
    winsize = 31  # odd number.
    cov_std = 2.5
    margin_pixel = 50
    amp_= 2

    time_steps = np.array(sorted(list(coords.keys())))
    end_frame = min(end_frame, time_steps[-1])
    start_frame = max(start_frame, time_steps[0])

    all_coords = []
    stacked_coords = {t:[] for t in time_steps if start_frame <= t <= end_frame}
    stacked_radii = {t:[] for t in time_steps if start_frame <= t <= end_frame}
    count_max = 0

    for t in time_steps:
        if start_frame <= t <= end_frame:
            if t%10 == 0: print(f'Calcul radius on cumulated molecules at frame:{t}')     
            for coord in coords[t]:
                if len(coord) == 3:
                    all_coords.append(coord)
            if t == start_frame:
                st_tmp = []
                for stack_t in range(t, t+frame_cumul):
                    time_st = []
                    if stack_t in time_steps:
                        for stack_coord in coords[stack_t]:
                            if len(stack_coord) == 3:
                                time_st.append(stack_coord)
                    st_tmp.append(time_st)
                prev_tmps=st_tmp
            else:
                stack_t = t+frame_cumul-1
                time_st = []
                if stack_t in time_steps:
                    for stack_coord in coords[stack_t]:
                        if len(stack_coord) == 3:
                            time_st.append(stack_coord)
                st_tmp = prev_tmps[1:]
                st_tmp.append(time_st)
                prev_tmps = st_tmp
            st_tmp = list(itertools.chain.from_iterable(st_tmp))
            stacked_coords[t]=np.array(st_tmp, dtype=np.float32)

            if gpu:
                cp_dist = cp.asarray(stacked_coords[t], dtype=cp.float16)
                paired_cp_dist = pairwise_distance(cp_dist, cp_dist, metric='euclidean')
                paired_cdist = cp.asnumpy(paired_cp_dist).astype(np.float16)
            else:
                paired_cdist = distance.cdist(stacked_coords[t], stacked_coords[t], 'euclidean')

            stacked_radii[t] = ((paired_cdist > radius[0])*(paired_cdist <= radius[1])).sum(axis=1)
            cur_max_count = np.max(stacked_radii[t])
            count_max = max(cur_max_count, count_max)
    all_coords = np.array(all_coords)

    if len(all_coords) == 0:
        return

    x_min = np.min(all_coords[:, 0])
    x_max = np.max(all_coords[:, 0])
    y_min = np.min(all_coords[:, 1])
    y_max = np.max(all_coords[:, 1])
    mycmap = plt.get_cmap('jet', lut=None)
    #color_seq = (np.array([mycmap(i)[:3] for i in range(mycmap.N)]) * 255).astype(int)
    stack_idx = 0
    stacked_imgs = np.zeros((end_frame - start_frame + 1, int((y_max - y_min)*amp_ + margin_pixel), int((x_max - x_min)*amp_ + margin_pixel)), dtype=np.float16)
    
    if dim == 2:
        if gauss:
            template = np.ones((1, (winsize)**2, 2), dtype=np.float16) * quantification(winsize)
            template = (np.exp((-1./2) * np.sum(template @ np.linalg.inv([[cov_std, 0], [0, cov_std]]) * template, axis=2))).reshape([winsize, winsize])
            template = template / np.max(template)

        for time in time_steps:
            if start_frame <= time <= end_frame:
                if time%100 == 0: print(f'Generating the image of frame:{time}') 
                selected_coords = stacked_coords[time]
                selected_radii = stacked_radii[time]
                max_rad_for_t = 0
                if len(selected_coords) > 0:
                    selected_coords[:, 0] -= x_min
                    selected_coords[:, 1] -= y_min    
                    selected_coords = np.round(selected_coords * amp_).astype(int)
                    selected_coords += margin_pixel//2

                    for coord_index, (roundup_coord, selec_rad) in enumerate(zip(selected_coords, selected_radii)):
                        max_rad_for_t = max(max_rad_for_t, selec_rad)
                        col = roundup_coord[0]
                        row = roundup_coord[1]
                        if gauss:
                            stacked_imgs[stack_idx, row - winsize//2: row + winsize//2 + 1, col - winsize//2: col + winsize//2 + 1]\
                                = np.maximum((template * selec_rad), stacked_imgs[stack_idx, row - winsize//2: row + winsize//2 + 1, col - winsize//2: col + winsize//2 + 1])
                        else:
                            stacked_imgs[stack_idx, row, col] = selec_rad
                    
                    if gauss:
                        stacked_imgs[stack_idx] = np.minimum(stacked_imgs[stack_idx], np.ones_like(stacked_imgs[stack_idx]) * max_rad_for_t)
                stack_idx +=1
        stacked_imgs = stacked_imgs[:, margin_pixel//2:stacked_imgs.shape[1]-margin_pixel//2, margin_pixel//2:stacked_imgs.shape[2]-margin_pixel//2]
        stacked_imgs = np.log(1 + stacked_imgs)
        stacked_imgs = stacked_imgs / np.max(stacked_imgs)

        mapped_imgs = np.empty([stacked_imgs.shape[0], stacked_imgs.shape[1], stacked_imgs.shape[2], 3], dtype=np.float16)
        for i in range(len(stacked_imgs)):
            if i%100 == 0: print(f'Mapping the image of frame:{i}') 
            mapped_imgs[i] = (mycmap(stacked_imgs[i])[:,:,:3]).astype(np.float16)
        del stacked_imgs
        gc.collect()
        mapped_imgs = (mapped_imgs * 255).astype(np.uint8)

        #with imageio.get_writer(f'{output_dir}_loc_{dim}d_density_video.gif', mode='I', fps=5, loop=1) as writer:
        #    for i in range(len(stacked_imgs)):
        #        writer.append_data(np.array(stacked_imgs[i]))
        tifffile.imwrite(f'{output_dir}_loc_{dim}d_density_video.tiff', data=mapped_imgs, imagej=True)


def make_loc_radius_video(video_dir, output_dir, coords, frame_cumul=100, radius=[1, 10], start_frame=1, end_frame=10000, gpu=False):
    sequence_save_folder = './image_sequences'
    filename = output_dir[0].split('/')[-1].split('.csv')[0]
    density_save_folder = f'{sequence_save_folder}/{filename}'
    if not os.path.exists(sequence_save_folder):
        os.mkdir(sequence_save_folder)
    if not os.path.exists(density_save_folder):
        os.mkdir(density_save_folder)

    images = read_tif(video_dir)[start_frame-1:end_frame,:,:]
    if gpu:
        import cupy as cp
        from cuvs.distance import pairwise_distance
        mempool = cp.get_default_memory_pool()
        mempool.set_limit(fraction=0.8)

    time_steps = np.array(sorted(list(coords.keys())))
    end_frame = min(end_frame, time_steps[-1])
    start_frame = max(start_frame, time_steps[0])

    all_coords = []
    stacked_coords = {t:[] for t in time_steps if start_frame <= t <= end_frame}
    stacked_radii = {t:[] for t in time_steps if start_frame <= t <= end_frame}
    count_max = 0

    for t in time_steps:
        if start_frame <= t <= end_frame:
            if t%10 == 0: print(f'Calcul radius on cumulated molecules at frame:{t}')     
            for coord in coords[t]:
                if len(coord) == 3:
                    all_coords.append(coord)
            if t == start_frame:
                st_tmp = []
                for stack_t in range(t, t+frame_cumul):
                    time_st = []
                    if stack_t in time_steps:
                        for stack_coord in coords[stack_t]:
                            if len(stack_coord) == 3:
                                time_st.append(stack_coord)
                    st_tmp.append(time_st)
                prev_tmps=st_tmp
            else:
                stack_t = t+frame_cumul-1
                time_st = []
                if stack_t in time_steps:
                    for stack_coord in coords[stack_t]:
                        if len(stack_coord) == 3:
                            time_st.append(stack_coord)
                st_tmp = prev_tmps[1:]
                st_tmp.append(time_st)
                prev_tmps = st_tmp
            st_tmp = list(itertools.chain.from_iterable(st_tmp))
            stacked_coords[t]=np.array(st_tmp, dtype=np.float32)

            if gpu:
                cp_dist = cp.asarray(stacked_coords[t], dtype=cp.float16)
                paired_cp_dist = pairwise_distance(cp_dist, cp_dist, metric='euclidean')
                paired_cdist = cp.asnumpy(paired_cp_dist).astype(np.float16)
            else:
                paired_cdist = distance.cdist(stacked_coords[t], stacked_coords[t], 'euclidean')

            stacked_radii[t] = ((paired_cdist > radius[0])*(paired_cdist <= radius[1])).sum(axis=1)
            cur_max_count = np.max(stacked_radii[t])
            count_max = max(cur_max_count, count_max)
    all_coords = np.array(all_coords)
    if len(all_coords) == 0:
        return

    image_idx = 0
    x_min = np.min(all_coords[:, 0])
    x_max = np.max(all_coords[:, 0])
    y_min = np.min(all_coords[:, 1])
    y_max = np.max(all_coords[:, 1])
    mycmap = plt.get_cmap('jet', lut=None)
    video_arr = np.empty([images.shape[0], images.shape[1], images.shape[2], 4], dtype=np.uint8)
    X, Y = np.mgrid[x_min:x_max:complex(f'{images.shape[2]}j'), y_min:y_max:complex(f'{images.shape[1]}j')]
    positions = np.vstack([X.ravel(), Y.ravel()])
    for time in time_steps:
        if start_frame <= time <= end_frame:
            if time%50 == 0: print(f'Rendering the image of frame:{time}') 
            selected_coords = stacked_coords[time]
            selected_radii = stacked_radii[time]
            if len(selected_coords) > 0:
                values = np.vstack([selected_coords[:, 0], selected_coords[:, 1]])
                kernel = stats.gaussian_kde(values, weights=(selected_radii / count_max))
                kernel.set_bandwidth(bw_method=kernel.factor / 2.)
                Z = np.reshape(kernel(positions).T, X.shape)
                Z = Z * (np.max(selected_radii) / np.max(Z))
                aximg = plt.imshow(Z.T, cmap=mycmap,
                                   extent=[x_min, x_max, y_min, y_max], vmin=0, vmax=count_max, alpha=0.7, origin='upper')
                rawimg = plt.imshow(images[image_idx: image_idx + frame_cumul,:,:].max(axis=(0)), alpha=.6, cmap='grey', extent=[x_min, x_max, y_min, y_max], origin='upper')
                arr = aximg.make_image(renderer=None, unsampled=True)[0][:,:,:4]
                arr2 = rawimg.make_image(renderer=None, unsampled=True)[0][:,:,:4]
                blended = cv2.addWeighted(arr, 0.4, arr2, 0.6, 0.0)
                video_arr[image_idx] = blended
            image_idx += 1
    tifffile.imwrite(f'{density_save_folder}/density_video.tiff', data=video_arr, imagej=True)


if __name__ == '__main__':
    # https://docs.rapids.ai/install/?_gl=1*1lwgnt1*_ga*OTY3MzQ5Mzk0LjE3NDEyMDc2NDA.*_ga_RKXFW6CM42*MTc0MTIwNzY0MC4xLjEuMTc0MTIwNzgxMS4yOS4wLjA.
    if len(sys.argv) < 2:
           sys.exit("Need loc.csv file to visualize density. Example) python3 visualize_density.py sample_loc.csv")
    video_path = sys.argv[1]
    loc_files = []
    for idx in range(2, len(sys.argv)):
        loc_files.append(sys.argv[idx].strip())


    all_loc = {}
    for loc_idx, loc_file in enumerate(loc_files):
        loc, loc_infos = read_localization(loc_file, None)
        if loc_idx == 0:
            for time in list(loc.keys()):
                all_loc[time] = list(loc[time]).copy()
        else:
            for time in list(loc.keys()):
                all_loc[time].extend(list(loc[time]).copy())
        del loc
        del loc_infos
    for t_tmp in list(all_loc.keys()):
        all_loc[t_tmp] = np.array(all_loc[t_tmp])

    #make_loc_depth_image(loc_file, all_loc, multiplier=4, winsize=7, resolution=2, dim=3)
    make_loc_radius_video(video_path, loc_files, all_loc, frame_cumul=1000, radius=[3, 20], start_frame=15000, end_frame=20000, gpu=True)
