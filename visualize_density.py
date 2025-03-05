import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import tifffile
import imageio
import itertools
import functools
import gc
from scipy.spatial import distance


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


def make_loc_depth_video(output_dir, coords, multiplier=16, frame_cumul=100, winsize=7, resolution=2, dim=2, start_frame=1, end_frame=5000):
    resolution = int(max(1, min(3, resolution)))  # resolution in [1, 2, 3]
    amp = 0
    multiplier = multiplier - 1 if multiplier % 2 == 1 else multiplier
    winsize += multiplier * resolution
    cov_std = multiplier * resolution
    amp_ = 10**amp
    margin_pixel = 2
    margin_pixel *= 10*amp_
    amp_*= resolution

    time_steps = np.array(list(coords.keys()))
    all_coords = []
    stacked_imgs = []
    stacked_coords = {t:[] for t in time_steps}
    for t in time_steps:
        st_tmp = []
        for coord in coords[t]:
            if len(coord) == 3:
                all_coords.append(coord)
        for stack_t in range(t, t+frame_cumul):
            if stack_t in time_steps:
                for stack_coord in coords[stack_t]:
                    if len(stack_coord) == 3:
                        st_tmp.append(stack_coord)
        stacked_coords[t]=np.array(st_tmp, dtype=np.float32)
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

    if dim == 2:
        mycmap = plt.get_cmap('jet', lut=None)
        color_seq = [mycmap(i)[:3] for i in range(mycmap.N)]
        template = np.ones((1, (winsize)**2, 2), dtype=np.float16) * quantification(winsize)
        template = (np.exp((-1./2) * np.sum(template @ np.linalg.inv([[cov_std, 0], [0, cov_std]]) * template, axis=2))).reshape([winsize, winsize])

        for time in time_steps:
            if start_frame <= time <= end_frame:
                image = np.zeros((int((y_max - y_min)*amp_ + margin_pixel), int((x_max - x_min)*amp_ + margin_pixel)), dtype=np.float16)
                selected_coords = stacked_coords[time]
                if len(selected_coords) > 0:
                    selected_coords[:, 1] -= x_min
                    selected_coords[:, 0] -= y_min
                    selected_coords = np.round(selected_coords * amp_)

                    for roundup_coord in selected_coords:
                        coord_col = int(roundup_coord[0] + margin_pixel//2)
                        coord_row = int(roundup_coord[1] + margin_pixel//2)
                        row = min(max(0, coord_row), image.shape[0])
                        col = min(max(0, coord_col), image.shape[1])
                        image[row - winsize//2: row + winsize//2 + 1, col - winsize//2: col + winsize//2 + 1] += template
                    
                    img_min, img_max = np.quantile(image, [0.01, 0.995])
                    image = np.minimum(image, np.ones_like(image) * img_max)
                    image = image / np.max(image)
                    mapped_data = mycmap(image)
                    stacked_imgs.append(np.array(mapped_data * 255, dtype=np.uint8))
                    del image
                    del mapped_data

        stacked_imgs = np.array(stacked_imgs)
        stacked_imgs = stacked_imgs.astype(np.uint8)
        #with imageio.get_writer(f'{output_dir}_loc_{dim}d_density_video.gif', mode='I', fps=5, loop=1) as writer:
        #    for i in range(len(stacked_imgs)):
        #        writer.append_data(np.array(stacked_imgs[i]))
        tifffile.imwrite(f'{output_dir}_loc_{dim}d_density_video.tiff', data=stacked_imgs, imagej=True)


def make_loc_radius_video(output_dir, coords, frame_cumul=100, radius=10, start_frame=1, end_frame=5000):
    resolution = 2  # resolution in [1, 2, 3]
    dim=2
    winsize=7
    multiplier = 4
    winsize += multiplier * resolution
    cov_std = 1
    margin_pixel = 50
    amp_= 2

    time_steps = np.array(sorted(list(coords.keys())))
    all_coords = []
    stacked_imgs = []
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
            paired_cdist = distance.cdist(stacked_coords[t], stacked_coords[t], 'euclidean')
            stacked_radii[t] = (paired_cdist <= radius).sum(axis=1)
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
    img_max = 0
    if dim == 2:
        template = np.ones((1, (winsize)**2, 2), dtype=np.float16) * quantification(winsize)
        template = (np.exp((-1./2) * np.sum(template @ np.linalg.inv([[cov_std, 0], [0, cov_std]]) * template, axis=2))).reshape([winsize, winsize])
        template = template / np.max(template)

        for time in time_steps:
            if start_frame <= time <= end_frame:
                if time%100 == 0: print(f'Generating the image of frame:{time}') 
                image = np.zeros((int((y_max - y_min)*amp_ + margin_pixel), int((x_max - x_min)*amp_ + margin_pixel)), dtype=np.float16)
                selected_coords = stacked_coords[time]
                selected_radii = stacked_radii[time]
                if len(selected_coords) > 0:
                    selected_coords[:, 0] -= x_min
                    selected_coords[:, 1] -= y_min    
                    selected_coords = np.round(selected_coords * amp_).astype(int)
                    selected_coords += margin_pixel//2

                    for coord_index, (roundup_coord, selec_rad) in enumerate(zip(selected_coords, selected_radii)):
                        col = roundup_coord[0]
                        row = roundup_coord[1]
                        image[row - winsize//2: row + winsize//2 + 1, col - winsize//2: col + winsize//2 + 1]\
                            += (template * (selec_rad / count_max))
                    
                    image = image[margin_pixel//2:image.shape[0]-margin_pixel//2, margin_pixel//2:image.shape[1]-margin_pixel//2]
                    img_max = max(np.max(image), img_max)
                    stacked_imgs.append(image)
        stacked_imgs = np.array(stacked_imgs, dtype=np.float16)

        stacked_imgs = np.log(1 + stacked_imgs)
        img_max = np.log(1 + img_max)

        mapped_imgs = []
        for i in range(len(stacked_imgs)):
            if i%100 == 0: print(f'Mapping the image of frame:{i}') 
            cmap_img = (mycmap(stacked_imgs[i] / img_max)[:,:,:3]).astype(np.float16)
            mapped_imgs.append(cmap_img)
        del stacked_imgs
        gc.collect()
        mapped_imgs = ((np.array(mapped_imgs, dtype=np.float16)) * 255).astype(np.uint8)

        #with imageio.get_writer(f'{output_dir}_loc_{dim}d_density_video.gif', mode='I', fps=5, loop=1) as writer:
        #    for i in range(len(stacked_imgs)):
        #        writer.append_data(np.array(stacked_imgs[i]))
        tifffile.imwrite(f'{output_dir}_loc_{dim}d_density_video.tiff', data=mapped_imgs, imagej=True)


if __name__ == '__main__':
    if len(sys.argv) < 2:
           sys.exit("Need loc.csv file to visualize density. Example) python3 visualize_density.py sample_loc.csv")

    loc_files = []
    for idx in range(len(sys.argv) - 1):
        loc_files.append(sys.argv[1].strip())
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
    #make_loc_depth_video(loc_file, all_loc, multiplier=4, frame_cumul=100, winsize=7, resolution=1, start_frame=1, end_frame=10000)
    make_loc_radius_video(loc_file, all_loc, frame_cumul=1000, radius=10, start_frame=5000, end_frame=20000)
