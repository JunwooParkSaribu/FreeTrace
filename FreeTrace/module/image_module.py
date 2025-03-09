import sys
import time
import cv2
import os
import pandas as pd
import numpy as np
import imageio
import tifffile
import tkinter as tk
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from scipy import stats
from scipy.spatial import distance
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from multiprocessing import Queue, Process, Value
from FreeTrace.module.trajectory_object import TrajectoryObj
from FreeTrace.module.data_load import read_trajectory, read_localization, read_multiple_locs


class NormalTkExit(Exception):
    def __init__(self, message="Normal tk widget exit to release resources"):
        self.message = message
        super().__init__(self.message)


class RealTimePlot(tk.Tk):
    def __init__(self, title='', job_type='loc', show_frame=False, show_option=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wm_title(string=title)
        self.queue = Queue()
        self.force_terminate = Value('b', 0)
        self.terminated_time_over = Value('b', 0)
        self.job_type = job_type
        self.past_t_steps = []
        self.text_kwargs = dict(fontsize=20)
        self.props = dict(boxstyle='round', facecolor='grey', alpha=0.15)  # bbox features
        self.cmap_plt = 'gist_gray'
        self.show_frame = show_frame
        self.show_option = show_option
        self.video_wait_max_time = 30 if job_type=='loc' else 120
        self.fps = 1
        self.max_queue_recv_size = 500
        self.q_idx = 0
        self.img_process = Process(target=self.start_main_loop, daemon=True)

    def clean_tk_widgets(self):
        self.terminated_time_over.value = 1
        self.destroy()
        del self.ax
        del self.canvas
        del self.figure
        self.force_terminate.value = 0
        sys.exit(0)
          
    def update_plot(self):
        self.q_idx += 1
        if self.force_terminate.value == 1:
            self.clean_tk_widgets()
        try:
            self.ax.clear()
            self.ax.margins(x=0, y=0)
            if self.job_type == 'loc':
                img, coords, frame = self.queue.get(timeout=self.video_wait_max_time)
                if frame == -1:
                    raise NormalTkExit
                self.ax.imshow(img, cmap=self.cmap_plt)
                if len(coords) > 0:
                    self.ax.scatter(coords[:, 1], coords[:, 0], marker='+', c='red', alpha=0.6)
            else:
                img, trajs, frame = self.queue.get(timeout=self.video_wait_max_time)
                if frame == -1:
                    raise NormalTkExit
                self.ax.imshow(img, cmap=self.cmap_plt)
                if len(trajs) > 0:
                    for traj in trajs:
                        if len(traj) > 1:
                            self.ax.plot(traj[:, 0], traj[:, 1], c='red', alpha=0.6)

            if self.show_frame:
                self.ax.set_title(f'Frame: {frame}', fontdict=self.text_kwargs)
            if self.show_option:
                self.ax.text(0, img.shape[0] + int(img.shape[0]/8), f'Max wait time: {self.video_wait_max_time}\n', fontsize=12, bbox=self.props)
            self.figure.canvas.draw()

            if self.q_idx % 2 == 0:
                cur_qsize = self.queue.qsize()
                if cur_qsize > self.max_queue_recv_size / 2:
                    self.fps = 1
                else:
                    self.fps = int((self.max_queue_recv_size * 30) / (self.queue.qsize()+1)**2) + 1
        except NormalTkExit as tke:
            print(f'Real time visualization turns off - [{tke}]')
            self.clean_tk_widgets()
        except Exception as e:
            print(f'')
            print(f'FreeTrace turns off the real-time viusualization if it waits more than [{self.video_wait_max_time}s] / [{e}], to reduce the usage of computational resource.')
            print(f'FreeTrace is running if you have still non-inferred frames. Please don\'t shut down, it is just slowed down due to high number of particles / resolution.')
            print(f'')
            self.clean_tk_widgets()

        self.after(self.fps, self.update_plot)

    def turn_on(self):
        self.img_process.start()

    def cleaning_queue(self):
        try:
            while self.queue.qsize() > 0:
                self.queue.get()
        except:
            return 1

    def turn_off(self):
        self.force_terminate.value = 1
        self.queue.put((np.zeros([2, 2]), [], -1))
        time.sleep(1.0)
        if self.img_process.is_alive():
            self.img_process.terminate()
        self.cleaning_queue()
        self.queue.close()
        self.queue.cancel_join_thread()
        del self.queue
        del self.force_terminate
        del self.img_process
 
    def start_main_loop(self):
        self.queue.get()
        self.figure = plt.figure(figsize=(12, 12))
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(padx=0, pady=0)
        self.update_plot()
        self.mainloop()

    def put_into_queue(self, data_zip, mod_n=1):
        if self.terminated_time_over.value == 1:
            return
        if self.queue.qsize() > self.max_queue_recv_size:
            return
        if self.job_type == 'loc':
            imgs = data_zip[0]
            coords_in_t = data_zip[1]
            t = data_zip[2]
            for data_idx in range(len(imgs)):
                if data_idx % mod_n == 0:
                    self.queue.put((imgs[data_idx], np.array(coords_in_t[data_idx]), t+data_idx+1))
        else:
            imgs = data_zip[0]
            paths = data_zip[1]
            time_steps = data_zip[2]
            loc = data_zip[3]
            for t in time_steps:
                if t not in self.past_t_steps:
                    tmp_coords = []
                    for path in paths:
                        tmp = []
                        ast = np.array([x[0] for x in path])
                        if t in ast:
                            for node in path[1:]:
                                if t-10 < node[0] <= t:
                                    node_xyz = loc[node[0]][node[1]][:2]
                                    tmp.append(node_xyz)
                        tmp_coords.append(np.array(tmp))
                    if t % mod_n == 0:
                        self.queue.put((imgs[t-1], tmp_coords, t))
                self.past_t_steps.append(t)


def read_tif(filepath, andi2=False):
    normalized_imgs = []
    if andi2:
        imgs = []
        with Image.open(filepath) as img:
            try:
                for i in range(9999999):
                    if i == 0:
                        indice_image = np.array(img.copy())
                    else:
                        imgs.append(np.array(img))
                        img.seek(img.tell() + 1)
            except Exception as e:
                pass
        imgs = np.array(imgs)
    else:
        with tifffile.TiffFile(filepath) as tif:
            imgs = tif.asarray()
            axes = tif.series[0].axes
            imagej_metadata = tif.imagej_metadata

    if len(imgs.shape) == 3:
        nb_tif = imgs.shape[0]
        y_size = imgs.shape[1]
        x_size = imgs.shape[2]

        s_min = np.min(np.min(imgs, axis=(1, 2)))
        s_max = np.max(np.max(imgs, axis=(1, 2)))
    elif len(imgs.shape) == 2:
        nb_tif = 1
        y_size = imgs.shape[0]
        x_size = imgs.shape[1]
        s_min = np.min(np.min(imgs, axis=(0, 1)))
        s_max = np.max(np.max(imgs, axis=(0, 1)))
    else:
        raise Exception 

    for i, img in enumerate(imgs):
        img = (img - s_min) / (s_max - s_min)
        normalized_imgs.append(img)

    normalized_imgs = np.array(normalized_imgs, dtype=np.float32).reshape(-1, y_size, x_size)
    normalized_imgs /= np.max(normalized_imgs, axis=(1, 2)).reshape(-1, 1, 1)  # normalize local
    
    if andi2:
        return normalized_imgs, indice_image
    else:
        return normalized_imgs
    

def read_tif_unnormalized(filepath):
    imgs = []
    with tifffile.TiffFile(filepath) as tif:
        imgs = (tif.asarray()).astype(np.float32)
        axes = tif.series[0].axes
        imagej_metadata = tif.imagej_metadata
    return imgs


def read_single_tif(filepath, ch3=True):
    with tifffile.TiffFile(filepath) as tif:
        imgs = tif.asarray()
        if len(imgs.shape) >= 3:
            imgs = imgs[0]
        axes = tif.series[0].axes
        imagej_metadata = tif.imagej_metadata
        tag = tif.pages[0].tags

    y_size = imgs.shape[0]
    x_size = imgs.shape[1]
    s_mins = np.min(imgs)
    s_maxima = np.max(imgs)
    signal_maxima_avg = np.mean(s_maxima)
    zero_base = np.zeros((y_size, x_size), dtype=np.uint8)
    one_base = np.ones((y_size, x_size), dtype=np.uint8)
    #img = img - mode
    #img = np.maximum(img, zero_base)
    imgs = (imgs - s_mins) / (s_maxima - s_mins)
    #img = np.minimum(img, one_base)
    normalized_imgs = np.array(imgs * 255, dtype=np.uint8)
    if ch3 is False:
        return normalized_imgs
    img_3chs = np.array([np.zeros(normalized_imgs.shape), normalized_imgs, np.zeros(normalized_imgs.shape)]).astype(np.uint8)
    img_3chs = np.moveaxis(img_3chs, 0, 2)
    return img_3chs


def stack_tif(filename, normalized_imgs):
    tifffile.imwrite(filename, normalized_imgs)


def scatter_optimality(trajectory_list):
    plt.figure()
    scatter_x = []
    scatter_y = []
    scatter_color = []
    for traj in trajectory_list:
        if traj.get_optimality() is not None:
            scatter_x.append(traj.get_index())
            scatter_y.append(traj.get_optimality())
            scatter_color.append(traj.get_color())
    plt.scatter(scatter_x, scatter_y, c=scatter_color, s=5, alpha=0.7)
    plt.savefig('entropy_scatter.png')


def make_image(output, trajectory_list, cutoff=0, pixel_shape=(512, 512), amp=1, add_index=True, add_time=True):
    img = np.zeros((pixel_shape[0] * (10**amp), pixel_shape[1] * (10**amp), 3), dtype=np.uint8)
    for traj in trajectory_list:
        if traj.get_trajectory_length() >= cutoff:
            xx = np.array([[int(x * (10**amp)), int(y * (10**amp))]
                           for x, y, _ in traj.get_positions()], np.int32)
            img_poly = cv2.polylines(img, [xx],
                                     isClosed=False,
                                     color=(int(traj.get_color()[0] * 255), int(traj.get_color()[1] * 255),
                                            int(traj.get_color()[2] * 255)),
                                     thickness=1)
    if add_index:
        for traj in trajectory_list:
            if traj.get_trajectory_length() >= cutoff:
                xx = np.array([[int(x * (10**amp)), int(y * (10**amp))]
                               for x, y, _ in traj.get_positions()], np.int32)
                cv2.putText(img, f'{  traj.get_index()}', org=xx[0], fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4,
                            color=(int(traj.get_color()[0] * 255), int(traj.get_color()[1] * 255),
                                   int(traj.get_color()[2] * 255)))
    if add_time:
        for traj in trajectory_list:
            if traj.get_trajectory_length() >= cutoff:
                xx = np.array([[int(x * (10**amp)), int(y * (10**amp))]
                               for x, y, _ in traj.get_positions()], np.int32)
                cv2.putText(img, f'[{traj.get_times()[0]},{traj.get_times()[-1]}]',
                            org=[xx[0][0], xx[0][1] + 12], fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4,
                            color=(int(traj.get_color()[0] * 255), int(traj.get_color()[1] * 255),
                                   int(traj.get_color()[2] * 255)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(output, img)


def make_image_seqs2(*trajectory_lists, output_dir, time_steps, cutoff=0, original_shape=(512, 512),
                    target_shape=(512, 512), amp=0, add_index=True):
    """
    Use:
    make_image_seqs(gt_list, trajectory_list, output_dir=output_img, time_steps=time_steps, cutoff=1,
    original_shape=(images.shape[1], images.shape[2]), target_shape=(1536, 1536), add_index=True)
    """
    img_origin = np.zeros((target_shape[0] * (10**amp), target_shape[1] * (10**amp), 3), dtype=np.uint8)
    result_stack = []
    x_amp = img_origin.shape[0] / original_shape[0]
    y_amp = img_origin.shape[1] / original_shape[1]
    for frame in time_steps:
        img_stack = []
        for trajectory_list in trajectory_lists:
            img = img_origin.copy()
            for traj in trajectory_list:
                times = traj.get_times()
                if times[-1] < frame - 2:
                    continue
                indices = [i for i, time in enumerate(times) if time <= frame]
                if traj.get_trajectory_length() >= cutoff:
                    xy = np.array([[int(x * x_amp), int(y * y_amp)]
                                   for x, y, _ in traj.get_positions()[indices]], np.int32)
                    font_scale = 0.1 * x_amp
                    img_poly = cv2.polylines(img, [xy],
                                             isClosed=False,
                                             color=(int(traj.get_color()[0] * 255), int(traj.get_color()[1] * 255),
                                                    int(traj.get_color()[2] * 255)),
                                             thickness=1)
                    for x, y in xy:
                        cv2.circle(img, (x, y), radius=1, color=(255, 255, 255), thickness=-1)
                    if len(indices) > 0:
                        cv2.putText(img, f'[{times[indices[0]]},{times[indices[-1]]}]',
                                    org=[xy[0][0], xy[0][1] + 12], fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale,
                                    color=(int(traj.get_color()[0] * 255), int(traj.get_color()[1] * 255),
                                           int(traj.get_color()[2] * 255)))
                        if add_index:
                            cv2.putText(img, f'{traj.get_index()}', org=xy[0], fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=font_scale,
                                        color=(int(traj.get_color()[0] * 255), int(traj.get_color()[1] * 255),
                                               int(traj.get_color()[2] * 255)))
            img[:, -1, :] = 255
            img_stack.append(img)
        hstacked_img = np.hstack(img_stack)
        result_stack.append(hstacked_img)
    result_stack = np.array(result_stack)
    tifffile.imwrite(output_dir, data=result_stack, imagej=True)


def make_image_seqs_old(trajectory_list, output_dir, img_stacks, time_steps, cutoff=2,
                        add_index=True, local_img=None, gt_trajectory=None, cps_result=None):
    if np.mean(img_stacks) < 0.35:
        bright_ = 1
    else:
        bright_ = 0

    if img_stacks.shape[1] * img_stacks.shape[2] < 256 * 256:
        upscailing_factor = 2  # int(512 / img_stacks.shape[1])
    else:
        upscailing_factor = 1
    result_stack = []
    for img, frame in zip(img_stacks, time_steps):
        img = cv2.resize(img, (img.shape[1]*upscailing_factor, img.shape[0]*upscailing_factor),
                         interpolation=cv2.INTER_AREA)
        if img.ndim == 2:
            img = np.array([img, img, img])
            img = np.moveaxis(img, 0, 2)
        img = np.ascontiguousarray(img)
        img_org = img.copy()
        if local_img is not None:
            local_img = img_org.copy()
            for traj in trajectory_list:
                times = traj.get_times()
                if frame in times:
                    indices = [i for i, time in enumerate(times) if time == frame]
                    xy = np.array([[int(np.around(x * upscailing_factor)), int(np.around(y * upscailing_factor))]
                                   for x, y, _ in traj.get_positions()[indices]], np.int32)
                    if local_img[xy[0][1], xy[0][0], 0] == 1 and local_img[xy[0][1], xy[0][0], 1] == 0 and local_img[xy[0][1], xy[0][0], 2] == 0:
                        local_img = draw_cross(local_img, xy[0][1], xy[0][0], (0, 0, 1))
                    else:
                        local_img = draw_cross(local_img, xy[0][1], xy[0][0], (1, 0, 0))
            local_img[:, -1, :] = 1

        if bright_:
            overlay = np.zeros(img.shape)
        else:
            overlay = np.ones(img.shape)
        for traj in trajectory_list:
            times = traj.get_times()
            if times[-1] < frame:
                continue
            indices = [i for i, time in enumerate(times) if time <= frame]
            if traj.get_trajectory_length() >= cutoff:
                xy = np.array([[int(np.around(x * upscailing_factor)), int(np.around(y * upscailing_factor))]
                               for x, y, _ in traj.get_positions()[indices]], np.int32)
                font_scale = 0.1 * 2
                img_poly = cv2.polylines(overlay, [xy],
                                         isClosed=False,
                                         color=(traj.get_color()[0],
                                                traj.get_color()[1],
                                                traj.get_color()[2]),
                                         thickness=1)
                if len(indices) > 0:
                    if add_index:
                        cv2.putText(overlay, f'[{times[indices[0]]},{times[indices[-1]]}]',
                                    org=[xy[-1][0], xy[-1][1] + 12], fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=font_scale,
                                    color=(traj.get_color()[0],
                                           traj.get_color()[1],
                                           traj.get_color()[2]))
                        cv2.putText(overlay, f'{traj.get_index()}', org=xy[-1], fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=font_scale,
                                    color=(traj.get_color()[0],
                                           traj.get_color()[1],
                                           traj.get_color()[2]))
        #img_org[:, -1, :] = 1
        if bright_:
            overlay = img_org + overlay
        else:
            overlay = img_org * overlay
        overlay = np.minimum(np.ones_like(overlay), overlay)
        if local_img is not None:
            hstacked_img = np.hstack((local_img, overlay))
        else:
            hstacked_img = overlay

        if gt_trajectory is not None:
            overlay = img.copy()
            for traj in gt_trajectory:
                times = traj.get_times()
                if times[-1] < frame:
                    continue
                indices = [i for i, time in enumerate(times) if time <= frame]
                if traj.get_trajectory_length() >= cutoff:
                    xy = np.array([[int(np.around(x * upscailing_factor)), int(np.around(y * upscailing_factor))]
                                   for x, y, _ in traj.get_positions()[indices]], np.int32)
                    font_scale = 0.1 * 2
                    img_poly = cv2.polylines(overlay, [xy],
                                             isClosed=False,
                                             color=(traj.get_color()[0],
                                                    traj.get_color()[1],
                                                    traj.get_color()[2]),
                                             thickness=1)
                    if len(indices) > 0:
                        if add_index:
                            cv2.putText(overlay, f'[{times[indices[0]]},{times[indices[-1]]}]',
                                        org=[xy[0][0], xy[0][1] + 12], fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=font_scale,
                                        color=(traj.get_color()[0],
                                               traj.get_color()[1],
                                               traj.get_color()[2]))
                            cv2.putText(overlay, f'{traj.get_index()}', org=xy[0], fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=font_scale,
                                        color=(traj.get_color()[0],
                                               traj.get_color()[1],
                                               traj.get_color()[2]))
            hstacked_img[:, -1, :] = 1
            hstacked_img = np.hstack((hstacked_img, overlay))
        result_stack.append(hstacked_img)
    result_stack = (np.array(result_stack) * 255).astype(np.uint8)

    if cps_result is not None:
        for traj_obj in trajectory_list:
            xyzs = traj_obj.get_positions()
            traj_idx = traj_obj.get_index()
            init_time = traj_obj.get_times()[0]
            cps = cps_result[traj_idx][3][:-1].astype(int)
            if len(cps) > 0:
                cps_set = set(np.array([[cp-1, cp, cp+1] for cp in cps]).flatten())
                cps_rad = {}
                for cp in cps:
                    for i, cpk in enumerate(range(cp-1, cp+2)):
                        cps_rad[cpk] = int(i*1 + 3)
                cp_xs = xyzs[:, 0]
                cp_ys = xyzs[:, 1]
                cp_zs = xyzs[:, 2]
                for frame in time_steps:
                    if frame in cps_set:
                        print(f'CPs containing frame: {init_time + frame}')
                        circle_overlay = cv2.circle(result_stack[init_time + frame], center=(int(np.around(cp_xs[frame] * upscailing_factor)), int(np.around(cp_ys[frame] * upscailing_factor))),
                                                    radius=cps_rad[frame], color=(255, 0, 0))
    tifffile.imwrite(output_dir, data=result_stack, imagej=True)


def make_image_seqs(trajectory_list, output_dir, img_stacks, time_steps):
    ret_img_stacks = []
    for img, frame in zip(img_stacks, time_steps):
        if img.ndim == 2:
            img = np.array([img, img, img])
            img = np.moveaxis(img, 0, 2)
        img = np.ascontiguousarray(img)
        for traj in trajectory_list:
            times = traj.get_times()
            if times[-1] < frame:
                continue
            indices = [i for i, time in enumerate(times) if time <= frame]
            if traj.get_trajectory_length() >= 2:
                xy = np.array([[int(np.around(x)), int(np.around(y))]
                               for x, y, _ in traj.get_positions()[indices]], np.int32)
                img_poly = cv2.polylines(img, [xy],
                                         isClosed=False,
                                         color=(traj.get_color()[0],
                                                traj.get_color()[1],
                                                traj.get_color()[2]),
                                         thickness=1)
        ret_img_stacks.append(img)
    ret_img_stacks = (np.array(ret_img_stacks)*255).astype(np.uint8)
    tifffile.imwrite(output_dir, data=ret_img_stacks, imagej=True)


def remake_visual_trajectories(output_path:str, trajectory_file:str, raw_imgs:str, start_frame=1, end_frame=10000, upscaling:int=1):
    assert 'traces.csv' in trajectory_file.split('/')[-1], "input trajectory file format is wrong, it needs video_traces.csv"
    assert '.tif' in raw_imgs.split('/')[-1], "input image file format is wrong, it needs image.tiff"
    filename = raw_imgs.strip().split('/')[-1].split('.tif')[0]
    traj_list = read_trajectory(trajectory_file)
    img_stacks = read_tif(raw_imgs)
    start_frame = max(0, start_frame) - 1
    end_frame = min(img_stacks.shape[0], end_frame)
    img_stacks = img_stacks[start_frame:end_frame, :, :]

    ret_img_stacks = []
    for img, frame in zip(img_stacks, np.arange(start_frame, end_frame, 1) + 1):
        if img.ndim == 2:
            img = np.array([img, img, img])
            img = np.moveaxis(img, 0, 2)
        img = np.ascontiguousarray(img)
        if upscaling >= 2:
            img = cv2.resize(img, (img.shape[1]*upscaling, img.shape[0]*upscaling),
                             interpolation=cv2.INTER_AREA)
        for traj in traj_list:
            times = traj.get_times()
            if times[-1] < frame:
                continue
            indices = [i for i, time in enumerate(times) if time <= frame]
            if traj.get_trajectory_length() >= 2:
                xy = np.array([[int(np.around(x * upscaling)), int(np.around(y * upscaling))]
                               for x, y, _ in traj.get_positions()[indices]], np.int32)
                img_poly = cv2.polylines(img, [xy],
                                         isClosed=False,
                                         color=(traj.get_color()[0],
                                                traj.get_color()[1],
                                                traj.get_color()[2]),
                                         thickness=1)
        ret_img_stacks.append((img * 255).astype(np.uint8))
    ret_img_stacks = np.array(ret_img_stacks, dtype=np.uint8)
    tifffile.imwrite(f'{output_path}/{filename}_{start_frame+1}_{end_frame}_tracevideo.tiff', data=ret_img_stacks, imagej=True)
    print(f'{output_path}/{filename}_{start_frame+1}_{end_frame}_tracevideo.tiff is successfully generated.')


def remake_visual_localizations(output_path:str, localization_file:str, raw_imgs:str, start_frame=1, end_frame=10000, upscaling:int=1):
    assert 'loc.csv' in localization_file.split('/')[-1], "input localization file format is wrong, it needs video_loc.csv"
    assert '.tif' in raw_imgs.split('/')[-1], "input image file format is wrong, it needs image.tiff"
    filename = raw_imgs.strip().split('/')[-1].split('.tif')[0]
    loc_list, loc_info = read_localization(localization_file)
    img_stacks = read_tif(raw_imgs)
    start_frame = max(1, start_frame)
    end_frame = min(img_stacks.shape[0], end_frame)
    img_stacks = img_stacks[start_frame-1:end_frame, :, :]

    ret_img_stacks = []
    for img_idx, frame in enumerate(np.arange(start_frame, end_frame+1, 1)):
        coords = loc_list[frame]
        img = img_stacks[img_idx]
        img = (img * 255).astype(np.uint8)
        if img.ndim == 2:
            img = np.array([img, img, img])
            img = np.moveaxis(img, 0, 2)
        img = np.ascontiguousarray(img)
        if upscaling >= 2:
            img = cv2.resize(img, (img.shape[1]*upscaling, img.shape[0]*upscaling),
                             interpolation=cv2.INTER_AREA)
        xy_cum = []
        for center_coord in coords:
            x, y = int(round(center_coord[1] * upscaling)), int(round(center_coord[0] * upscaling))
            if (x, y) in xy_cum:
                img = draw_cross(img, x, y, (0, 0, 1))
            else:
                img = draw_cross(img, x, y, (1, 0, 0))
            xy_cum.append((x, y))

        ret_img_stacks.append(img)
    ret_img_stacks = np.array(ret_img_stacks, dtype=np.uint8)
    tifffile.imwrite(f'{output_path}/{filename}_{start_frame}_{end_frame}_locvideo.tiff', data=ret_img_stacks, imagej=True)
    print(f'{output_path}/{filename}_{start_frame}_{end_frame}_locvideo.tiff is successfully generated.')


def make_loc_radius_video(output_path:str, raw_imgs:str, localization_file:str, frame_cumul=100, radius=[1, 10], start_frame=1, end_frame=10000, alpha1=0.65, alpha2=0.35, gpu=False):
    assert 'loc.csv' in localization_file.split('/')[-1], "input localization file format is wrong, it needs video_loc.csv"
    assert len(radius) == 2, "radius should be 2 length of list such as [1, 10]."
    assert radius[0] < radius[1] and radius[0] > 0, "radius[0] should be smaller than radius[1] and radius[0] should be greater than 0."
    assert 0.999 < alpha1 + alpha2 < 1.001, "Sum of alpha1 and alpha2 should be equal to 1."
    coords, coord_info = read_localization(localization_file)
    sequence_save_folder = f'{output_path}'
    filename = localization_file.split('/')[-1].split('.csv')[0]
    if not os.path.exists(sequence_save_folder):
        os.mkdir(sequence_save_folder)

    if gpu:
        import cupy as cp
        from cuvs.distance import pairwise_distance
        mempool = cp.get_default_memory_pool()
        mempool.set_limit(fraction=0.8)

    time_steps = np.array(sorted(list(coords.keys())))
    end_frame = min(end_frame, time_steps[-1])
    start_frame = max(start_frame, time_steps[0])
    images = read_tif(raw_imgs)[start_frame-1:end_frame,:,:]
    PBAR = tqdm(total=end_frame - start_frame + 1, desc="Radius calcul", unit="frame", ncols=120)

    all_coords = []
    stacked_coords = {t:[] for t in time_steps if start_frame <= t <= end_frame}
    stacked_radii = {t:[] for t in time_steps if start_frame <= t <= end_frame}
    count_max = 0

    for t in time_steps:
        if start_frame <= t <= end_frame:
            PBAR.update(1)
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
    PBAR.close()
    if len(all_coords) == 0:
        return

    image_idx = 0
    x_min = np.min(all_coords[:, 0])
    x_max = np.max(all_coords[:, 0])
    y_min = np.min(all_coords[:, 1])
    y_max = np.max(all_coords[:, 1])
    mycmap = plt.get_cmap('jet', lut=None)
    video_arr = np.empty([images.shape[0], images.shape[1], images.shape[2], 4], dtype=np.uint8)
    PBAR = tqdm(total=video_arr.shape[0], desc="Rendering", unit="frame", ncols=120)
    X, Y = np.mgrid[x_min:x_max:complex(f'{images.shape[2]}j'), y_min:y_max:complex(f'{images.shape[1]}j')]
    positions = np.vstack([X.ravel(), Y.ravel()])
    for time in time_steps:
        if start_frame <= time <= end_frame:
            PBAR.update(1)
            if images[image_idx: image_idx + frame_cumul,:,:].shape[0] == 0:
                break
            selected_coords = stacked_coords[time]
            selected_radii = stacked_radii[time]
            if len(selected_coords) > 0:
                values = np.vstack([selected_coords[:, 0], selected_coords[:, 1]])
                kernel = stats.gaussian_kde(values, weights=(selected_radii / count_max))
                kernel.set_bandwidth(bw_method=kernel.factor / 2.)
                Z = np.reshape(kernel(positions).T, X.shape)
                Z = Z * (np.max(selected_radii) / np.max(Z))
                aximg = plt.imshow(Z.T, cmap=mycmap,
                                   extent=[x_min, x_max, y_min, y_max], vmin=0, vmax=count_max, alpha=1.0, origin='upper')
                rawimg = plt.imshow(images[image_idx: image_idx + frame_cumul,:,:].max(axis=(0)), alpha=1.0, cmap='grey', extent=[x_min, x_max, y_min, y_max], origin='upper')
                arr = aximg.make_image(renderer=None, unsampled=True)[0][:,:,:4]
                arr2 = rawimg.make_image(renderer=None, unsampled=True)[0][:,:,:4]
                blended = cv2.addWeighted(arr, alpha1, arr2, alpha2, 0.0)
                video_arr[image_idx] = blended
            image_idx += 1

    tifffile.imwrite(f'{sequence_save_folder}/{filename}_density_video_frame_{start_frame}_{end_frame}_radius_{radius[0]}_{radius[1]}_cumul_{frame_cumul}.tiff', data=video_arr, imagej=True)
    PBAR.close()
    print(f'{sequence_save_folder}/{filename}_density_video_frame_{start_frame}_{end_frame}_radius_{radius[0]}_{radius[1]}_cumul_{frame_cumul}.tiff is successfully generated.')


def make_red_circles(imgs, localized_xys, hstack=False):
    if hstack:
        original_imgs = imgs.copy()
    stacked_imgs = []
    for img_n, coords in enumerate(localized_xys):
        xy_cum = []
        for center_coord in coords:
            x, y = int(round(center_coord[0])), int(round(center_coord[1]))
            if (x, y) in xy_cum:
                imgs[img_n] = draw_cross(imgs[img_n], x, y, (0, 0, 1))
            else:
                imgs[img_n] = draw_cross(imgs[img_n], x, y, (1, 0, 0))
            xy_cum.append((x, y))
        if hstack:
            stacked_imgs.append(np.hstack((original_imgs[img_n], imgs[img_n])))
        else:
            stacked_imgs.append(imgs[img_n])
    return np.array(stacked_imgs, dtype=np.uint8)


def make_whole_img(trajectory_list, output_dir, img_stacks):
    if img_stacks.shape[1] * img_stacks.shape[2] < 1024 * 1024:
        upscailing_factor = int(1024 / img_stacks.shape[1])
    else:
        upscailing_factor = 1
    imgs = np.zeros((img_stacks.shape[1] * upscailing_factor, img_stacks.shape[2] * upscailing_factor, 3))
    for traj in trajectory_list:
        xy = np.array([[int(np.around(x * upscailing_factor)), int(np.around(y * upscailing_factor))]
                       for x, y, _ in traj.get_positions()], np.int32)
        img_poly = cv2.polylines(imgs, [xy],
                                 isClosed=False,
                                 color=(traj.get_color()[2],
                                        traj.get_color()[1],
                                        traj.get_color()[0]),
                                 thickness=1)
    cv2.imwrite(output_dir, (imgs * 255).astype(np.uint8))


def draw_cross(img, row, col, color):
    comb = [[row-2, col], [row-1, col], [row, col], [row+1, col], [row+2, col], [row, col-2], [row, col-1], [row, col+1], [row, col+2]]
    for r, c in comb:
        if 0 <= r < img.shape[0] and 0 <= c < img.shape[1]:
            for i, couleur in enumerate(color):
                if couleur >= 1:
                    img[r, c, i] = 255
                else:
                    img[r, c, i] = 0
    return img


def compare_two_localization_visual(output_dir, images, localized_xys_1, localized_xys_2):
    orignal_imgs_3ch = np.array([images.copy(), images.copy(), images.copy()])
    orignal_imgs_3ch = np.ascontiguousarray(np.moveaxis(orignal_imgs_3ch, 0, 3))
    original_imgs_3ch_2 = orignal_imgs_3ch.copy()
    stacked_imgs = []
    frames = np.sort(list(localized_xys_1.keys()))
    for img_n in frames:
        for center_coord in localized_xys_1[img_n]:
            if (center_coord[0] > orignal_imgs_3ch.shape[1] or center_coord[0] < 0
                    or center_coord[1] > orignal_imgs_3ch.shape[2] or center_coord[1] < 0):
                print("ERR")
                print(img_n, 'row:', center_coord[0], 'col:', center_coord[1])
            x, y = int(round(center_coord[1])), int(round(center_coord[0]))
            orignal_imgs_3ch[img_n-1][x][y][0] = 1
            orignal_imgs_3ch[img_n-1][x][y][1] = 0
            orignal_imgs_3ch[img_n-1][x][y][2] = 0

        for center_coord in localized_xys_2[img_n]:
            if (center_coord[0] > original_imgs_3ch_2.shape[1] or center_coord[0] < 0
                    or center_coord[1] > original_imgs_3ch_2.shape[2] or center_coord[1] < 0):
                print("ERR")
                print(img_n, 'row:', center_coord[0], 'col:', center_coord[1])
            x, y = int(round(center_coord[1])), int(round(center_coord[0]))
            original_imgs_3ch_2[img_n-1][x][y][0] = 1
            original_imgs_3ch_2[img_n-1][x][y][1] = 0
            original_imgs_3ch_2[img_n-1][x][y][2] = 0
        stacked_imgs.append(np.hstack((orignal_imgs_3ch[img_n-1], original_imgs_3ch_2[img_n-1])))
    stacked_imgs = np.array(stacked_imgs)
    tifffile.imwrite(f'{output_dir}/local_comparison.tiff', data=(stacked_imgs * 255).astype(np.uint8), imagej=True)


def concatenate_image_stack(output_fname, org_img, concat_img):
    org_img = read_tif(org_img)
    org_img = (org_img * 255).astype(np.uint8)
    concat_img = read_tif_unnormalized(concat_img)
    if org_img.shape != concat_img.shape:
        tmp_img = np.zeros_like(concat_img)
        for i, o_img in enumerate(org_img):
            o_img = cv2.resize(o_img, (concat_img.shape[2], concat_img.shape[1]), interpolation=cv2.INTER_AREA)
            for channel in range(3):
                tmp_img[i, :, :, channel] = o_img
    org_img = tmp_img
    org_img[:,:,-1,:] = 255
    stacked_imgs = np.concatenate((org_img, concat_img), axis=2)
    tifffile.imwrite(f'{output_fname}_hconcat.tiff', data=stacked_imgs, imagej=True)


def load_datas(datapath):
    if datapath.endswith(".csv"):
        df = pd.read_csv(datapath)
        return df
    else:
        None


def cps_visualization(image_save_path, video, cps_result, trace_result):
    cps_trajectories = {}
    try:
        with open(cps_result, 'r') as cp_file:
            lines = cp_file.readlines()
            for line in lines[:-1]:
                line = line.strip().split(',')
                traj_index = int(line[0])
                cps_trajectories[traj_index] = [[], [], [], []] # diffusion_coef, alpha, traj_type, changepoint
                for idx, data in enumerate(line[1:]):
                    cps_trajectories[traj_index][idx % 4].append(float(data))
                cps_trajectories[traj_index] = np.array(cps_trajectories[traj_index])
        df = load_datas(trace_result)
        video = read_tif(video)
        if video.shape[0] <= 1:
            sys.exit('Image squence length error: Cannot track on a single image.')
    except Exception as e:
        print(e)
        print('File load failed.')

    time_steps = []
    trajectory_list = []
    for traj_idx in cps_trajectories.keys():
        frames = np.array(df[df.traj_idx == traj_idx])[:, 1].astype(int)
        xs = np.array(df[df.traj_idx == traj_idx])[:, 2]
        ys = np.array(df[df.traj_idx == traj_idx])[:, 3]
        obj = TrajectoryObj(traj_idx)
        for t, x, y, z in zip(frames, xs, ys, np.zeros_like(xs)):
            obj.add_trajectory_position(t, x, y, z)
            time_steps.append(t)
        trajectory_list.append(obj)
    time_steps = np.arange(video.shape[0])
    make_image_seqs(trajectory_list, output_dir=image_save_path, img_stacks=video, time_steps=time_steps, cutoff=2,
                    add_index=False, local_img=None, gt_trajectory=None, cps_result=cps_trajectories)


def make_loc_depth_image(output_dir:str, coords:list, multiplier=4, winsize=7, resolution=2, dim=2):
    assert len(coords) > 0, "no coordinate data or path"

    resolution = int(max(1, min(3, resolution)))  # resolution in [1, 2, 3]
    amp = 1
    multiplier = multiplier - 1 if multiplier % 2 == 1 else multiplier
    winsize += multiplier * resolution
    cov_std = multiplier * resolution
    amp_ = 10**amp
    margin_pixel = 2
    margin_pixel *= 10*amp_
    amp_*= resolution

    if type(coords[0]) is str:
        locs, infos = read_multiple_locs(coords)
        time_steps = sorted(list(locs.keys()))
        all_coords = []
        for t in time_steps:
            all_coords.extend(locs[t])
        all_coords = np.array(all_coords)
        # xy exchange
        xy_flip = all_coords[:,1].copy()
        all_coords[:,1] = all_coords[:,0]
        all_coords[:,0] = xy_flip
    else:
        time_steps = np.arange(len(coords))
        all_coords = []
        for t in time_steps:
            for coord in coords[t]:
                all_coords.append(coord)
        all_coords = np.array(all_coords)
        if len(all_coords) == 0:
            return

    x_min = np.min(all_coords[:, 1])
    x_max = np.max(all_coords[:, 1])
    y_min = np.min(all_coords[:, 0])
    y_max = np.max(all_coords[:, 0])
    z_min = np.min(all_coords[:, 2])
    z_max = np.max(all_coords[:, 2])
    z_min, z_max = np.quantile(all_coords[:, 2], [0.01, 0.99])
    all_coords[:, 1] -= x_min
    all_coords[:, 0] -= y_min

    if dim == 2:
        mycmap = plt.get_cmap('hot', lut=None)
        color_seq = [mycmap(i)[:3] for i in range(mycmap.N)]
        image = np.zeros((int((y_max - y_min)*amp_ + margin_pixel), int((x_max - x_min)*amp_ + margin_pixel)), dtype=np.float32)
        all_coords = np.round(all_coords * amp_)
        template = np.ones((1, (winsize)**2, 2), dtype=np.float32) * quantification(winsize)
        template = (np.exp((-1./2) * np.sum(template @ np.linalg.inv([[cov_std, 0], [0, cov_std]]) * template, axis=2))).reshape([winsize, winsize])

        for roundup_coord in all_coords:
            coord_col = int(roundup_coord[1] + margin_pixel//2)
            coord_row = int(roundup_coord[0] + margin_pixel//2)
            row = min(max(0, coord_row), image.shape[0])
            col = min(max(0, coord_col), image.shape[1])
            image[row - winsize//2: row + winsize//2 + 1, col - winsize//2: col + winsize//2 + 1] += template
        
        img_min, img_max = np.quantile(image, [0.01, 0.995])
        image = np.minimum(image, np.ones_like(image) * img_max)
        image = image / np.max(image)
        image = (image * 255).astype(np.uint8)
        image = plt.imshow(image, cmap=mycmap, origin='upper')
        image = image.make_image(renderer=None, unsampled=True)[0][:,:,:3]
        image = Image.fromarray(image)
        image.save(f'{output_dir}_loc_{dim}d_density.png', dpi=(300, 300))
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
            coord_col = int(roundup_coord[1] + margin_pixel//2)
            coord_row = int(roundup_coord[0] + margin_pixel//2)
            color_z = color_seq[min(int(np.round(len(color_seq) * z_coord)), len(color_seq)-1)]
            row = min(max(0, coord_row), image.shape[0])
            col = min(max(0, coord_col), image.shape[1])
            image[row - winsize//2: row + winsize//2 + 1, col - winsize//2: col + winsize//2 + 1, 0] += template * color_z[0]
            image[row - winsize//2: row + winsize//2 + 1, col - winsize//2: col + winsize//2 + 1, 1] += template * color_z[1]
            image[row - winsize//2: row + winsize//2 + 1, col - winsize//2: col + winsize//2 + 1, 2] += template * color_z[2]
        
        img_min, img_max = np.quantile(image, [0.01, 0.995])
        image = np.minimum(image, np.ones_like(image) * img_max)
        image = image / np.max(image)
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
        image.save(f'{output_dir}_loc_{dim}d_density.png', dpi=(300, 300))


def quantification(window_size):
    x = np.arange(-(window_size-1)/2, (window_size+1)/2)
    y = np.arange(-(window_size-1)/2, (window_size+1)/2)
    xv, yv = np.meshgrid(x, y, sparse=True)
    grid = np.stack(np.meshgrid(xv, yv), -1).reshape(window_size * window_size, 2)
    return grid.astype(np.float32)


def to_gif(image_stack_path, save_path, fps=10, loop=30):
    images = read_tif_unnormalized(image_stack_path).astype(np.uint8)
    with imageio.get_writer(f'{save_path}.gif', mode='I', fps=fps, loop=loop) as writer:
        for i in range(len(images)):
            writer.append_data(np.array(images[i]))


def to_mp4(image_stack_path, save_path, fps=10, resolution='high'):
    images = read_tif_unnormalized(image_stack_path)
    if resolution == 'high':
        fourcc = cv2.VideoWriter_fourcc(*'HFYU') #lossless
    else:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
    v_out = cv2.VideoWriter(f'{save_path}.avi', fourcc, fps, (images.shape[2], images.shape[1]))
    for idx in range(images.shape[0]):
        video_frame = cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB)
        v_out.write(video_frame)
    v_out.release()


def animation(image_stack_path, save_path, fps=10, resolution='high'):
    images = read_tif_unnormalized(image_stack_path)
    if resolution == 'high':
        fourcc = cv2.VideoWriter_fourcc(*'HFYU') #lossless
    else:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
    v_out = cv2.VideoWriter(f'{save_path}.avi', fourcc, fps, (images.shape[2], images.shape[1]))
    for idx in range(images.shape[0]):
        video_frame = cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB)
        v_out.write(video_frame)
    v_out.release()

#vis_cps_file_name = ''
#cps_visualization(f'./{vis_cps_file_name}_cps.tiff', f'./inputs/{vis_cps_file_name}.tiff', f'./{vis_cps_file_name}_traces.txt', f'./outputs/{vis_cps_file_name}_traces.csv')
#concatenate_image_stack(f'{vis_cps_file_name}', f'./{vis_cps_file_name}.tiff', f'./{vis_cps_file_name}_cps.tiff')
#to_gif(f'./outputs/3.tif', f'./outputs/3', fps=20, loop=2)
#to_mp4('outputs/alpha_test10_locvideo.tiff', 'outputs/vid')