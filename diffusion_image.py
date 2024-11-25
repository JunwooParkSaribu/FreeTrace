import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from module.FileIO import read_trajectory
from module import TrajectoryObject


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
