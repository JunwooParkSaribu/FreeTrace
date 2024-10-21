import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2


class TrajectoryObj:
    def __init__(self, index, localizations=None, max_pause=1):
        self.index = index
        self.paused_time = 0
        self.max_pause = max_pause
        self.trajectory_tuples = []
        self.localizations = localizations
        self.times = []
        self.closed = False
        self.color = (np.random.randint(0, 256)/255.,
                      np.random.randint(0, 256)/255.,
                      np.random.randint(0, 256)/255.)
        self.optimality = 0.
        self.positions = []

    def add_trajectory_tuple(self, next_time, next_position):
        assert self.localizations is not None
        self.trajectory_tuples.append((next_time, next_position))
        x, y, z = self.localizations[next_time][next_position][:3]
        self.positions.append([x, y, z])
        self.times.append(next_time)
        self.paused_time = 0

    def get_trajectory_tuples(self):
        return self.trajectory_tuples

    def add_trajectory_position(self, time, x, y, z):
        self.times.append(time)
        self.positions.append([x, y, z])
        self.paused_time = 0

    def get_positions(self):
        return np.array(self.positions)

    def trajectory_status(self):
        return self.closed

    def close(self):
        self.paused_time = 0
        self.closed = True

    def wait(self):
        if self.paused_time == self.max_pause:
            self.close()
            return self.trajectory_status()
        else:
            self.paused_time += 1
            return self.trajectory_status()

    def get_index(self):
        return self.index

    def get_times(self):
        return np.array(self.times)

    def set_color(self, color):
        self.color = color

    def get_color(self):
        return self.color

    def set_trajectory_tuple(self, trajectory):
        self.trajectory_tuples = trajectory
        self.paused_time = 0

    def get_last_tuple(self):
        return self.trajectory_tuples[-1]

    def get_trajectory_length(self):
        return len(self.get_positions())

    def get_paused_time(self):
        return self.paused_time

    def set_optimality(self, val):
        self.optimality = val

    def get_optimality(self):
        return self.optimality

    def get_expected_pos(self, t):
        if len(self.get_times()) < t+1:
            return np.array(self.positions[-1]), None
        else:
            vector = (np.array(self.positions[-1]) - np.array(self.positions[-1 - t])) / t
            return np.array(self.positions[-1]) + (vector * (self.paused_time + 1)), np.sqrt(vector[0]**2 + vector[1]**2)

    def delete(self, cutoff=2):
        if len(self.positions) < cutoff:
            return True
        else:
            return False

    def get_inst_diffusion_coefs(self, time_interval, t_range=None, ndim=2):
        if t_range is None:
            t_range = [0, len(self.get_positions())]
        considered_positions = self.get_positions()[t_range[0]: t_range[1]]
        considered_times = self.get_times()[t_range[0]: t_range[1]]

        diff_coefs = []
        for i in range(len(considered_positions) - 1):
            j = i + 1
            prev_x, prev_y, prev_z = considered_positions[i]
            prev_t = considered_times[i]
            x, y, z = considered_positions[j]
            t = considered_times[j]
            diff_coef = ((x - prev_x) ** 2 + (y - prev_y) ** 2 + (z - prev_z) ** 2) / (2 * ndim * (t - prev_t))
            diff_coefs.append(diff_coef)
        diff_coefs = np.array(diff_coefs)
        diff_coefs_intervals = []
        for i in range(len(diff_coefs)):
            left_idx = i - time_interval//2
            right_idx = i + time_interval//2
            diff_coefs_intervals.append(np.mean(diff_coefs[max(0, left_idx):min(len(diff_coefs), right_idx+1)]))

        # make length equal to length of xy pos
        #diff_coefs_intervals.append(0.0)
        return np.array(diff_coefs_intervals)

    def get_trajectory_angles(self, time_interval, t_range=None):
        """
        available only for 2D data.
        """
        if t_range is None:
            t_range = [0, len(self.get_positions())]
        considered_positions = self.get_positions()[t_range[0]: t_range[1]]
        considered_times = self.get_times()[t_range[0]: t_range[1]]

        angles = []
        for i in range(len(considered_positions) - 2):
            prev_x, prev_y, prev_z = considered_positions[i]
            prev_t = considered_times[i]
            x, y, z = considered_positions[i+1]
            t = considered_times[i+1]
            next_x, next_y, next_z = considered_positions[i+2]
            next_t = considered_times[i+2]
            vec_prev_cur = np.array([x - prev_x, y - prev_y, z - prev_z]) / (t - prev_t)
            vec_cur_next = np.array([next_x - x, next_y - y, next_z - z]) / (next_t - t)

            ang = np.arccos((vec_prev_cur @ vec_cur_next) /
                            (np.sqrt(vec_prev_cur[0] ** 2 + vec_prev_cur[1] ** 2 + vec_prev_cur[2] ** 2)
                             * np.sqrt(vec_cur_next[0] ** 2 + vec_cur_next[1] ** 2 + vec_cur_next[2] ** 2)))
            angles.append(ang)
        angles = np.array(angles)

        angles_intervals = []
        for i in range(len(angles)):
            left_idx = i - time_interval//2
            right_idx = i + time_interval//2
            angles_intervals.append(np.mean(angles[max(0, left_idx):min(len(angles), right_idx+1)]))

        # make length equal to length of xy pos
        angles_intervals.append(0.0)
        angles_intervals.append(0.0)
        return np.array(angles_intervals)

    def get_msd(self, time_interval, t_range=None):
        if t_range is None:
            t_range = [0, len(self.get_positions())]
        considered_positions = self.get_positions()[t_range[0]: t_range[1]]
        considered_times = self.get_times()[t_range[0]: t_range[1]]

        MSD = []
        for (x, y, z), t in zip(considered_positions, considered_times):
            MSD.append(np.sqrt((x - considered_positions[0][0]) ** 2 +
                               (y - considered_positions[0][1]) ** 2 +
                               (z - considered_positions[0][2]) ** 2))
        MSD = np.array(MSD)

        MSD_intervals = []
        for i in range(len(MSD)):
            left_idx = i - time_interval//2
            right_idx = i + time_interval//2
            MSD_intervals.append(np.mean(MSD[max(0, left_idx):min(len(MSD), right_idx+1)]))
        return np.array(MSD_intervals)

    def get_density(self, radius, t_range=None):
        if t_range is None:
            t_range = [0, len(self.get_positions())]
        considered_positions = self.get_positions()[t_range[0]: t_range[1]]
        considered_times = self.get_times()[t_range[0]: t_range[1]]

        density = []
        for i in range(len(considered_positions)):
            nb = 0
            for (x, y, z), t in zip(considered_positions, considered_times):
                disp = np.sqrt((x - considered_positions[i][0]) ** 2 +
                               (y - considered_positions[i][1]) ** 2 +
                               (z - considered_positions[i][2]) ** 2)
                if disp < radius:
                    nb += 1
            density.append(nb)
        return np.array(density).astype(np.float32)


def read_trajectory(file: str) -> dict | list:
    """
    @params : filename(String), cutoff value(Integer)
    @return : dictionary of H2B objects(Dict)
    Read a single trajectory file and return the dictionary.
    key is consist of filename@id and the value is H2B object.
    Dict only keeps the histones which have the trajectory length longer than cutoff value.
    """
    filetypes = ['trxyt', 'trx']

    # Check filetype.
    assert file.strip().split('.')[-1].lower() in filetypes
    # Read file and store the trajectory and time information in H2B object
    if file.strip().split('.')[-1].lower() in ['trxyt', 'trx']:
        try:
            trajectory_list = []
            with open(file, 'r', encoding="utf-8") as f:
                input = f.read()
            lines = input.strip().split('\n')
            nb_traj = 0
            old_index = -999
            for line in lines:
                temp = line.split('\t')
                index = int(float(temp[0].strip()))
                time = float(temp[3].strip())
                x_pos = float(temp[1].strip())
                y_pos = float(temp[2].strip())
                z_pos = 0.0
                if index != old_index:
                    nb_traj += 1
                    trajectory_list.append(TrajectoryObj(index=index, max_pause=5))
                    trajectory_list[nb_traj - 1].add_trajectory_position(time, x_pos, y_pos, z_pos)
                else:
                    trajectory_list[nb_traj - 1].add_trajectory_position(time, x_pos, y_pos, z_pos)
                old_index = index
            return trajectory_list
        except Exception as e:
            print(f"Unexpected error, check the file: {file}")
            print(e)


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
    if len(args) == 0:
        print('example of use:')
        print('python make_image.py filename.trxyt')
        exit(1)

    if '.trxyt' not in args[0] and '.trx' not in args[0]:
        print(args)
        print('input trajectory file extension must be .trxyt or .trx')
        exit(1)

    if len(args) > 1:
        trajectory_length_cutoff = int(args[1])  #minimum length of trajectory legnth to display 
    else:
        trajectory_length_cutoff = 0

    scale_factor = 2 # decide the resolution of image. (higher value produce better image). Max value must be lower than 3, if you don't have good RAMs.
    background_color = 'black'
    colormap = 'jet'  # matplotlib colormap
    thickness = 1  # thickness of line
    scale_bar_in_log10 = [-1.25, 0.25]   # linear color mapping of log10(diffusion coefficient) in range[-3, 0] micrometer^2/second, if log_diff_coef is < than -3, set it to the first color of cmap, if log_diff_coef is > than 0, set it to the last color of cmap
    margins = (1 * 10**scale_factor, 1 * 10**scale_factor)  # image margin in pixel

    mycmap = plt.get_cmap(colormap, lut=None)
    color_seq = [mycmap(i)[:3] for i in range(mycmap.N)][::-1]
    trajectory_list = read_trajectory(args[0])

    make_image(trajectory_list, f'{args[0].split(".trx")[0]}_diffusion.png', cutoff=trajectory_length_cutoff, margins=margins, scale_factor=scale_factor, color_seq=color_seq, thickness=thickness, scale_bar_in_log10=scale_bar_in_log10, background=background_color)

