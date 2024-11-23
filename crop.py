"""
crop image with ROI(csv file, start x, start y, end x, end y)
"""
import sys
import numpy as np
from module.FileIO import read_trajectory, write_trajectory

def crop_ROI(csv_file, start_x, start_y, end_x, end_y):
    filtered_trajectory_list = []
    trajectory_list = read_trajectory(csv_file)

    for trajectory in trajectory_list:
        xyz = trajectory.get_positions()
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]
        x_cond = np.sum((x >= start_x) * (x<=end_x)) == len(x)
        y_cond = np.sum((y >= start_y) * (y<=end_y)) == len(y)
        if x_cond and y_cond:
            filtered_trajectory_list.append(trajectory)
    write_trajectory(f'{".".join(csv_file.split('.')[:-1])}_cropped.csv', filtered_trajectory_list)


if __name__ == '__main__':
    print(sys.argv)
    if len(sys.argv) != 6:
        sys.exit('Example of command: python3 crop.py file_traces.csv start x start y end x end y')
    else:
        crop_ROI(sys.argv[1], float(eval(sys.argv[2])), float(eval(sys.argv[3])), float(eval(sys.argv[4])), float(eval(sys.argv[5])))
