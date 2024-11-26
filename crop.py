import sys
import os
import numpy as np
from module.FileIO import read_trajectory, write_trajectory


"""
crop image with 
1. ROI(start x, start y, end x, end y)
2. Frames(start frame, end frame)
"""
def crop_ROI_and_frame(csv_file, roi_input, frame_input):
    start_x, start_y, end_x, end_y = roi_input
    start_frame, end_frame = frame_input
    filtered_trajectory_list = []
    trajectory_list = read_trajectory(csv_file)

    for trajectory in trajectory_list:
        xyz = trajectory.get_positions()
        times = trajectory.get_times()
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]
        if start_x is not None:
            x_cond = np.sum((x >= start_x) * (x<=end_x)) == len(x)
            y_cond = np.sum((y >= start_y) * (y<=end_y)) == len(y)
        if start_x is not None and start_frame is not None:
            if x_cond and y_cond and times[0] >= start_frame and times[-1] <= end_frame:
                filtered_trajectory_list.append(trajectory)
        elif start_x is not None and start_frame is None:
            if x_cond and y_cond:
                filtered_trajectory_list.append(trajectory)
        elif start_x is None and start_frame is not None:
            if times[0] >= start_frame and times[-1] <= end_frame:
                filtered_trajectory_list.append(trajectory)
        else:
            filtered_trajectory_list.append(trajectory)

    print(f'cropping info: ROI[{start_x}, {start_y}, {end_x}, {end_y}],  Frame:[{start_frame}, {end_frame}]')
    print(f'Number of trajectories before filtering:{len(trajectory_list)}, after filtering:{len(filtered_trajectory_list)}')
    write_trajectory(f'{".".join(csv_file.split("traces.csv")[:-1])}cropped_traces.csv', filtered_trajectory_list)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('Example of command: python3 crop.py video_traces.csv')
    csv_file = sys.argv[1].strip()
    if not os.path.exists(csv_file):
        sys.exit(f'Input file: {csv_file} is not exist. please check again')
    print(f"input file name: {csv_file}")

    while True:
        roi_input =input(f'Crop with ROI in pixel: start x, start y, end x, end y\nexample of use: 10,10,25,30\nyour ROI input: ')
        roi_input = roi_input.strip().split(',')
        if len(roi_input) == 1 and roi_input[0] == '':
            print('*** skip crop with ROI ***')
            roi_input = [None, None, None, None]
            break
        if len(roi_input) != 4:
            print('input is wrong, please enter 4 integers in pixel.\n')
            continue
        for i in range(4):
            roi_input[i] = int(eval(roi_input[i]))
        if roi_input[0] >= roi_input[2] or roi_input[1] >= roi_input[3]:
            print('input is wrong, end x or end y must be greater than start x or start y.\n')
            continue
        break
    print('')
    while True:
        frame_input =input(f'Crop with frame: start frame, end frame\nexample of use: 100,500\nyour frame input: ')
        frame_input = frame_input.strip().split(',')
        if len(frame_input) == 1 and frame_input[0] == '':
            print('*** skip crop with frames ***')
            frame_input = [None, None]
            break
        if len(frame_input) != 2:
            print('input is wrong, please enter 2 frame numbers.\n')
            continue
        for i in range(2):
            frame_input[i] = int(eval(frame_input[i]))
        if frame_input[0] >= frame_input[1]:
            print('input is wrong, end frame must be greater than start frame.\n')
            continue
        break

    crop_ROI_and_frame(csv_file, roi_input, frame_input)
