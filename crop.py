import sys
import os
import numpy as np
import cv2
from roifile import ImagejRoi
from FreeTrace.module.FileIO import read_trajectory, write_trajectory
from FreeTrace.module.ImageModule import make_whole_img


"""
crop image with 
1. ROI (myroi.roi)
2. Frames (start frame, end frame)
"""
def crop_ROI_and_frame(csv_file, contours, start_frame, end_frame, crop_comparison):
    filtered_trajectory_list = []
    trajectory_list = read_trajectory(csv_file)
    xmin = 999999
    xmax = -1
    ymin = 999999
    ymax = -1

    for trajectory in trajectory_list:
        skip = 0
        xyz = trajectory.get_positions()
        times = trajectory.get_times()
        xs = xyz[:, 0]
        ys = xyz[:, 1]
        zs = xyz[:, 2]
        xmin = min(np.min(xs), xmin)
        ymin = min(np.min(ys), ymin)
        xmax = max(np.max(xs), xmax)
        ymax = max(np.max(ys), ymax)

        if contours is not None:
            for x, y in zip(xs, ys):
                masked = cv2.pointPolygonTest(contours, (x, y), False)
                if masked == -1:
                    skip = 1
                    break
            
            if skip == 1:
                continue

        if times[0] >= start_frame and times[-1] <= end_frame:
            filtered_trajectory_list.append(trajectory)

    print(f'cropping info: ROI[{ROI_FILE}],  Frame:[{start_frame}, {end_frame}]')
    print(f'Number of trajectories before filtering:{len(trajectory_list)}, after filtering:{len(filtered_trajectory_list)}')
    write_trajectory(f'{".".join(csv_file.split("traces.csv")[:-1])}cropped_traces.csv', filtered_trajectory_list)

    if crop_comparison:
        print("*** Crop comparison image generated ***")
        dummy_stack = np.empty((1, int(ymax+1), int(xmax+1), 1))
        make_whole_img(trajectory_list, output_dir=f'{".".join(csv_file.split("traces.csv")[:-1])}before_crop.png', img_stacks=dummy_stack)
        make_whole_img(filtered_trajectory_list, output_dir=f'{".".join(csv_file.split("traces.csv")[:-1])}after_crop.png', img_stacks=dummy_stack)


if __name__ == '__main__':
    if not(len(sys.argv) == 3 or len(sys.argv) == 5 or len(sys.argv) == 6):
        print('1. Example of command only with ROI: python3 crop.py video_traces.csv roi_file.roi')
        print('2. Example of command only with frames: python3 crop.py video_traces.csv None start_frame end_frame')
        sys.exit('3. Example of command with ROI and frames: python3 crop.py video_traces.csv roi_file.roi start_frame end_frame')
    
    csv_file = sys.argv[1].strip()  
    roi_filename = sys.argv[2].strip()
    if len(sys.argv) == 3:
        start_frame = -1
        end_frame = 9999999
        crop_comparison = False
    elif len(sys.argv) == 5:
        start_frame = int(eval(sys.argv[3].strip()))
        end_frame = int(eval(sys.argv[4].strip()))
        crop_comparison = False
    else:
        start_frame = int(eval(sys.argv[3].strip()))
        end_frame = int(eval(sys.argv[4].strip()))
        crop_comparison = True

    if roi_filename.lower() != 'none':
        assert os.path.exists(roi_filename), f'{roi_filename} is not found... check again the ROI name.'
        global ROI_FILE
        ROI_FILE = roi_filename
        contours = ImagejRoi.fromfile(ROI_FILE).coordinates().astype(np.int32)
    else:
        contours = None
    assert os.path.exists(csv_file), f'{csv_file} is not found... check again the csv name. The format should be same as trajectory result file of FreeTrace.'

    print(f"input file name: {csv_file}")
    crop_ROI_and_frame(csv_file, contours, start_frame, end_frame, crop_comparison)
