import cv2
import sys
import os
import numpy as np
from roifile import ImagejRoi
from FreeTrace.module.data_save import write_trajectory
from FreeTrace.module.data_load import read_trajectory
from FreeTrace.module.image_module import make_whole_img


def initialization(gpu, reg_model_nums=[], ptype=-1, verbose=False, batch=False):
    TF = False
    cuda = False
    freetrace_path = ""
    for it in __file__.split("FreeTrace")[:-1]:
        freetrace_path += it
        freetrace_path += 'FreeTrace/'

    if not os.path.exists(f'{freetrace_path}/models/theta_hat.npz'):
        print(f'\n***** Parmeters[theta_hat.npz] are not found for trajectory inference, please contact author for the pretrained models. *****\n')
        sys.exit('**********************************\n')

    if gpu:
        try:
            import cupy as cp
            if cp.cuda.is_available():
                cuda = True
            else:
                cuda = False
            del cp
        except:
            cuda = False
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if len(gpus) > 0:
                TF = True
            else:
                TF = False
            del gpus
        except:
            TF = False

    if TF and ptype==1:
        for reg_model_num in reg_model_nums:
            if not os.path.exists(f'{freetrace_path}/models/reg_model_{reg_model_num}.keras'):
                print(f'\n***** reg_model_{reg_model_num}.keras is not found, contact author for the pretrained models. *****')
                print(f'***********  Contacts  ***********')
                sys.exit('**********************************\n')

    if not batch and verbose:
        track_ = True if ptype==1 else False
        print(f'\n******************************** OPTIONS *****************************************')
        if cuda and TF:
            if track_:
                print(f'***** Cuda: Ok, Tensorflow: Ok, Tracking performs slow/complete inferences. ******')
            else:
                print(f'**** Cuda: Ok, Tensorflow: Ok, Localization performs fast/complete inferences. ***')

        elif cuda and not TF:
            if track_:
                print(f'***** Cuda: Ok, Tensorflow: X, Tracking performs fast/incomplete inferences. ******')
            else:
                print(f'**** Cuda: Ok, Tensorflow: X, Localization performs fast/complete inferences. ****')

        elif not cuda and TF:
            if track_:
                print(f'***** Cuda: X, Tensorflow: Ok, Tracking performs slow/complete inferences. ******')
            else:
                print(f'**** Cuda: X, Tensorflow: Ok, Localization performs slow/complete inferences. ****')
                
        else:
            if track_:
                print(f'***** Cuda: X, Tensorflow: X, Tracking performs fast/incomplete inferences. ******') 
            else:
                print(f'***** Cuda: X, Tensorflow: X, Localization performs slow/complete inferences. ****') 
        print(f'**********************************************************************************\n')
        
    if batch and verbose:
        print(f'\n******************************** OPTIONS *****************************************')
        if cuda and TF:
            print(f'***** Cuda: Ok, Tensorflow: Ok, FreeTrace performs fast/complete inferences. *****')

        elif cuda and not TF:
            print(f'****** Cuda: Ok, Tensorflow: X, FreeTrace performs fast/complete inferences. *****')

        elif not cuda and TF:
            print(f'****** Cuda: X, Tensorflow: Ok, FreeTrace performs slow/complete inferences. *****')
                
        else:
            print(f'**** Cuda: X, Tensorflow: X, FreeTrace performs slow/complete inferences. ****') 
        print(f'**********************************************************************************\n')
    return cuda, TF


def crop_trace_roi_and_frame(trace_file:str, roi_file:str|None, start_frame:0, end_frame:9999999, option=0, crop_comparison=False):
    """
    Cropping trajectory result with ROI(region of interest) or frames.
    trace_file: video_traces.csv or equivalent format of trajectory result file.
    roi_file: region_of_interest.roi which containes the roi information in pixel.
    start_frame: start frame to crop the trajectory result with frame.
    end_frame: end frame to crop the trajectory result with frame.
    option: 0 -> considers the trajectories stay only inside the ROI. 1 -> incldues all the trajectories passing trough the ROI.
    crop_comparison: boolean to visualize cropped result.
    """

    assert "traces.csv" in trace_file, "Wrong trajectory file format, result_traces.csv is needed to crop with ROI or frames"
    assert end_frame > start_frame, "The number of end frame must be greater than start frame."
    assert option == 0 or option == 1, "The option must be 0 or 1. 0: consider the trajectories stay only inside the ROI. 1: incldues all the trajectories passed the ROI."

    if roi_file is None:
        contours = None
    elif type(roi_file) is str and len(roi_file) == 0:
        contours = None
    else:
        contours = ImagejRoi.fromfile(roi_file).coordinates().astype(np.int32)

    filtered_trajectory_list = []
    trajectory_list = read_trajectory(trace_file)
    xmin = 999999
    xmax = -1
    ymin = 999999
    ymax = -1
    start_frame = max(start_frame, 0)
    end_frame = min(end_frame, 9999999)

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

        if option == 0:
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
        else:
            if contours is not None:
                for x, y in zip(xs, ys):
                    masked = cv2.pointPolygonTest(contours, (x, y), False)
                    if masked == 1:
                        if times[0] >= start_frame and times[-1] <= end_frame:
                            filtered_trajectory_list.append(trajectory)
                        break

    print(f'cropping info: ROI[{roi_file}],  Frame:[{start_frame}, {end_frame}]')
    print(f'Number of trajectories before filtering:{len(trajectory_list)}, after filtering:{len(filtered_trajectory_list)}')
    write_trajectory(f'{".".join(trace_file.split("traces.csv")[:-1])}cropped_{start_frame}_{end_frame}_traces.csv', filtered_trajectory_list)
    print(f'{".".join(trace_file.split("traces.csv")[:-1])}cropped_{start_frame}_{end_frame}_traces.csv is successfully generated.')

    if crop_comparison:
        dummy_stack = np.empty((1, int(ymax+1), int(xmax+1), 1))
        make_whole_img(trajectory_list, output_dir=f'{".".join(trace_file.split("traces.csv")[:-1])}before_crop.png', img_stacks=dummy_stack)
        make_whole_img(filtered_trajectory_list, output_dir=f'{".".join(trace_file.split("traces.csv")[:-1])}after_crop.png', img_stacks=dummy_stack)
        print(f'{".".join(trace_file.split("traces.csv")[:-1])}before_crop.png is successfully generated.')
        print(f'{".".join(trace_file.split("traces.csv")[:-1])}after_crop.png is successfully generated.')


def calibration_3d(xyz_coords, reg_infos, calib_data):
    """
    3D calibration for astigmatism.
    under development.
    """
    for t in range(len(xyz_coords)):
        for particle_idx in range(len(xyz_coords[t])):
            xvar = reg_infos[t][particle_idx][0]
            yvar = reg_infos[t][particle_idx][1]
            z_coord = (yvar - xvar)
            xyz_coords[t][particle_idx][2] = z_coord
    return xyz_coords
