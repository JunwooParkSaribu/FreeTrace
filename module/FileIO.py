import os
import sys
import numpy as np
from FreeTrace.module.TrajectoryObject import TrajectoryObj
from FreeTrace.module.ImageModule import read_tif


author_emails = [f'junwoo.park@sorbonne-universite.fr']


def read_trajectory(file: str, andi_gt=False, pixel_microns=1.0, frame_rate=1.0) -> dict | list:
    """
    @params : filename(String), cutoff value(Integer)
    @return : dictionary of H2B objects(Dict)
    Read a single trajectory file and return the dictionary.
    key is consist of filename@id and the value is H2B object.
    Dict only keeps the histones which have the trajectory length longer than cutoff value.
    """
    filetypes = ['trxyt', 'trx', 'csv']
    # Check filetype.
    assert file.strip().split('.')[-1].lower() in filetypes
    # Read file and store the trajectory and time information in H2B object
    if file.strip().split('.')[-1].lower() in ['trxyt', 'trx']:
        localizations = {}
        tmp = {}
        try:
            with open(file, 'r', encoding="utf-8") as f:
                input = f.read()
            lines = input.strip().split('\n')
            for line in lines:
                temp = line.split('\t')
                x_pos = float(temp[1].strip()) * pixel_microns
                y_pos = float(temp[2].strip()) * pixel_microns
                z_pos = 0. * pixel_microns
                time_step = float(temp[3].strip()) * frame_rate
                if time_step in tmp:
                    tmp[time_step].append([x_pos, y_pos, z_pos])
                else:
                    tmp[time_step] = [[x_pos, y_pos, z_pos]]

            time_steps = np.sort(np.array(list(tmp.keys())))
            first_frame, last_frame = time_steps[0], time_steps[-1]
            steps = np.arange(int(np.round(first_frame * 100)), int(np.round(last_frame * 100)) + 1)
            for step in steps:
                if step/100 in tmp:
                    localizations[step] = tmp[step/100]
                else:
                    localizations[step] = []
            return localizations
        except Exception as e:
            print(f"Unexpected error, check the file: {file}")
            print(e)
    else:
        try:
            trajectory_list = []
            with open(file, 'r', encoding="utf-8") as f:
                input = f.read()
            lines = input.strip().split('\n')
            nb_traj = 0
            old_index = -999
            for line in lines[1:]:
                temp = line.split(',')
                index = int(float(temp[0].strip()))
                frame = int(float(temp[1].strip()))
                x_pos = float(temp[2].strip())
                y_pos = float(temp[3].strip())
                if andi_gt:
                    x_pos = float(temp[3].strip())
                    y_pos = float(temp[2].strip())
                if len(temp) > 4:
                    z_pos = float(temp[4].strip())
                else:
                    z_pos = 0.0

                if index != old_index:
                    nb_traj += 1
                    trajectory_list.append(TrajectoryObj(index=index, max_pause=5))
                    trajectory_list[nb_traj - 1].add_trajectory_position(frame * frame_rate, x_pos * pixel_microns, y_pos * pixel_microns, z_pos * pixel_microns)
                else:
                    trajectory_list[nb_traj - 1].add_trajectory_position(frame * frame_rate, x_pos * pixel_microns, y_pos * pixel_microns, z_pos * pixel_microns)
                old_index = index
            return trajectory_list
        except Exception as e:
            print(f"Unexpected error, check the file: {file}")
            print(e)


def write_trajectory(file: str, trajectory_list: list):
    try:
        with open(file, 'w', encoding="utf-8") as f:
            input_str = 'traj_idx,frame,x,y,z\n'
            for trajectory_obj in trajectory_list:
                for (xpos, ypos, zpos), time in zip(trajectory_obj.get_positions(), trajectory_obj.get_times()):
                    input_str += f'{trajectory_obj.get_index()},{time},{xpos},{ypos},{zpos}\n'
            f.write(input_str)
    except Exception as e:
        print(f"Unexpected error, check the file: {file}")
        print(e)


def write_trxyt(file: str, trajectory_list: list, pixel_microns=1.0, frame_rate=1.0):
    try:
        with open(file, 'w', encoding="utf-8") as f:
            input_str = ''
            for index, trajectory_obj in enumerate(trajectory_list):
                for (xpos, ypos, zpos), time in zip(trajectory_obj.get_positions(), trajectory_obj.get_times()):
                    input_str += f'{index}\t{xpos * pixel_microns:.5f}\t{ypos * pixel_microns:.5f}\t{time * frame_rate:.3f}\n'
            f.write(input_str)
    except Exception as e:
        print(f"Unexpected error, check the file: {file}")
        print(e)


def write_localization(output_dir, coords, all_pdfs, infos):
    lines = f'frame,x,y,z,xvar,yvar,rho,norm_cst,intensity,window_size\n'
    for frame, (coord, pdfs, info) in enumerate(zip(coords, all_pdfs, infos)):
        for pos, (x_var, y_var, rho, amp), pdf in zip(coord, info, pdfs):
            window_size = int(np.sqrt(len(pdf)))
            peak_val = pdf[int((len(pdf) - 1) / 2)]
            lines += f'{frame + 1}'
            if len(pos) == 3:
                lines += f',{pos[1]},{pos[0]},{pos[2]}'
            elif len(pos) == 2:
                lines += f',{pos[1]},{pos[0]},0.0'
            elif len(pos) == 1:
                lines += f',{pos[0]},0.0,0.0'
            else:
                print(f'Localization writing Err')
                raise Exception
            lines += f',{x_var},{y_var},{rho},{amp},{peak_val},{window_size}'
            lines += f'\n'

    with open(f'{output_dir}_loc.csv', 'w') as f:
        f.write(lines)


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


def read_andi2_trajectory_label(input_file, index=None):
    trajectory = {}
    if type(input_file) is str:
        with open(input_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split('\n')[0].split(',')
                index = int(float(line[0]))
                line = line[1:]

                diff_coefs = []
                alphas = []
                state_nums = []
                cps = [0]
                cp_state = [False]
                turn_on = False
                for i, item in enumerate(line):
                    if i % 4 == 0:
                        turn_on = False
                        diff_coef = float(item)
                    elif i % 4 == 1:
                        alpha = float(item)
                    elif i % 4 == 2:
                        state_num = int(float(item))
                    elif i % 4 == 3:
                        cp = int(float(item))
                        turn_on = True
                    if turn_on:
                        cp_range = cp - cps[-1]

                        diff_coefs.extend([diff_coef] * cp_range)
                        alphas.extend([alpha] * cp_range)
                        state_nums.extend([state_num] * cp_range)
                        cps.extend([cp] * cp_range)
                        cp_state.extend([0] * (cp_range-1))
                        if i != len(line) - 1:
                            cp_state.append(1)
                        trajectory[index] = [np.array(diff_coefs), np.array(alphas), np.array(state_nums), np.array(cp_state)]
    else:
        trajectory = {}
        diff_coefs = []
        alphas = []
        state_nums = []
        cps = [0]
        cp_state = [False]
        turn_on = False
        if index is None:
            index = 0

        for traj_length, label_list in enumerate(np.array(input_file).T):
            for i, label in enumerate(label_list):
                if i % 4 == 0:
                    turn_on = False
                    diff_coef = float(label)
                elif i % 4 == 1:
                    alpha = float(label)
                elif i % 4 == 2:
                    state_num = int(float(label))
                elif i % 4 == 3:
                    cp = int(float(label))
                    turn_on = True
                if turn_on:
                    cp_range = cp - cps[-1]

                    diff_coefs.extend([diff_coef] * cp_range)
                    alphas.extend([alpha] * cp_range)
                    state_nums.extend([state_num] * cp_range)
                    cps.extend([cp] * cp_range)
                    cp_state.extend([0] * (cp_range-1))
                    if traj_length != len(input_file[0]) - 1:
                        cp_state.append(1)
                    trajectory[index] = [np.array(diff_coefs), np.array(alphas), np.array(state_nums), np.array(cp_state)]
    return trajectory


def read_parameters(param_file):
    params = {'localization': {}, 'tracking': {}}
    try:
        with open(param_file, 'r', encoding="utf-8") as f:
            input = f.read()
        lines = input.strip().split('\n')

        for line in lines:
            if 'video' in line.lower():
                params['localization']['VIDEO'] = line.strip().split('=')[1]
            if 'output_dir' in line.lower():
                params['localization']['OUTPUT_DIR'] = line.strip().split('=')[1]
            if 'sigma' in line.lower():
                params['localization']['SIGMA'] = float(eval(line.strip().split('=')[1]))
            if 'window_size' in line.lower():
                params['localization']['WINSIZE'] = int(eval(line.strip().split('=')[1]))
            if 'threshold_alpha' in line.lower():
                params['localization']['THRES_ALPHA'] = float(eval(line.strip().split('=')[1]))
            if 'deflation_loop_in_backward' in line.lower():
                params['localization']['DEFLATION_LOOP_IN_BACKWARD'] = int(eval(line.strip().split('=')[1]))
            if 'shift' in line.lower():
                params['localization']['SHIFT'] = int(eval(line.strip().split('=')[1]))
            if 'loc_visualization' in line.lower():
                if 'true' in line.lower().strip().split('=')[1]:
                    params['localization']['LOC_VISUALIZATION'] = True
                else:
                    params['localization']['LOC_VISUALIZATION'] = False
            if 'gpu_loc' in line.lower():
                if 'true' in line.lower().strip().split('=')[1]:
                    params['localization']['GPU'] = True
                else:
                    params['localization']['GPU'] = False

            if 'video' in line.lower():
                params['tracking']['VIDEO'] = line.strip().split('=')[1]
            if 'output_dir' in line.lower():
                params['tracking']['OUTPUT_DIR'] = line.strip().split('=')[1]
            if 'pixel_microns' in line.lower():
                params['tracking']['PIXEL_MICRONS'] = float(eval(line.strip().split('=')[1]))
            if 'frame_per_sec' in line.lower():
                params['tracking']['FRAME_PER_SEC'] = float(eval(line.strip().split('=')[1]))
            if 'blink_lag' in line.lower():
                params['tracking']['BLINK_LAG'] = int(eval(line.strip().split('=')[1]))
            if 'cutoff' in line.lower():
                params['tracking']['CUTOFF'] = int(eval(line.strip().split('=')[1]))
            if 'track_visualization' in line.lower():
                if 'true' in line.lower().strip().split('=')[1]:
                    params['tracking']['TRACK_VISUALIZATION'] = True
                else:
                    params['tracking']['TRACK_VISUALIZATION'] = False
            if 'gpu_track' in line.lower():
                if 'true' in line.lower().strip().split('=')[1]:
                    params['tracking']['GPU'] = True
                else:
                    params['tracking']['GPU'] = False
    except Exception as e:
        print(f"Unexpected error, check the config file")
        sys.exit(f'ERR msg: {e}')
    finally:
        return params


def check_video_ext(args, andi2=False):
    if len(args) == 0:
        print(f'no input file')
        exit(1)
    if '.tif' not in args and '.tiff' not in args:
        print(f'video format err, only .tif or .tiff are acceptable')
        exit(1)
    else:
        return read_tif(args, andi2)
    

def initialization(gpu, reg_model_nums=[], ptype=-1, verbose=False, batch=False):
    TF = False
    cuda = False

    if not os.path.exists(f'{__file__.split("FreeTrace")[0]}/FreeTrace/models/theta_hat.npz'):
        print(f'\n***** Parmeters[theta_hat.npz] are not found for trajectory inference, please contact author for the pretrained models. *****\n')
        print(f'***********  Contacts  ***********')
        for author in author_emails:
            print(author)
        sys.exit('**********************************\n')

    if gpu:
        try:
            import cupy as cp
            if cp.cuda.is_available():
                cuda = True
            else:
                cuda = False
        except:
            cuda = False
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if len(gpus) > 0:
                TF = True
            else:
                TF = False
        except:
            TF = False

    if TF and ptype==1:
        for reg_model_num in reg_model_nums:
            if not os.path.exists(f'{__file__.split("FreeTrace")[0]}/FreeTrace/models/reg_model_{reg_model_num}.keras'):
                print(f'\n***** reg_model_{reg_model_num}.keras is not found, contact author for the pretrained models. *****')
                print(f'***********  Contacts  ***********')
                for author in author_emails:
                    print(author)
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
