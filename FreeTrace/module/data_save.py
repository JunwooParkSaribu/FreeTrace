import os
import csv
import numpy as np


def save_report(data: list, path='', all=False) -> list:
    """
    @params : data(list), path(String), all(boolean)
    @return : list of saved report names(list)
    Save reports in a given path.
    """
    histones = {}
    report_names = []

    for chunked_data in data:
        histones |= chunked_data

    if not all:
        histone_names = list(histones.keys())
        filenames = set()
        for histone in histone_names:
            filenames.add(histone.split('\\')[-1].split('@')[0])

        for filename in filenames:
            h = {}
            for histone in histone_names:
                if filename in histone:
                    h[histone] = histones[histone]
            write_file_name = f'{path}/{filename}.csv'
            with open(write_file_name, 'w', newline='') as f:
                report_names.append(write_file_name)
                fieldnames = ['filename', 'h2b_id', 'predicted_class_id', 'predicted_class_name', 'probability',
                              'maximum_radius', 'first_x_position', 'first_y_position']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for i, key in enumerate(h):
                    trajectory = histones[key].get_trajectory()
                    first_x_pos = trajectory[0][0]
                    first_y_pos = trajectory[0][1]
                    file_name = histones[key].get_file_name()
                    h2b_id = histones[key].get_id()
                    pred_class_id = histones[key].get_predicted_label()
                    max_r = histones[key].get_max_radius()
                    proba = histones[key].get_predicted_proba()

                    pred_class_name = 'unidentified'
                    if pred_class_id == 0:
                        pred_class_name = 'Immobile'
                    if pred_class_id == 1:
                        pred_class_name = 'Hybrid'
                    if pred_class_id == 2:
                        pred_class_name = 'Mobile'

                    writer.writerow({'filename':file_name, 'h2b_id':h2b_id, 'predicted_class_id':pred_class_id,
                                     'predicted_class_name':pred_class_name, 'probability':proba, 'maximum_radius':max_r,
                                     'first_x_position':first_x_pos, 'first_y_position':first_y_pos})
    else:
        write_file_name = f'{path}/prediction_all.csv'
        with open(write_file_name, 'w', newline='') as f:
            report_names.append(write_file_name)
            fieldnames = ['filename', 'h2b_id', 'predicted_class_id', 'predicted_class_name', 'probability',
                          'maximum_radius', 'first_x_position', 'first_y_position']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for i, key in enumerate(histones):
                trajectory = histones[key].get_trajectory()
                first_x_pos = trajectory[0][0]
                first_y_pos = trajectory[0][1]
                file_name = histones[key].get_file_name()
                h2b_id = histones[key].get_id()
                pred_class_id = histones[key].get_predicted_label()
                max_r = histones[key].get_max_radius()
                proba = histones[key].get_predicted_proba()

                pred_class_name = 'unidentified'
                if pred_class_id == 0:
                    pred_class_name = 'Immobile'
                if pred_class_id == 1:
                    pred_class_name = 'Hybrid'
                if pred_class_id == 2:
                    pred_class_name = 'Mobile'

                writer.writerow({'filename': file_name, 'h2b_id': h2b_id, 'predicted_class_id': pred_class_id,
                                 'predicted_class_name': pred_class_name, 'probability': proba, 'maximum_radius': max_r,
                                 'first_x_position': first_x_pos, 'first_y_position': first_y_pos})
    return report_names


def write_model_info(training_model, path: str, history: list, nb_histones: int, date: str) -> str:
    """
    @params : model(tensorflow model object), path(String), history(list), nb_histones(Integer), data(String)
    @return : saved model name(String)
    Save the logs and the history of the model and return the model name.
    """
    new_model_num = 0
    try:
        if os.path.isdir(path):
            contents = os.listdir(path)
            for content in contents:
                if 'model' in content:
                    model_num = int(content.split('_')[0].split('model')[-1])
                    new_model_num = max(new_model_num, model_num)
            modelname = f'model{new_model_num + 1}'
        training_model.save(f'{path}/{modelname}')
    except Exception as e:
        print('Model directory creation err')
        print(e)

    with open(f'{path}/{modelname}/log.txt', 'w') as info_file:
        info_file.write(f'{date}, number of trained h2bs:{str(nb_histones)}\n')
        info_file.write(f'train history, test history, train_acc, test_acc\n')
        for line_num in range(len(history[0])):
            info_file.write(f'{str(history[0][line_num])}\t{str(history[1][line_num])}\t'
                            f'{str(history[2][line_num])}\t{str(history[3][line_num])}\n')
    return modelname


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
