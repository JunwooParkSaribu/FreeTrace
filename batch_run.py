import os
import sys
from datetime import datetime
from tqdm import tqdm
from FreeTrace import Tracking, Localization
from FreeTrace.module.FileIO import initialization, read_parameters


"""
Read configuration file.
"""
batch_folder = 'inputs'
params = read_parameters('./config.txt')
OUTPUT_DIR = params['localization']['OUTPUT_DIR']


WINSIZE = params['localization']['WINSIZE']
THRES_ALPHA = params['localization']['THRES_ALPHA']
DEFLATION_LOOP_IN_BACKWARD = params['localization']['DEFLATION_LOOP_IN_BACKWARD']
SHIFT = params['localization']['SHIFT']
SAVE_VIDEO_LOC = params['localization']['LOC_VISUALIZATION']
LOC_GPU_AVAIL = params['localization']['GPU']


TIME_FORECAST = params['tracking']['TIME_FORECAST']
CUTOFF = params['tracking']['CUTOFF']
SAVE_VIDEO_TRACK = params['tracking']['TRACK_VISUALIZATION']
TRACK_GPU_AVAIL = params['tracking']['GPU']


if __name__ == "__main__":
    if not os.path.isdir(batch_folder):
        sys.exit(f'{batch_folder} is not a directory containing files')
    else:
        failed_tasks = []
        if not os.path.exists(f'{OUTPUT_DIR}'):
            os.makedirs(f'{OUTPUT_DIR}')
        file_list = os.listdir(f'{batch_folder}')
        print(f'\n*****  Batch processing on {len(file_list)} files. ({len(file_list)*2} tasks: Localizations + Trackings)  *****')
        initialization(gpu=True, verbose=True, batch=True)
        PBAR = tqdm(total=len(file_list)*2, desc="Batch", unit="task", ncols=120, miniters=1)
        for idx in range(len(file_list)):
            loc = False
            track = False
            file = file_list[idx]
            if file.strip().split('.')[-1] == 'tif' or file.strip().split('.')[-1] == 'tiff':
                PBAR.set_postfix(File=file, refresh=True)
                try:
                    loc = Localization.run_process(input_video_path=f'{batch_folder}/{file}', output_path=OUTPUT_DIR,
                                                   window_size=WINSIZE, threshold=THRES_ALPHA,
                                                   deflation=DEFLATION_LOOP_IN_BACKWARD, shift=SHIFT,
                                                   gpu_on=LOC_GPU_AVAIL, save_video=SAVE_VIDEO_LOC, verbose=0, batch=True)
                    PBAR.update(1)
                    if loc:
                        track = Tracking.run_process(input_video_path=f'{batch_folder}/{file}', output_path=OUTPUT_DIR,
                                                     time_forecast=TIME_FORECAST, cutoff=CUTOFF,
                                                     gpu_on=TRACK_GPU_AVAIL, save_video=SAVE_VIDEO_TRACK, verbose=0, batch=True)
                    PBAR.update(1)

                except Exception as e:
                    failed_tasks.append(file)
                    print(f"ERROR on {file}, code:{e}")
                    with open('./outputs/error_log.txt', 'a') as error_log:
                        dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                        input_str = f'{file} has an err[{e}]. DATE: {dt_string}\n'        
                        error_log.write(input_str)
        PBAR.close()
        if len(failed_tasks) > 0:
            print(f'Prediction failed on {failed_tasks}, please check error_log file.')
        else:
            print('Batch prediction finished succesfully.')
