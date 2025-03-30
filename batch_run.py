"""
Run FreeTrace on multiple .tiff files. 
"""

import os
import sys
from tqdm import tqdm
from datetime import datetime
from FreeTrace import Tracking, Localization


input_folder = 'inputs'
OUTPUT_DIR = 'outputs'


WINSIZE = 7
THRESHOLD = 1.0
SAVE_VIDEO_LOC = False
REAL_LOC = True
LOC_GPU_AVAIL = True


TIME_FORECAST = 2
CUTOFF = 2
JUMP_THRESHOLD = None
SAVE_VIDEO_TRACK = False
REAL_TRACK = True
TRACK_GPU_AVAIL = True


if __name__ == "__main__":
    if not os.path.isdir(input_folder):
        sys.exit(f'{input_folder} is not a directory containing files')
    else:
        failed_tasks = []
        if not os.path.exists(f'{OUTPUT_DIR}'):
            os.makedirs(f'{OUTPUT_DIR}')
        file_list = os.listdir(f'{input_folder}')
        print(f'\n*****  Batch processing on {len(file_list)} files. ({len(file_list)*2} tasks: Localizations + Trackings)  *****')
        PBAR = tqdm(total=len(file_list)*2, desc="Batch", unit="task", ncols=120, miniters=1)
        for idx in range(len(file_list)):
            loc = False
            track = False
            file = file_list[idx]
            if file.strip().split('.')[-1] == 'tif' or file.strip().split('.')[-1] == 'tiff':
                PBAR.set_postfix(File=file, refresh=True)
                try:
                    loc = Localization.run_process(input_video_path=f'{input_folder}/{file}', output_path=OUTPUT_DIR,
                                                   window_size=WINSIZE, threshold=THRESHOLD,
                                                   gpu_on=LOC_GPU_AVAIL, save_video=SAVE_VIDEO_LOC,
                                                   realtime_visualization=REAL_LOC, verbose=0, batch=True)
                    PBAR.update(1)
                    if loc:
                        track = Tracking.run_process(input_video_path=f'{input_folder}/{file}', output_path=OUTPUT_DIR,
                                                     time_forecast=TIME_FORECAST, cutoff=CUTOFF, jump_threshold=JUMP_THRESHOLD,
                                                     gpu_on=TRACK_GPU_AVAIL, save_video=SAVE_VIDEO_TRACK, 
                                                     realtime_visualization=REAL_TRACK, verbose=0, batch=True)
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
