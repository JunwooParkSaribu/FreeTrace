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



"""
Basic parameters.
"""
WINDOW_SIZE = 7  # Size of sliding window for particle localisation. 
THRESHOLD = 1.0  # Detection threshold of particle localisation.
CUTOFF = 3  # Minimum length (frame) of trajectory output.



"""
Advanced parameters.
"""
GPU_FOR_LOCALIZATION = True  # GPU acceleration with CUDA. (only available with NVIDIA GPU)
REALTIME_LOCALIZATION = False  # If you set this option as True, the computation will be slower.
SAVE_LOCALIZATION_VIDEO = False

FBM_MODE = True  # Inference under fBm, if True (slow). Otherwise, classical Brownian motion if False (fast).
JUMP_THRESHOLD = None  # Maximum jump-distance of particles.
GRAPH_DEPTH = 3  # Delta T.
REALTIME_TRACKING = False  # If you set this option as True, the computation will be slower.
SAVE_TRACKING_VIDEO = False



if __name__ == "__main__":
    if not os.path.isdir(input_folder):
        sys.exit(f'{input_folder} is not a directory containing files')
    else:
        failed_tasks = []
        if not os.path.exists(f'{OUTPUT_DIR}'):
            os.makedirs(f'{OUTPUT_DIR}')
        file_list = os.listdir(f'{input_folder}')
        nb_videos_in_batch = len([file for file in file_list if '.tif' in file])

        print(f'\n*****  Batch processing on {len(file_list)} videos. ({len(file_list)*2} tasks: Localizations + Trackings)  *****')
        PBAR = tqdm(total=nb_videos_in_batch*2, desc="Batch", unit="task", ncols=120, miniters=1)
        for idx in range(len(file_list)):
            loc = False
            track = False
            file = file_list[idx]
            if file.strip().split('.')[-1] == 'tif' or file.strip().split('.')[-1] == 'tiff':
                PBAR.set_postfix(File=file, refresh=True)
                try:
                    loc = Localization.run_process(input_video_path=f'{input_folder}/{file}', output_path=OUTPUT_DIR,
                                                   window_size=WINDOW_SIZE,
                                                   threshold=THRESHOLD,
                                                   gpu_on=GPU_FOR_LOCALIZATION,
                                                   save_video=SAVE_LOCALIZATION_VIDEO,
                                                   realtime_visualization=REALTIME_LOCALIZATION,
                                                   verbose=0,
                                                   batch=True)
                    PBAR.update(1)
                    if loc:
                        track = Tracking.run_process(input_video_path=f'{input_folder}/{file}', output_path=OUTPUT_DIR,
                                                     graph_depth=GRAPH_DEPTH,
                                                     cutoff=CUTOFF,
                                                     jump_threshold=JUMP_THRESHOLD,
                                                     gpu_on=FBM_MODE,
                                                     save_video=SAVE_TRACKING_VIDEO,
                                                     realtime_visualization=REALTIME_TRACKING,
                                                     verbose=0,
                                                     batch=True)
                    PBAR.update(1)

                except Exception as e:
                    failed_tasks.append(file)
                    print(f"ERROR on {file}, code:{e}")
                    with open(f'{OUTPUT_DIR}/error_log.txt', 'a') as error_log:
                        dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                        input_str = f'{file} has an err[{e}]. DATE: {dt_string}\n'        
                        error_log.write(input_str)
        PBAR.close()
        if len(failed_tasks) > 0:
            print(f'Prediction failed on {failed_tasks}, please check error_log file.')
        else:
            print('Batch prediction finished succesfully.')
