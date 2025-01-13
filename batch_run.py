import os
import sys
import subprocess
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
SIGMA = params['localization']['SIGMA']
SHIFT = params['localization']['SHIFT']
LOC_VISUALIZATION = params['localization']['LOC_VISUALIZATION']
LOC_GPU_AVAIL = params['localization']['GPU']

BLINK_LAG = params['tracking']['BLINK_LAG']
CUTOFF = params['tracking']['CUTOFF']
TRACK_VISUALIZATION = params['tracking']['TRACK_VISUALIZATION']
PIXEL_MICRONS = params['tracking']['PIXEL_MICRONS']
FRAME_RATE = params['tracking']['FRAME_PER_SEC']
TRACK_GPU_AVAIL = params['tracking']['GPU']


def run_command(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process


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
                    loc = Localization.run_process(input_video=f'{batch_folder}/{file}', outpur_dir=OUTPUT_DIR,
                                                window_size=WINSIZE, threshold=THRES_ALPHA,
                                                deflation=DEFLATION_LOOP_IN_BACKWARD, sigma=SIGMA, shift=SHIFT,
                                                gpu_on=LOC_GPU_AVAIL, save_video=True, verbose=0, batch=True)
                    PBAR.update(1)
                    if loc:
                        track = Tracking.run_process(input_video=f'{batch_folder}/{file}', outpur_dir=OUTPUT_DIR,
                                                    blink_lag=BLINK_LAG, cutoff=CUTOFF,
                                                    pixel_microns=PIXEL_MICRONS, frame_rate=FRAME_RATE,
                                                    gpu_on=TRACK_GPU_AVAIL, save_video=TRACK_VISUALIZATION, verbose=0, batch=True)
                    PBAR.update(1)

                    if os.path.exists('diffusion_image.py') and track:
                        proc = run_command([sys.executable.split('/')[-1], f'diffusion_image.py', f'{OUTPUT_DIR}/{file.strip().split(".tif")[0]}_traces.csv', str(PIXEL_MICRONS), str(FRAME_RATE)])
                        proc.wait()
                        if not proc.poll() == 0:
                            print(f'diffusion map -> failed with status:{proc.poll()}')
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
