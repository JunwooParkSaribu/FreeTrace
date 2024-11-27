import os
import sys
import subprocess
from datetime import datetime
from tqdm import tqdm
from module.FileIO import initialization

"""
Configuration file settings for the parameters of FreeTrace.
"""

WINDOW_SIZE = 9
THRESHOLD_ALPHA = 1.0
GPU_LOC = True

CUTOFF = 2
PIXEL_MICRONS = 0.16
FRAME_RATE = 0.01
GPU_TRACK = True


def run_command(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process


def write_config(filename):
    content = \
    f"\
    VIDEO=./inputs/{filename}\n\
    OUTPUT_DIR=./outputs\n\
    \n\
    \n\
    # LOCALIZATION\n\
    WINDOW_SIZE = {WINDOW_SIZE}\n\
    THRESHOLD_ALPHA = {THRESHOLD_ALPHA}\n\
    DEFLATION_LOOP_IN_BACKWARD = 0\n\
    SIGMA = 4.0\n\
    LOC_VISUALIZATION = False\n\
    GPU_LOC = {GPU_LOC}\n\
    \n\
    \n\
    # TRACKING\n\
    CUTOFF = {CUTOFF}\n\
    BLINK_LAG = 2\n\
    PIXEL_MICRONS = {PIXEL_MICRONS}\n\
    FRAME_PER_SEC = {FRAME_RATE}\n\
    TRACK_VISUALIZATION = False\n\
    GPU_TRACK = {GPU_TRACK}\n\
    \n\
    \n\
    # SUPP\n\
    SHIFT = 1\n\
    "
    with open("./config.txt", 'w') as config:
        config.write(content)


failed_tasks = []
if not os.path.exists('./outputs'):
    os.makedirs('./outputs')
file_list = os.listdir('./inputs')
print(f'\n*****  Batch processing on {len(file_list)} files. ({len(file_list)*2} tasks: Localizations + Trackings)  *****')
initialization(gpu=True, verbose=True, batch=True)
PBAR = tqdm(total=len(file_list)*2, desc="Batch", unit="task", ncols=120, miniters=1)


for idx in range(len(file_list)):
    file = file_list[idx]
    if file.strip().split('.')[-1] == 'tif' or file.strip().split('.')[-1] == 'tiff':
        write_config(file)
        PBAR.set_postfix(File=file, refresh=True)
        try:
            pid = subprocess.run([sys.executable, 'Localization.py', '0' ,'1'], capture_output=True)
            if pid.returncode != 0:
                raise Exception(pid)
            PBAR.update(1)
            pid = subprocess.run([sys.executable, 'Tracking.py', '0', '1'], capture_output=True)
            if pid.returncode != 0:
                raise Exception(pid)
            PBAR.update(1)
            if os.path.exists('diffusion_image.py') and pid.returncode==0:
                proc = run_command([sys.executable.split('/')[-1], f'diffusion_image.py', f'./outputs/{file.strip().split(".tif")[0]}_traces.csv', str(PIXEL_MICRONS), str(FRAME_RATE)])
                proc.wait()
                if not proc.poll() == 0:
                    print(f'diffusion map -> failed with status:{proc.poll()}')
        except:
            failed_tasks.append(file)
            print(f"ERROR on {file}, code:{pid.returncode}")
            with open('./outputs/error_log.txt', 'a') as error_log:
                dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                input_str = f'{file} has an err[{pid.stderr.decode("utf-8")}]. DATE: {dt_string}\n'        
                error_log.write(input_str)
PBAR.close()
if len(failed_tasks) > 0:
    print(f'Prediction failed on {failed_tasks}, please check error_log file.')
else:
    print('Batch prediction finished succesfully.')
