import os
import sys
import subprocess
from datetime import datetime


def run_command(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process


def write_config(filename):
    content = \
    f"\
    VIDEO=./inputs/{filename}\n\
    OUTPUT_DIR=./outputs\n\
    \n\
    # LOCALIZATION\n\
    WINDOW_SIZE = 9\n\
    THRESHOLD_ALPHA = 1.0\n\
    DEFLATION_LOOP_IN_BACKWARD = 1\n\
    SIGMA = 4.0\n\
    LOC_VISUALIZATION = False\n\
    \n\
    \n\
    # TRACKING\n\
    CUTOFF = 2\n\
    BLINK_LAG = 2\n\
    PIXEL_MICRONS = 0.16\n\
    FRAME_PER_SEC = 0.01\n\
    TRACK_VISUALIZATION = False\n\
    \n\
    \n\
    # SUPP\n\
    GPU = True\n\
    CORE = 4\n\
    DIV_Q = 50\n\
    SHIFT = 1\n\
    "
    with open("./config.txt", 'w') as config:
        config.write(content)


if not os.path.exists('./outputs'):
    os.makedirs('./outputs')
file_list = os.listdir('./inputs')
for file in file_list:
    if file.strip().split('.')[-1] == 'tif' or file.strip().split('.')[-1] == 'tiff':
        print(f"------- processing on {file} -------")
        write_config(file)
        try:
            with open("Localization.py") as f:
                exec(f.read())
            with open("Tracking.py") as f:
                exec(f.read())
            if os.path.exists('diffusion_image.py'):
                proc = run_command([sys.executable.split('/')[-1], f'diffusion_image.py', f'./outputs/{file.strip().split(".tif")[0]}_traces.trxyt'])
                proc.wait()
                if proc.poll() == 0:
                    print(f'diffusion map -> successfully finished')
                else:
                    print(f'diffusion map -> failed with status:{proc.poll()}')
            print(f"------- {file} is succesfully finished -------")
        except Exception as e:
            print(f"ERROR on {file}: {e}")
            with open('./error_log.txt', 'a') as error_log:
                dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                input_str = f'{file} has an err[{e}]. DATE: {dt_string}\n'        
                error_log.write(input_str)
