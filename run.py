import os
import sys
import subprocess
from FreeTrace import Tracking, Localization
from FreeTrace.module.FileIO import read_parameters, initialization


"""
Read configuration file.
"""
params = read_parameters('./config.txt')
video_name = params['localization']['VIDEO']
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
    try:
        if not os.path.exists(f'{OUTPUT_DIR}'):
            os.makedirs(f'{OUTPUT_DIR}')
        loc = False
        track = False

        initialization(False, verbose=False, batch=False)
        """
        loc = Localization.run_process(input_video=video_name, outpur_dir=OUTPUT_DIR,
                                    window_size=WINSIZE, threshold=THRES_ALPHA,
                                    deflation=DEFLATION_LOOP_IN_BACKWARD, sigma=SIGMA, shift=SHIFT,
                                    gpu_on=LOC_GPU_AVAIL, save_video=False, realtime_vis=True, verbose=1, batch=False)
        """
        #if loc:
        track = Tracking.run_process(input_video=video_name, outpur_dir=OUTPUT_DIR,
                                    blink_lag=BLINK_LAG, cutoff=CUTOFF,
                                    pixel_microns=PIXEL_MICRONS, frame_rate=FRAME_RATE,
                                    gpu_on=TRACK_GPU_AVAIL, save_video=False, verbose=1, batch=False)

        if os.path.exists('diffusion_image.py') and track:
            proc = run_command([sys.executable.split('/')[-1], f'diffusion_image.py', f'./{OUTPUT_DIR}/{video_name.strip().split("/")[-1].split(".tif")[0]}_traces.csv', str(PIXEL_MICRONS), str(FRAME_RATE)])
            proc.wait()
            if proc.poll() == 0:
                print(f'diffusion map -> successfully finished')
            else:
                print(f'diffusion map -> failed with status:{proc.poll()}: {proc.stderr.read().decode()}')
    except:
        sys.exit(f'Err code:{proc.returncode} on file:{video_name}')


