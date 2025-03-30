"""
Run FreeTrace on single .tiff file. 
"""

import os
import sys
from FreeTrace import Tracking, Localization


video_name = 'inputs/sample0.tiff'
OUTPUT_DIR = 'outputs'


WINSIZE = 7
THRESHOLD = 1.0
SAVE_VIDEO_LOC = False
REAL_LOC = False
LOC_GPU_AVAIL = True


TIME_FORECAST = 2
CUTOFF = 2
JUMP_THRESHOLD = None
SAVE_VIDEO_TRACK = False
REAL_TRACK = False
TRACK_GPU_AVAIL = True


if __name__ == "__main__":
    try:
        if not os.path.exists(f'{OUTPUT_DIR}'):
            os.makedirs(f'{OUTPUT_DIR}')
        loc = False
        track = False

        loc = Localization.run_process(input_video_path=video_name, output_path=OUTPUT_DIR,
                                       window_size=WINSIZE, threshold=THRESHOLD,
                                       gpu_on=LOC_GPU_AVAIL, save_video=SAVE_VIDEO_LOC, realtime_visualization=REAL_LOC, verbose=1, batch=False)
        if loc:
            track = Tracking.run_process(input_video_path=video_name, output_path=OUTPUT_DIR,
                                         time_forecast=TIME_FORECAST, cutoff=CUTOFF, jump_threshold=JUMP_THRESHOLD,
                                         gpu_on=TRACK_GPU_AVAIL, save_video=SAVE_VIDEO_TRACK, realtime_visualization=REAL_TRACK, verbose=1, batch=False)
            
    except Exception as e:
        sys.exit(f'Err code:{e} on file:{video_name}')
