import os
import sys
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
SHIFT = params['localization']['SHIFT']
SAVE_VIDEO_LOC = params['localization']['LOC_VISUALIZATION']
LOC_GPU_AVAIL = params['localization']['GPU']


TIME_FORECAST = params['tracking']['TIME_FORECAST']
CUTOFF = params['tracking']['CUTOFF']
SAVE_VIDEO_TRACK = params['tracking']['TRACK_VISUALIZATION']
TRACK_GPU_AVAIL = params['tracking']['GPU']


if __name__ == "__main__":
    try:
        if not os.path.exists(f'{OUTPUT_DIR}'):
            os.makedirs(f'{OUTPUT_DIR}')
        loc = False
        track = False
        initialization(False, verbose=False, batch=False)
        """
        loc = Localization.run_process(input_video_path=video_name, output_path=OUTPUT_DIR,
                                       window_size=WINSIZE, threshold=THRES_ALPHA,
                                       deflation=DEFLATION_LOOP_IN_BACKWARD, shift=SHIFT,
                                       gpu_on=LOC_GPU_AVAIL, save_video=SAVE_VIDEO_LOC, realtime_visualization=True, verbose=1, batch=False)
        if loc:
        """
        track = Tracking.run_process(input_video_path=video_name, output_path=OUTPUT_DIR,
                                        time_forecast=TIME_FORECAST, cutoff=CUTOFF,
                                        gpu_on=TRACK_GPU_AVAIL, save_video=SAVE_VIDEO_TRACK, realtime_visualization=True, verbose=1, batch=False)
            
    except Exception as e:
        sys.exit(f'Err code:{e} on file:{video_name}')


