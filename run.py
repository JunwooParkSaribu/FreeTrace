"""
Run FreeTrace on single .tiff file. 
"""

import os
import sys
from FreeTrace import Tracking, Localization



video_name = 'inputs/sample0.tiff'
OUTPUT_DIR = 'outputs'



"""
Basic parameters.
"""
WINDOW_SIZE = 7  # Size of sliding window for particle localisation. 
THRESHOLD = 1.0  # Detection threshold of particle localisation.
CUTOFF = 3  # Minimum length for reconstruction of trajectories.



"""
Advanced parameters.
"""
GPU_FOR_LOCALIZATION = True  # GPU acceleration with CUDA. (only available with NVIDIA GPU)
REALTIME_LOCALIZATION = False  # If you set this option as True, the processing time will be slower.
SAVE_LOCALIZATION_VIDEO = False

FBM_MODE = True  # Inference under fBm, if True (slow). Otherwise, classical Brownian motion if False (fast).
JUMP_THRESHOLD = None  # Maximum jump-distance for 1 frame.
GRAPH_DEPTH = 3  # Delta T.
REALTIME_TRACKING = False  # If you set this option as True, the processing time will be slower.
SAVE_TRACKING_VIDEO = False



if __name__ == "__main__":
    try:
        if not os.path.exists(f'{OUTPUT_DIR}'):
            os.makedirs(f'{OUTPUT_DIR}')
            
        loc = False
        track = False

        loc = Localization.run_process(input_video_path=video_name, output_path=OUTPUT_DIR,
                                       window_size=WINDOW_SIZE,
                                       threshold=THRESHOLD,
                                       gpu_on=GPU_FOR_LOCALIZATION,
                                       save_video=SAVE_LOCALIZATION_VIDEO,
                                       realtime_visualization=REALTIME_LOCALIZATION,
                                       verbose=1,
                                       batch=False)
        if loc:
            track = Tracking.run_process(input_video_path=video_name, output_path=OUTPUT_DIR,
                                         graph_depth=GRAPH_DEPTH,
                                         cutoff=CUTOFF,
                                         jump_threshold=JUMP_THRESHOLD,
                                         gpu_on=FBM_MODE,
                                         save_video=SAVE_TRACKING_VIDEO,
                                         realtime_visualization=REALTIME_TRACKING,
                                         verbose=1,
                                         batch=False)
            
    except Exception as e:
        sys.exit(f'Err code:{e} on file:{video_name}')
