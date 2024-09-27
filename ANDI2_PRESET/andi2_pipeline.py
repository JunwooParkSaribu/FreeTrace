import subprocess
import sys

N_EXP = 12
N_FOVS = 30
public_data_path = 'public_data_challenge_v0'


def run_command(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process


def write_config(exp_n, fov_n):
    content = \
    f"\
    VIDEO=./{public_data_path}/track_1/exp_{exp_n}/videos_fov_{fov_n}.tiff\n\
    OUTPUT_DIR=./{public_data_path}/track_1/exp_{exp_n}\n\
    \n\
    # LOCALIZATION\n\
    WINDOW_SIZE = 5\n\
    THRESHOLD_ALPHA = 1.0\n\
    DEFLATION_LOOP_IN_BACKWARD = 1\n\
    SIGMA = 4.0\n\
    LOC_VISUALIZATION = False\n\
    \n\
    \n\
    # TRACKING\n\
    CUTOFF = 2\n\
    BLINK_LAG = 1\n\
    PIXEL_MICRONS = 1\n\
    FRAME_PER_SEC = 1\n\
    TRACK_VISUALIZATION = False\n\
    \n\
    \n\
    # SUPP\n\
    LOC_PARALLEL = False\n\
    CORE = 4\n\
    DIV_Q = 50\n\
    SHIFT = 1\n\
    \n\
    TRACKING_PARALLEL = False\n\
    AMP_MAX_LEN = 1.5\n\
    "

    with open("./andi2_config.txt", 'w') as config:
        config.write(content)


for exp in range(0, N_EXP):
    for fov in range(0, N_FOVS):
        while True:
            write_config(exp, fov)
            proc_loc = run_command([sys.executable.split('/')[-1], f'./ANDI_Localization.py'])
            proc_loc.wait()
            if proc_loc.poll() == 0:
                print(f'Exp:{exp} Fov:{fov} localization finished')
            else:
                print(f'Exp:{exp} Fov:{fov} localization has failed: status:{proc_loc.poll()}')

            proc_track = run_command([sys.executable.split('/')[-1], f'./ANDI_Tracking.py'])
            proc_track.wait()
            if proc_track.poll() == 0:
                print(f'Exp:{exp} Fov:{fov} tracking has successfully finished')
            else:
                print(f'Exp:{exp} Fov:{fov} tracking has failed: status:{proc_track.poll()}')

            if proc_loc.poll() == 0 and proc_track.poll() == 0:
                break
