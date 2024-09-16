import os


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
    LOC_PARALLEL = False\n\
    CORE = 4\n\
    DIV_Q = 50\n\
    SHIFT = 1\n\
    \n\
    TRACKING_PARALLEL = False\n\
    AMP_MAX_LEN = 1.3\n\
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
        with open("Localization.py") as f:
            exec(f.read())
        with open("Tracking.py") as f:
            exec(f.read())
