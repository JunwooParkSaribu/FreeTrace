import os
import sys
import subprocess
from module.FileIO import read_parameters, initialization


def run_command(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process


params = read_parameters('./config.txt')
video_name = params['localization']['VIDEO']

try:
    initialization(False, verbose=False, batch=False)
    proc = subprocess.run([sys.executable, 'Localization.py', '1', '0'])
    if proc.returncode != 0:
        raise Exception(proc)
    proc = subprocess.run([sys.executable, 'Tracking.py', '1', '0'])
    if proc.returncode != 0:
        raise Exception(proc)
    if os.path.exists('diffusion_image.py') and proc.returncode == 0:
        proc = run_command([sys.executable.split('/')[-1], f'diffusion_image.py', f'./outputs/{video_name.strip().split("/")[-1].split(".tif")[0]}_traces.trxyt'])
        proc.wait()
        if proc.poll() == 0:
            print(f'diffusion map -> successfully finished')
        else:
            print(f'diffusion map -> failed with status:{proc.poll()}')
except:
    sys.exit(f'Err code:{proc.returncode} on file:{video_name}')


