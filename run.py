from module.FileIO import read_parameters
import subprocess
import os
import sys


def run_command(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process


params = read_parameters('./config.txt')
video_name = params['localization']['VIDEO']


with open("Localization.py") as f:
    exec(f.read())
with open("Tracking.py") as f:
    exec(f.read())
if os.path.exists('diffusion_image.py'):
    proc = run_command([sys.executable.split('/')[-1], f'diffusion_image.py', f'./outputs/{video_name.strip().split("/")[-1].split(".tif")[0]}_traces.trxyt'])
    proc.wait()
    if proc.poll() == 0:
        print(f'diffusion map -> successfully finished')
    else:
        print(f'diffusion map -> failed with status:{proc.poll()}')
