from module.FileIO import read_parameters, initialization
import subprocess
import os
import sys


def run_command(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process


params = read_parameters('./config.txt')
video_name = params['localization']['VIDEO']

initialization(True, verbose=True, batch=False)
pid = subprocess.run([sys.executable, 'Localization.py', '1', '1'])
if pid.returncode != 0:
    raise Exception(pid)
pid = subprocess.run([sys.executable, 'Tracking.py', '1', '1'])
if pid.returncode != 0:
    raise Exception(pid)
if os.path.exists('diffusion_image.py') and pid==0:
    proc = run_command([sys.executable.split('/')[-1], f'diffusion_image.py', f'./outputs/{video_name.strip().split("/")[-1].split(".tif")[0]}_traces.trxyt'])
    proc.wait()
    if proc.poll() == 0:
        print(f'diffusion map -> successfully finished')
    else:
        print(f'diffusion map -> failed with status:{proc.poll()}')
