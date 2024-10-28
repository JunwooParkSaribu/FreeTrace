import os
import sys
import subprocess


include_path = '/usr/include/python3.10'
#include_path = '/Library/Frameworks/Python.framework/Versions/3.10/include/python3.10'
non_installed_packages = {}


with open('./requirements.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        package = line.strip().split('\n')[0]
        try:
            pid = subprocess.run([sys.executable, '-m', 'pip', 'install', package])
            if pid.returncode != 0:
                non_installed_packages[package] = pid.returncode
        except:
            pass

try:
    subprocess.run(['clang', '-Wno-unused-result', '-Wsign-compare', '-Wunreachable-code', '-fno-common', '-dynamic', '-DNDEBUG', '-g', '-fwrapv', '-O3', '-Wall', '-arch', 'arm64', '-arch', 'x86_64', '-g', '-fPIC', '-I', f'{include_path}',
                    '-c', './module/image_pad.c', '-o', './module/image_pad.o'])
    subprocess.run(['clang', '-bundle', '-shared', '-g', '-fwrapv', '-undefined', 'dynamic_lookup', '-arch', 'arm64', '-arch', 'x86_64',
                    '-g', './module/image_pad.o', '-o', './module/image_pad.so'])
    subprocess.run(['clang', '-Wno-unused-result', '-Wsign-compare', '-Wunreachable-code', '-fno-common', '-dynamic', '-DNDEBUG', '-g', '-fwrapv', '-O3', '-Wall', '-arch', 'arm64', '-arch', 'x86_64', '-g', '-fPIC', '-I', f'{include_path}',
                    '-c', './module/regression.c', '-o', './module/regression.o'])
    subprocess.run(['clang', '-bundle', '-shared', '-g', '-fwrapv', '-undefined', 'dynamic_lookup', '-arch', 'arm64', '-arch', 'x86_64',
                    '-g', './module/regression.o', '-o', './module/regression.so'])
    subprocess.run(['rm', './module/image_pad.o', './module/regression.o'])
    if os.path.exists(f'./module/image_pad.so') and os.path.exists(f'./module/regression.so'):
        print('')
        print(f'***** module compiling finished successfully. *****')
except Exception as e:
    print(f'\n***** Compiling Error: {e} *****')
    pass


if len(list(non_installed_packages.keys())) == 0:
    print('')
    print(f'***** Pacakge installations finished succesfully. *****')
    print('')
else:
    print('')
    for non_installed_pacakge in non_installed_packages.keys():
        print(f'***** Package [{non_installed_pacakge}] installation failed due to subprocess exit code:{non_installed_packages[non_installed_pacakge]}, please install it manually. *****')
    print('')
if not os.path.exists(f'./models/theta_hat.npz'):
    print(f'***** Parmeters[theta_hat.npz] are not found for trajectory inference, please contact author for the pretrained models. *****\n')
