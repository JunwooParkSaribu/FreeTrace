import os
import sys
import time
import subprocess


"""
Pre-requisite: Corresponding Cuda versions.
https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_network
"""


#include_path = '/usr/include/python3.10'
#include_path = '/Library/Frameworks/Python.framework/Versions/3.10/include/python3.10'
non_installed_packages = {}
include_path = None
found_head_file = 0
freetrace_path = ""
for it in __file__.split("FreeTrace")[:-1]:
    freetrace_path += it
    freetrace_path += 'FreeTrace/'

if '3.12' in sys.version or '3.11' in sys.version:
    tf_version = 'tensorflow[and-cuda]==2.17'
    python_version = 3.12
elif '3.11' in sys.version:
    tf_version = 'tensorflow[and-cuda]==2.17'
    python_version = 3.11
elif '3.10' in sys.version:
    tf_version = 'tensorflow[and-cuda]==2.14.1'
    python_version = 3.10
else:
    sys.exit('***** python version 3.10/11/12 required *****')

print("***** FreeTrace needs Cuda. if Cuda is not installed, please visit [https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_network] *****")
print("***** FreeTrace includes the installation of below items. *****")
with open(f"{freetrace_path}/requirements.txt", 'r') as f:
    lines = f.readlines()
    for line in lines:
        if f'tensorflow' in line:
            print(tf_version)
        else:
            print(f'{line.split("\n")[0]}', end=' ')
print("unzip clang python3-tk python3-dev python3-pip\n")
print("If you don't want to continue, please stop the process.")
time.sleep(5)



print("\n***** Installtion starts *****")
subprocess.run(['sudo', 'apt-get', 'update'])
subprocess.run(['sudo', 'apt-get', 'install', 'unzip'])
subprocess.run(['sudo', 'apt', 'install', 'clang'])
subprocess.run(['sudo', 'apt-get', 'install', 'python3-tk'])
subprocess.run(['sudo', 'apt', 'install', 'python3-dev'])
subprocess.run(['sudo', 'apt', 'install', 'python3-pip'])


if python_version == 3.10:
    subprocess.run(['rm', '-r', f'{freetrace_path}/models'])
    subprocess.run(['wget', 'https://psilo.sorbonne-universite.fr/index.php/s/WqoCoFBc99A3Xbc/download/models_2_14.zip', '-P' f'{freetrace_path}'])
    subprocess.run(['unzip', '-o', f'{freetrace_path}/models_2_14.zip', '-d', f'{freetrace_path}'])
    subprocess.run(['cp', '-r', f'{freetrace_path}/models_2_14', f'{freetrace_path}/models'])
    subprocess.run(['rm', '-r', f'{freetrace_path}/models_2_14'])
    subprocess.run(['rm', f'{freetrace_path}/models_2_14.zip'])
else:
    subprocess.run(['rm', '-r', f'{freetrace_path}/models'])
    subprocess.run(['wget', 'https://psilo.sorbonne-universite.fr/index.php/s/9W2pby29MGkQLDd/download/models_2_17.zip', '-P' f'{freetrace_path}'])
    subprocess.run(['unzip', '-o', f'{freetrace_path}/models_2_17.zip', '-d', f'{freetrace_path}'])
    subprocess.run(['cp', '-r', f'{freetrace_path}/models_2_17', f'{freetrace_path}/models'])
    subprocess.run(['rm', '-r', f'{freetrace_path}/models_2_17'])
    subprocess.run(['rm', f'{freetrace_path}/models_2_17.zip'])
    

for root, dirs, files in os.walk("/usr", topdown=False):
    for name in files:
        if 'Python.h' in name:
            include_path = f'{root}'
            found_head_file = 1

if found_head_file == 0 :
    for root, dirs, files in os.walk("/Library", topdown=False):
        for name in files:
            if 'Python.h' in name:
                include_path = f'{root}'
                found_head_file = 1

if include_path is None and found_head_file == 0:
    sys.exit(f'***** Please install python-dev to install modules, Python.h header file was not found. *****')
    
if not os.path.exists(f'{freetrace_path}/models/theta_hat.npz') or not os.path.exists(f'{freetrace_path}/models/std_sets.npz'):
    print(f'***** Parmeters[theta_hat.npz, std_sets.npz] are not found for trajectory inference, please contact author for the pretrained models. *****\n')
    sys.exit()

if os.path.exists(f'{freetrace_path}/requirements.txt'):
    with open(f'{freetrace_path}/requirements.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            package = line.strip().split('\n')[0]
            if 'tensorflow' in package:
                package = tf_version
            try:
                pid = subprocess.run([sys.executable, '-m', 'pip', 'install', package])
                if pid.returncode != 0:
                    non_installed_packages[package] = pid.returncode
            except:
                pass

try:
    if python_version == 3.10:
        subprocess.run(['clang', '-Wno-unused-result', '-Wsign-compare', '-Wunreachable-code', '-fno-common', '-dynamic', '-DNDEBUG', '-g', '-fwrapv', '-O3', '-Wall', '-arch', 'arm64', '-arch', 'x86_64', '-g', '-fPIC', '-I', f'{include_path}',
                        '-c', f'{freetrace_path}/module/image_pad.c', '-o', f'{freetrace_path}/module/image_pad.o'])
        subprocess.run(['clang', '-shared', '-g', '-fwrapv', '-undefined', 'dynamic_lookup', '-arch', 'arm64', '-arch', 'x86_64',
                        '-g', f'{freetrace_path}/module/image_pad.o', '-o', f'{freetrace_path}/module/image_pad.so'])
        subprocess.run(['clang', '-Wno-unused-result', '-Wsign-compare', '-Wunreachable-code', '-fno-common', '-dynamic', '-DNDEBUG', '-g', '-fwrapv', '-O3', '-Wall', '-arch', 'arm64', '-arch', 'x86_64', '-g', '-fPIC', '-I', f'{include_path}',
                        '-c', f'{freetrace_path}/module/regression.c', '-o', f'{freetrace_path}/module/regression.o'])
        subprocess.run(['clang', '-shared', '-g', '-fwrapv', '-undefined', 'dynamic_lookup', '-arch', 'arm64', '-arch', 'x86_64',
                        '-g', f'{freetrace_path}/module/regression.o', '-o', f'{freetrace_path}/module/regression.so'])
    else:
        subprocess.run(['clang', '-Wno-unused-result', '-Wsign-compare', '-Wunreachable-code', '-fno-common', '-dynamic', '-DNDEBUG', '-g', '-fwrapv', '-O3', '-Wall', '-g', '-fPIC', '-I', f'{include_path}',
                        '-c', f'{freetrace_path}/module/image_pad.c', '-o', f'{freetrace_path}/module/image_pad.o'])
        subprocess.run(['clang', '-shared', '-g', '-fwrapv', '-undefined', 'dynamic_lookup',
                        '-g', f'{freetrace_path}/module/image_pad.o', '-o', f'{freetrace_path}/module/image_pad.so'])
        subprocess.run(['clang', '-Wno-unused-result', '-Wsign-compare', '-Wunreachable-code', '-fno-common', '-dynamic', '-DNDEBUG', '-g', '-fwrapv', '-O3', '-Wall', '-g', '-fPIC', '-I', f'{include_path}',
                        '-c', f'{freetrace_path}/module/regression.c', '-o', f'{freetrace_path}/module/regression.o'])
        subprocess.run(['clang', '-shared', '-g', '-fwrapv', '-undefined', 'dynamic_lookup',
                        '-g', f'{freetrace_path}/module/regression.o', '-o', f'{freetrace_path}/module/regression.so'])

    subprocess.run(['rm', f'{freetrace_path}/module/image_pad.o', f'{freetrace_path}/module/regression.o'])
    if os.path.exists(f'{freetrace_path}/module/image_pad.so') and os.path.exists(f'{freetrace_path}/module/regression.so'):
        print('')
        print(f'***** module compiling finished successfully. *****')
except Exception as e:
    print(f'\n***** Compiling Error: {e} *****')
    pass


if len(list(non_installed_packages.keys())) == 0:
    print(f'***** Pacakge installations finished succesfully. *****')
    print(f'***** Python veirsion: {str(python_version)} *****')
    print('')
else:
    for non_installed_pacakge in non_installed_packages.keys():
        print(f'***** Package [{non_installed_pacakge}] installation failed due to subprocess exit code:{non_installed_packages[non_installed_pacakge]}, please install it manually. *****')
    print('')
