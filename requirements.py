import sys
import subprocess

non_installed_packages = {}
with open('./requirements.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        package = line.strip().split('\n')[0]
        try:
            pid = subprocess.run([sys.executable, '-m', 'pip', 'install', package])
            if pid.returncode != 0:
                non_installed_packages[package] = pid.returncode
            if 'Cython' in package and pid.returncode == 0:
                subprocess.run([sys.executable, 'cython_setup.py', 'build_ext', '--inplace'])
        except:
            pass

print('')
for non_installed_pacakge in non_installed_packages.keys():
    print(f'***** Package [{non_installed_pacakge}] installation failed due to subprocess exit code:{non_installed_packages[non_installed_pacakge]}, please install it manually. *****')
print('')

#subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
