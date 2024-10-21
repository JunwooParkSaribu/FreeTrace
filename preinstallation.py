import sys
import subprocess

subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
subprocess.run([sys.executable, 'cython_setup.py', 'build_ext', '--inplace'])
