from setuptools import setup
from Cython.Build import cythonize
import os
import glob
import shutil
import re


freetrace_path = ''
freetrace_path += 'FreeTrace'.join(re.split(r'FreeTrace', __file__)[:-1]) + 'FreeTrace'
freetrace_path = freetrace_path.replace('\\', '/')
build_path = f"{freetrace_path}/module/cython_build"
print(f"Current build path: {build_path}")


setup(
    name='FreeTrace app',
    ext_modules=cythonize([f"{build_path}/image_pad.pyx", f"{build_path}/regression.pyx", f"{build_path}/cost_function.pyx"], language_level = "3", annotate=True),
)


source_file = glob.glob(f"{build_path}/cost_function*")[0]
extens = source_file.split(".")[-1]
destination_path = f"{freetrace_path}/module/cost_function.{extens}"
shutil.copy(source_file, destination_path)
os.remove(source_file)


source_file = glob.glob(f"{build_path}/image_pad*")[0]
extens = source_file.split(".")[-1]
destination_path = f"{freetrace_path}/module/image_pad.{extens}"
shutil.copy(source_file, destination_path)
os.remove(source_file)


source_file = glob.glob(f"{build_path}/regression*")[0]
extens = source_file.split(".")[-1]
destination_path = f"{freetrace_path}/module/regression.{extens}"
shutil.copy(source_file, destination_path)
os.remove(source_file)


directory_to_remove = f"{build_path}/build"
if os.path.exists(directory_to_remove):
    try:
        shutil.rmtree(directory_to_remove)
    except OSError as e:
        print(f"Error: {e.filename} - {e.strerror}.")


# Install Cython with pip, and build c object file with below command.
# python cython_setup.py build_ext --inplace