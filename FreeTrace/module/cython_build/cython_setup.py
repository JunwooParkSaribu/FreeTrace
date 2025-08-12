from setuptools import setup
from Cython.Build import cythonize
import os
import glob
import shutil


build_path = f"{os.getcwd()}/FreeTrace/module/cython_build"
setup(
    name='FreeTrace app',
    ext_modules=cythonize([f"{build_path}/image_pad.pyx", f"{build_path}/regression.pyx", f"{build_path}/cost_function.pyx"], language_level = "3", annotate=True),
)


source_file = glob.glob(f"{os.getcwd()}/cost_function*")[0]
extens = source_file.split(".")[-1]
destination_path = f"{os.getcwd()}/FreeTrace/module/cost_function.{extens}"
shutil.copy(source_file, destination_path)
os.remove(source_file)


source_file = glob.glob(f"{os.getcwd()}/image_pad*")[0]
extens = source_file.split(".")[-1]
destination_path = f"{os.getcwd()}/FreeTrace/module/image_pad.{extens}"
shutil.copy(source_file, destination_path)
os.remove(source_file)


source_file = glob.glob(f"{os.getcwd()}/regression*")[0]
extens = source_file.split(".")[-1]
destination_path = f"{os.getcwd()}/FreeTrace/module/regression.{extens}"
shutil.copy(source_file, destination_path)
os.remove(source_file)


directory_to_remove = f"{os.getcwd()}/build"
if os.path.exists(directory_to_remove):
    try:
        shutil.rmtree(directory_to_remove)
    except OSError as e:
        print(f"Error: {e.filename} - {e.strerror}.")


# Install Cython with pip, and build c object file with below command.
# python cython_setup.py build_ext --inplace