*** 
Before run the prediction on videos with pipeline,
Please make sure you have already installed every pacakge such as Cython.
Then, please build C object files with setup.py following the command inside of the script.
To make sure that you've installed all necessary packages, Run ANDI_Localizaiton.py and ANDI_Tracking.py on a single video before running ANDI_pipeline.py for all videos.
For the filename, output directory and other parameters, please check andi2_config.txt file.
***

To remake results on AnDi2 final-phase datasets, please follow as below.

1. Clone the repository on your local device.
2. Download datasets, place the public_data_challenge_v0 folder inside of ANDI2_PRESET folder.
3. Build C object files with setup.py.
4. Run andi2_pipeline.py script with python.
5. Trajectory results will be made in the dataset folder.

References
[AnDi datasets](https://doi.org/10.5281/zenodo.10259556)
