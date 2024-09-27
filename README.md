[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13336251.svg)](https://doi.org/10.5281/zenodo.13336251)
## FreeTrace

> [!IMPORTANT]  
> Requirements </br>
> - Python 3.10 or higher
> - Cython

---------------------------------------------------------------------------------------------------- </br>
<b>*** To run the <b>FreeTrace</b> for [AnDi2 Challenge](http://andi-challenge.org/challenge-2024/#andi2seminar) final-phase datasets[^1], check ANDI2_PRESET folder. ***</b></br>
PRE-REQUISITE: Build C object files by running setup.py on your platform(check setup.py file).</br>
---------------------------------------------------------------------------------------------------- </br>

<b>FreeTrace</b> infers individual trajectory from time-series images. To detect the particles and their positions at sub-pixel level, FreeTrace first extends the image sequences by sampling of noises at the edges of images. These extensions of images allow detecting the particles at the edges of images since FreeTrace utilizes sliding windows to calculate the particle's position at sub-pixel level. Next, FreeTrace estimates the likelihoods with a given PSF function for each pixel and makes hypothesis maps to determine whether a particle exists at a given pixel. FreeTrace then finds local maxima from the constructed hypothesis maps to perform a Gaussian regression on the maxima. To find the precise position of particles at sub-pixel level, FreeTrace performs 2D Gaussian regression in a linear system. Finally, FreeTrace reconnects the detected particles by constructing a network and infer the most probable trajectories..

<h2>Visualized result of FreeTrace</h2>
<table border="0"> 
        <tr> 
            <td><img src="https://github.com/JunwooParkSaribu/FreeTrace/main/tmps/result0.png" width="128" height="128"></td> 
            <td><img src="https://github.com/JunwooParkSaribu/FreeTrace/main/tmps/result1.png" width="128" height="128"></td> 
        </tr> 
</table>


<h3> Brief description of the method </h3>

</br></br>


<h3> To remake results on AnDi2 final-phase datasets</h3>
1. Clone the repository on your local device.</br>
2. Download datasets, place the *public_data_challenge_v0* folder inside of *ANDI2_PRESET* folder.</br>
3. Build object files with setup.py.</br>
4. Run *andi2_pipeline.py* script with python.</br>
5. Trajectory results will be made in the dataset folder.

<h3> Contacts </h3>
junwoo.park@sorbonne-universite.fr<br>

<h3> References </h3>
[^1]: [AnDi datasets](https://doi.org/10.5281/zenodo.10259556)

