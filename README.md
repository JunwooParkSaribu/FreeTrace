[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13336251.svg)](https://doi.org/10.5281/zenodo.13336251)
## FreeTrace

> [!IMPORTANT]  
> Requirements </br>
> - C compiler</br>
> - Python 3.10 or higher</br>
> - Cython</br>

---------------------------------------------------------------------------------------------------- </br>
<b>*** For the video prediction of [AnDi2 Challenge](http://andi-challenge.org/challenge-2024/#andi2seminar) final-phase datasets, please check [ANDI2_PRESET](https://github.com/JunwooParkSaribu/FreeTrace/blob/main/ANDI2_PRESET). ***</b></br>
PRE-REQUISITE: Build C object files by running setup.py on your platform (check setup.py file).</br>
---------------------------------------------------------------------------------------------------- </br>

<b>FreeTrace</b> infers individual trajectory from time-series images. To detect the particles and their positions at sub-pixel level, FreeTrace first extends the image sequences by sampling of noises at the edges of images. These extensions of images allow detecting the particles at the edges of images since FreeTrace utilizes sliding windows to calculate the particle's position at sub-pixel level. Next, FreeTrace estimates the likelihoods with a given PSF function for each pixel and makes hypothesis maps to determine whether a particle exists at a given pixel. FreeTrace then finds local maxima from the constructed hypothesis maps to perform a Gaussian regression on the maxima. To find the precise position of particles at sub-pixel level, FreeTrace performs 2D Gaussian regression in a linear system. Finally, FreeTrace reconnects the detected particles by constructing a network and infer the most probable trajectories.</br>

<h2>Visualized result of FreeTrace</h2>

![](https://github.com/JunwooParkSaribu/FreeTrace/blob/main/tmps/result0.png)
![](https://github.com/JunwooParkSaribu/FreeTrace/blob/main/tmps/result1.png)

</br>
[Brief description of the method](https://github.com/JunwooParkSaribu/FreeTrace/blob/main/tmps/method_description.pdf)

<h3> Contacts </h3>
junwoo.park@sorbonne-universite.fr<br>
