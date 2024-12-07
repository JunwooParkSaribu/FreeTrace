[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13336251.svg)](https://doi.org/10.5281/zenodo.13336251)
## FreeTrace

> [!IMPORTANT]  
> Requirements </br>
> - C compiler (clang)</br>
> - Python3.10 &#8593;, python3-dev, pip</br>
> - GPU & Cuda12 on Linux/WSL2(Ubuntu22.04 &#8593;) (recommended)</br>
> - With GPU, Pre-trained [models](https://github.com/JunwooParkSaribu/FreeTrace/blob/main/models/README.md) (recommended)</br>

> [!NOTE]  
> PRE-REQUISITE: package installation by running installation.py</br>
> Check [compatibilities](https://github.com/JunwooParkSaribu/FreeTrace/blob/main/models/README.md) of Python, Ubuntu and Tensorflow to run FreeTrace with source code.</br>
> with GPU off on tracking, FreeTrace only considers standard Brownian motion for inferences.</br>
> on MacOS, FreeTrace can inference in tracking with tensorflow-metal, however the speed is slower.</br>

------------------------------------------------------------------------------------------------------------------</br>
<b>*** For the video prediction of [AnDi2 Challenge](http://andi-challenge.org/challenge-2024/#andi2seminar) final-phase datasets, please check [ANDI2_PRESET](https://github.com/JunwooParkSaribu/FreeTrace/blob/main/ANDI2_PRESET). ***</b></br>
------------------------------------------------------------------------------------------------------------------</br>

&nbsp;&nbsp;<b>FreeTrace</b> infers individual trajectories from time-series images. To detect the particles and their positions at sub-pixel level, FreeTrace first extends the image sequences by sampling noises at the edges of images. These extensions of images allow detecting the particles at the edges of images since FreeTrace utilises sliding windows to calculate the particle's position at sub-pixel level. Next, FreeTrace estimates the existence of particles at a pixel with a given PSF function for each sliding window and makes a hypothesis map to determine whether a particle exists at a given sliding window or not. FreeTrace then finds local maxima from the constructed hypothesis maps. To find the precise centre-position of particles at sub-pixel level, FreeTrace performs 2D Gaussian regression by transforming it into a linear system. Finally, FreeTrace reconnects the detected particles by constructing a network and infer the most probable trajectories by calculating the reconnection-likelihoods on paths.</br>

<h2>Visualized result of FreeTrace</h2>
<img width="825" src="https://github.com/JunwooParkSaribu/FreeTrace/blob/main/tmps/stars.gif">
<table border="0"> 
        <tr> 
            <td><img src="https://github.com/JunwooParkSaribu/FreeTrace/blob/main/tmps/trjs0.gif" width="230" height="230"></td> 
            <td><img src="https://github.com/JunwooParkSaribu/FreeTrace/blob/main/tmps/trjs1.gif" width="230" height="230"></td>
            <td><img src="https://github.com/JunwooParkSaribu/FreeTrace/blob/main/tmps/trjs2.gif" width="285" height="230"></td>
        </tr>  
</table>

[Brief description of the method] will be available soon.

<h3> Contacts </h3>

<junwoo.park@sorbonne-universite.fr>

<br>
