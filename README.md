[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13336251.svg)](https://doi.org/10.5281/zenodo.13336251)
## FreeTrace

> [!IMPORTANT]  
> Requirements </br>
> - C compiler</br>
> - Python 3.10 or higher</br>
> - GPU & Cuda on Linux/WSL2 (recommended)</br>
> - With GPU, Pre-trained [models](https://drive.google.com/file/d/1WF0eW8Co23-mKQiHNH-KHHK_lJiIW-WC/view?usp=sharing) (recommended)</br>

> [!NOTE]  
> PRE-REQUISITE: package installation by running requirements.py</br>
> with GPU off on tracking, FreeTrace only considers std. brownian motions for the inferences.</br>
> on MacOS, FreeTrace can infereces with tensorflow-metal, however the speed is slower without Cuda.</br>

------------------------------------------------------------------------------------------------------------------</br>
<b>*** For the video prediction of [AnDi2 Challenge](http://andi-challenge.org/challenge-2024/#andi2seminar) final-phase datasets, please check [ANDI2_PRESET](https://github.com/JunwooParkSaribu/FreeTrace/blob/main/ANDI2_PRESET). ***</b></br>
------------------------------------------------------------------------------------------------------------------</br>

&nbsp;&nbsp;<b>FreeTrace</b> infers individual trajectories from time-series images. To detect the particles and their positions at sub-pixel level, FreeTrace first extends the image sequences by sampling noises at the edges of images. These extensions of images allow detecting the particles at the edges of images since FreeTrace utilises sliding windows to calculate the particle's position at sub-pixel level. Next, FreeTrace estimates the existence of particles at a pixel with a given PSF function for each sliding window and makes a hypothesis map to determine whether a particle exists at a given sliding window or not. FreeTrace then finds local maxima from the constructed hypothesis maps. To find the precise centre-position of particles at sub-pixel level, FreeTrace performs 2D Gaussian regression by transforming it into a linear system. Finally, FreeTrace reconnects the detected particles by constructing a network and infer the most probable trajectories by calculating the reconnection-likelihoods on paths.</br>

<h2>Visualized result of FreeTrace</h2>
<table border="0"> 
        <tr> 
            <td><img src="https://github.com/JunwooParkSaribu/FreeTrace/blob/main/tmps/trjs0.gif" width="230" height="230"></td> 
            <td><img src="https://github.com/JunwooParkSaribu/FreeTrace/blob/main/tmps/trjs1.gif" width="230" height="230"></td>
            <td><img src="https://github.com/JunwooParkSaribu/FreeTrace/blob/main/tmps/trjs2.gif" width="285" height="230"></td>
        </tr>  
</table>

[Brief description of the method](https://github.com/JunwooParkSaribu/FreeTrace/blob/main/tmps/method_description.pdf)

<h3> Contacts </h3>

<junwoo.park@sorbonne-universite.fr>

<br>
