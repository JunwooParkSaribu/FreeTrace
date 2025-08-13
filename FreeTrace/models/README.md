*** 
## Compatibility </br>
<p align="center">
  <img width="825" src="https://github.com/JunwooParkSaribu/FreeTrace/blob/main/tmps/compatibility_table.png">
</p>

Python 3.12 / 3.11 &#8594; Models [downloads](https://psilo.sorbonne-universite.fr/index.php/s/w9PrAQbxsNJrEFc/download/models_2_17.zip) for Tensorflow 2.17 </br>
Python 3.10 &#8594; Models [downloads](https://psilo.sorbonne-universite.fr/index.php/s/o8SZrWt4HePY8js/download/models_2_14.zip) for Tensorflow 2.14 </br>
</br>
</br> 
Tensorflow performs the trajectory reconstruction with the fractional Brownian motion in tracking step.</br>
This makes FreeTrace slower because of an additional inference, however this increases the quality of trajectories in general.</br>
If you don't want to tracking with the fBm, please set TRACK_GPU_AVAIL to False. Then, FreeTrace doesn't use Tensorflow and GPU for tracking.</br>
***