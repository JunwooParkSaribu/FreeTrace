*** 
## Compatibility </br>
Python3.12 : tensorflow2.17 &#8594; pre-trained models [downloads](https://psilo.sorbonne-universite.fr/index.php/s/9W2pby29MGkQLDd)</br>
Python3.11 : tensorflow2.17 &#8594; pre-trained models [downloads](https://psilo.sorbonne-universite.fr/index.php/s/9W2pby29MGkQLDd)</br>
Python3.10 : tensorflow2.14 &#8594; pre-trained models [downloads](https://psilo.sorbonne-universite.fr/index.php/s/WqoCoFBc99A3Xbc)</br>
</br>
</br>
Please place the downloaded .keras, .npz files in this directory(FreeTrace/models).</br> 
Tensorflow performs an additional inference step for the trajectory reconstruction in tracking step.</br>
This would make FreeTrace slower because of the additional inference, however the quality of trajectory reconstruction would increase 4-5\% in general.</br>
If you don't want to turn on this additional inference, please set GPU_TRACK to False in config.txt. With GPU_TRACK=False, FreeTrace doesn't need Tensorflow.</br>
***
