*** 
## Compatibility </br>
Python3.12 : tensorflow2.17 &#8594; pre-trained models [downloads](https://psilo.sorbonne-universite.fr/index.php/s/9W2pby29MGkQLDd/download/models_2_17.zip)</br>
Python3.11 : tensorflow2.17 &#8594; pre-trained models [downloads](https://psilo.sorbonne-universite.fr/index.php/s/9W2pby29MGkQLDd/download/models_2_17.zip)</br>
Python3.10 : tensorflow2.14 &#8594; pre-trained models [downloads](https://psilo.sorbonne-universite.fr/index.php/s/WqoCoFBc99A3Xbc/download/models_2_14.zip)</br>
</br>
</br> 
Tensorflow performs an additional inference step for the trajectory reconstruction in tracking step.</br>
This would make FreeTrace slower because of the additional inference, however the inference will be powered by neural networks for better quality.</br>
If you don't want to turn on this additional inference, please set the parameter of tracking, gpu_on to False. With gpu_on=False, FreeTrace won't use Tensorflow.</br>
The downloads of above models and C compilations can be automatically done via importing FreeTrace.installtion in python interpreter on Linux.</br>
***