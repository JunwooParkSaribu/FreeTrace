*** 
## Compatibility </br>
Python3.13 : tensorflow X, inference without GPU</br>
Python3.12 : tensorflow2.17 &#8594; pre-trained models [downloads](https://psilo.sorbonne-universite.fr/index.php/s/w9PrAQbxsNJrEFc/download/models_2_17.zip)</br>
Python3.11 : tensorflow2.17 &#8594; pre-trained models [downloads](https://psilo.sorbonne-universite.fr/index.php/s/w9PrAQbxsNJrEFc/download/models_2_17.zip)</br>
Python3.10 : tensorflow2.14 &#8594; pre-trained models [downloads](https://psilo.sorbonne-universite.fr/index.php/s/o8SZrWt4HePY8js/download/models_2_14.zip)</br>
</br>
</br> 
Tensorflow performs an additional inference step for the trajectory reconstruction in tracking step.</br>
This would make FreeTrace slower because of the additional inference, however the inference will be powered by neural networks for better quality.</br>
If you don't want to turn on this additional inference, please set GPU_ON to False. With GPU_ON=False, FreeTrace doesn't use Tensorflow.</br>
***