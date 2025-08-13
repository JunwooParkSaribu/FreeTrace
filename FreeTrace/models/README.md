*** 
## Compatibility </br>
<img width="625" src="https://github.com/JunwooParkSaribu/FreeTrace/blob/main/tmps/compatibility_table.png">
Tensorflow2.17 &#8594; Models [downloads](https://psilo.sorbonne-universite.fr/index.php/s/w9PrAQbxsNJrEFc/download/models_2_17.zip) for Python 3.12 or 3.11 </br>
Tensorflow2.14 &#8594; Models [downloads](https://psilo.sorbonne-universite.fr/index.php/s/o8SZrWt4HePY8js/download/models_2_14.zip) for Python 3.10 </br>
</br>
</br> 
Tensorflow performs an additional inference step for the trajectory reconstruction in tracking step.</br>
This would make FreeTrace slower because of the additional inference, however the inference will be powered by neural networks for better quality.</br>
If you don't want to turn on this additional inference, please set GPU_ON to False. With GPU_ON=False, FreeTrace doesn't use Tensorflow.</br>
***