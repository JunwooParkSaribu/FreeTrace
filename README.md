
[![Static Badge](https://img.shields.io/badge/bioRxiv-FreeTrace-red)](https://doi.org/10.64898/2026.01.08.698486)
![Static Badge](https://img.shields.io/badge/%3E%3D3.10-1?style=flat&label=Python&color=blue)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13336251.svg)](https://doi.org/10.5281/zenodo.13336251)
![GitHub License](https://img.shields.io/github/license/JunwooParkSaribu/FreeTrace)

## FreeTrace

> [!IMPORTANT]  
> Requirements </br>
> - Windows(10/11) / GNU/Linux(Debian/Ubuntu) / MacOS(Sequoia/Tahoe)</br>
> - C compiler (clang)</br>
> - Python3.10 &#8593;</br>
> - GPU & Cuda12 on GNU/Linux with pre-trained [models](https://github.com/JunwooParkSaribu/FreeTrace/blob/main/FreeTrace/models/README.md) (recommended)</br>


> [!NOTE]  
> - PRE-REQUISITE: pre-installation and compilation, check [tutorial](https://github.com/JunwooParkSaribu/FreeTrace/blob/main/tutorial.ipynb). </br>
> - Check [compatibilities](https://github.com/JunwooParkSaribu/FreeTrace/blob/main/FreeTrace/models/README.md) of Python and Tensorflow to run FreeTrace with source code.</br>
> - Without GPU, FreeTrace is slow if it infers under fractional Brownian motion.</br>
> - Current version is stable with python 3.10 / 3.11 / 3.12</br>
> - Tutorial on [Colab](https://colab.research.google.com/github/JunwooParkSaribu/FreeTrace/blob/main/tutorial.ipynb)</br>

<h2>Visualised FreeTrace results</h2>
<img width="825" src="https://github.com/JunwooParkSaribu/FreeTrace/blob/main/tmps/stars.gif">
<table border="0"> 
        <tr> 
            <td><img src="https://github.com/JunwooParkSaribu/FreeTrace/blob/main/tmps/trjs0.gif" width="230" height="230"></td> 
            <td><img src="https://github.com/JunwooParkSaribu/FreeTrace/blob/main/tmps/trjs1.gif" width="230" height="230"></td>
            <td><img src="https://github.com/JunwooParkSaribu/FreeTrace/blob/main/tmps/trjs2.gif" width="285" height="230"></td>
        </tr>  
</table>


&nbsp;&nbsp;<b>[FreeTrace](https://doi.org/10.64898/2026.01.08.698486)</b> infers individual trajectories from time-series images with reconnection of the detected particles under fBm.</br>

<h3> Contact person </h3>

<junwoo.park@sorbonne-universite.fr>

<h3> Contributors </h3>

> If you use this software, please cite it as below. </br>
```
@article {Park2026.01.08.698486,
	author = {Park, Junwoo and Sokolovska, Nataliya and Cabriel, Cl{\'e}ment and Kobayashi, Asaki and Corsin, Enora and Garcia Fernandez, Fabiola and Izeddin, Ignacio and Min{\'e}-Hattab, Judith},
	title = {Novel estimation of memory in molecular dynamics with extended and comprehensive single-molecule tracking software: FreeTrace},
	elocation-id = {2026.01.08.698486},
	year = {2026},
	doi = {10.64898/2026.01.08.698486},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2026/01/10/2026.01.08.698486},
	eprint = {https://www.biorxiv.org/content/early/2026/01/10/2026.01.08.698486.full.pdf},
	journal = {bioRxiv}
}
```
<br>
