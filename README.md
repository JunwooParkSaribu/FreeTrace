
[![Static Badge](https://img.shields.io/badge/bioRxiv-FreeTrace-red)](https://doi.org/10.64898/2026.01.08.698486)
![Static Badge](https://img.shields.io/badge/%3E%3D3.10-1?style=flat&label=Python&color=blue)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13336251.svg)](https://doi.org/10.5281/zenodo.13336251)
![GitHub License](https://img.shields.io/github/license/JunwooParkSaribu/FreeTrace)
[![PyPI](https://img.shields.io/pypi/v/freetrace)](https://pypi.org/project/freetrace/)

## FreeTrace

**Data privacy:** FreeTrace runs entirely on your local machine. No data is transmitted to any external server. Your images, localizations, and trajectories never leave your computer.

> [!IMPORTANT]
> Requirements </br>
> - Windows(10/11) / GNU/Linux(Debian/Ubuntu) / MacOS(Sequoia/Tahoe)</br>
> - Python3.10 &#8593;</br>
> - GPU & Cuda12 on GNU/Linux with pre-trained [models](https://github.com/JunwooParkSaribu/FreeTrace/blob/main/FreeTrace/models/README.md) (recommended)</br>


> [!TIP]
> **Windows standalone installer (GPU):** Download the self-contained FreeTrace C++ installer with full GPU support — no compilation needed (only NVIDIA GPU driver required):
> **[Download FreeTrace for Windows (GPU)](https://psilo.sorbonne-universite.fr/public.php/dav/files/XmTL99cCx4iXDdH/?accept=zip)**
> **Note:** The standalone installer uses cuDNN 9.2, which does **not** support RTX 5000 series (Blackwell) or newer GPUs. On unsupported GPUs, H (Hurst exponent) estimation will silently produce incorrect values. Supported GPUs: RTX 2000/3000/4000 series and equivalent. <!-- Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15 -->
>
> FreeTrace is available on PyPI and can be installed with:
> ```
> pip install freetrace
> ```
> **C++ version**: A standalone C++ port of FreeTrace (no Python/GPU dependency) is available at [FreeTrace_cpp](https://github.com/JunwooParkSaribu/FreeTrace_cpp). It produces identical results and runs **2–7x faster** for tracking. Recommended for integration, automation, and AI-agent workflows. See the [pipeline architecture](https://github.com/JunwooParkSaribu/FreeTrace_cpp#pipeline-architecture) for details on GPU/CPU dispatch and fBm mode branching.

> [!NOTE]
> - PRE-REQUISITE: pre-installation and compilation, check [Tutorial](https://colab.research.google.com/github/JunwooParkSaribu/FreeTrace/blob/main/tutorial.ipynb) in Colab. </br>
> - Check [compatibilities](https://github.com/JunwooParkSaribu/FreeTrace/blob/main/FreeTrace/models/README.md) of Python and Tensorflow to run FreeTrace with source code.</br>
> - Without GPU, FreeTrace is slow if it infers under fractional Brownian motion.</br>
> - Current version is stable with python 3.10 / 3.11 / 3.12</br>
> - The updates from version 1.6 may include Claude-generated code, and any code modified by Claude-like will be explicitly marked with comments. This aims to make FreeTrace easy to use, such as by providing a generated GUI.  <em><strong>Version 1.5.19</em></strong> is the last version completely maintained only by the author, without any modification via Claude-like. The result affecting modification will be noticed via <em><strong>Major update</em></strong> tag.</br>


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
