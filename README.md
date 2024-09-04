[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13336251.svg)](https://doi.org/10.5281/zenodo.13336251)

# FreeTrace
Single Particle Tracking from high resolution video.<br>

FreeTrace rebuilds the trajectories of single particles which follow fractional brownian motion in nature from super-resolution image sequences. FreeTrace is consist of two base algorithms. It first localizes particles in sub-pixel levels from images. the precision can be tens of nanometers for the super-resolution microscopy images. It then builds a temporal network of potential paths with reconnectable positions of particles in next few frames, and selects the local optimal path to reconstruct the entire trajectory. FreeTrace performs iteratively the above main algorithms for each single particle and provides reconstructed 2 or 3 dimensional time-series trajectory data for the entire images.