# MRspecNET

This repository contains a collection of Python codes to set up and train a convolutional neural network for denoising in magnetic resonance spectroscopy (MRS). The network is trained using a set of simulated metabolite spectra, which are loaded from a file, randomly varied and decorated with Gaussian noise. This provides a pair of noisy (simulated) data and ground truth. Besides the metabolite basis sets the simulation tool offers the opportunity to add a macromolecular baseline to the simulated spectrum, which is commonly present in the human brain (for certain MRS sequences).

![an example of a noisy spectrum and its denoised version](https://github.com/frank-roesler/MRspecNET/blob/main/Figure_1.png)

The main components of our framework are:
* `MMBG_basis(PATH,kwargs)`: a class that loads and stores the basis sets for the macromolecular baseline located at `PATH`. Accepts a dict of keyword arguments that define how the basis sets are combined into a single spectrum;
* `Metab_basis(PATH,kwargs)`: a class that loads and stores the basis sets for the metabolites located at `PATH`. Accepts a dict of keyword arguments that define how the basis sets are combined into a single spectrum;
* `Simulator(metab_basis,mmbg_basis)`: a class that contains the main simulation tool. The method `Simulator.simulate()` combines the data from `MMBG_basis` and `Metab_basis` into a realistic MR spectrum and adds a degree of random variation;
* `MRspecNET(kernel_size,n_channels_layers)`: The neural net (a convolutional autoencoder) used for denoising. Written in PyTorch;
* `train.py`: The main training script. Adds noise to simulated spectra, combines them into a batch and performs stochastic gradient minimization with MSE Loss.

The concentration values we use to generate realistic simulated spectra are those found in [Hoefemann, Maike, et al. "Parameterization of metabolite and macromolecule contributions in interrelated MR spectra of human brain using multidimensional modeling." NMR in Biomedicine 33.9 (2020): e4328.]

Any comments or queries are welcome at https://frank-roesler.github.io/contact/
