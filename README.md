# Inside-out cross-covariance for multivariate GPs

This package implements Bayesian hierarchical models for fitting GPs with IOX cross-covariance. 

![Simulated data from IOX](https://github.com/mkln/spiox-paper/figures/prior_sample_3.png)

Noise-free trivariate GP using IOX.


![Cross-covariance](https://github.com/mkln/spiox-paper/figures/cij_plot.png)

Cross-covariance between two variables, for varying spatial range (left) or smoothness (right) of one of them relative to the other.


## Install

In R, `devtools::install_github("mkln/spiox")` to compile from source.

Some sampling algorithms implemented in `spiox` require SuperLU. 

- On Ubuntu, `sudo apt install libsuperlu-dev`.
- On Mac, [install superlu from Homebrew](https://formulae.brew.sh/formula/superlu). The `Makevars` file in `spiox` assumes a Homebrew install. 