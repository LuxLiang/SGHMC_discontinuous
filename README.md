# Non-asymptotic convergence analysis of the stochastic gradient Hamiltonian Monte Carlo algorithm with discontinuous stochastic gradient with applications to training of ReLU neural networks

This repository is the official implementation of "Non-asymptotic convergence analysis of the stochastic gradient Hamiltonian Monte Carlo algorithm with discontinuous stochastic gradient with applications to training of ReLU neural networks". 

Authors: Luxu LIANG, Ariel Neufeld, and Ying Zhang.

Abstract: In this paper, we provide a non-asymptotic analysis of the convergence of the stochastic gradient Hamiltonian Monte Carlo (SGHMC) algorithm to a target measure in Wasserstein-1 and Wasserstein-2 distance. Crucially, compared to the existing literature on SGHMC, we allow its stochastic gradient to be discontinuous.  This allows us to provide explicit upper bounds, which can be controlled to be arbitrarily small, for the expected excess risk of non-convex stochastic optimization problems with discontinuous stochastic gradients, including, among others, the training of neural networks with ReLU activation function. To illustrate the applicability of our main results, we consider numerical experiments on quantile estimation and on several optimization problems involving ReLU neural networks relevant in finance and artificial intelligence.

## Dependencies:
- Python 3.6
- Pytorch 1.8.0 + cuda
- scikit-learn

## Guide

This repository contains three applications in finance: the multi-period portfolio optimization, transfer learning in the multi-period portfolio optimization, and the insurance claim prediction.

**Quantile Estimation** (Section 4.1)

Please refer to the folder named Quantile Estimation. The numerical codes and results are in Quantile_Estimation.ipynb and Quantile_Estimation_Supplement.ipynb.

**Solving regularized optimization problems using neural network** (Section 4.2)

*Transfer Learning* (Section 4.2.1)

Please refer to the folder Transfer Learning. Excute model.py file.

*Hedging under asymmetric risk* (Section 4.2.2)

Please refer to the folder Hedging. Excute run_BS.sh and the results are summarized in plot_Results.ipynb.

*Real-world datasets* (Section 4.2.3)

Please refer to the folder named Regression and classification. Excute run_main.sh file for training and use visualization.ipynb to visualize.
