# Nonlinear-Tructated-GCR
Optimization, Iterative Algorithm, Numerical Analysis
Under Review in SIAM Journal On Optimization

## Abstract: 
This paper develops a new class of nonlinear acceleration algorithms based on extending conjugate residual-type procedures from linear to nonlinear equations. The main algorithm has strong similarities with Anderson acceleration as well as with inexact Newton methods - depending on which variant is implemented. We prove theoretically and verify experimentally, on a variety of problems from simulation experiments to deep learning applications, that our method is a powerful accelerated iterative algorithm.

## Matlab:
Usage:

- To run the code, run `run_me_first.m` first. To reproduce the experiments in Figure 5.1-5.3 of the paper, go to `scripts` folder.

Contents:
- `src` folder contains implementations of baselines and NLTGCR with nonlinear, linear, and adaptive update.
- `scripts` folder contains experiments of the Bratu's problem (Section 5.1) and the Lennard-Jones problem (Section 5.2).
- `problem` folder contains functions to compute the gradient and cost of the Bratu's problem and the Lennard-Jones problem at a given point.
- `line_search` folder contains auxilary functions for line search in baselines.

## Python:
Contents:
- `test_nltgcr_demo.py` solves a linear system. The convergence is identical to CG, which verifies the correctness of implementation. 
- `main.py` is for baselines inlcuding SGD, Nesterov, Adam. main_nltgcr.py is for our method.


## Paper:
[NLTGCR: A class of Nonlinear Acceleration Procedures based on Conjugate Residuals](https://arxiv.org/abs/2306.00325)