# Nonlinear-Tructated-GCR
Optimization, Iterative Algorithm, Numerical Analysis

## Matlab:
 To run the code, run 'run_me_first.m' first. 
src folder contains implementations of baselines and nltgcr
 To run an experiment, go to scripts folder

## Matlab_SIMAX:
Usage:

- To run the code, run `run_me_first.m` first. To reproduce the experiments in Figure 5.1-5.3 of the paper, go to `scripts` folder.

Contents:
- `src` folder contains implementations of baselines and NLTGCR with nonlinear, linear, and adaptive update.
- `scripts` folder contains experiments of the Bratu's problem (Section 5.1) and the Lennard-Jones problem (Section 5.2).
- `problem` folder contains functions to compute the gradient and cost of the Bratu's problem and the Lennard-Jones problem at a given point.
- `line_search` folder contains auxilary functions for line search in baselines.

## Python:
 Each folder contains one deep learning application. main.py is for baselines inlcuding SGD, Nesterov, Adam. main_nltgcr.py is for our method.
test_nltgcr_demo.py solves a linear system. The convergence is identical to CG, which verifies the correctness of implementation. 
