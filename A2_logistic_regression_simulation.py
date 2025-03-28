#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 12:11:28 2021

@author: Bracci Lorenzo - Musazzi Federica - Schiavi Francesco
"""

#import packages

import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

 
# n denotes the sample size 
n = 1000


# we simulate two types of features
np.random.seed(1) # imposing seed for reproducibility

half_sample=int(n/2)
x1 = np.random.multivariate_normal([-0.5, 1], [[1, 0.7],[0.7, 1]], half_sample)
x2 = np.random.multivariate_normal([2, -1], [[1, 0.7],[0.7, 1]], half_sample)
simulated_features = np.vstack((x1, x2)).astype(np.float64)


# the underlying value of beta in the simulation; the value we want to retrieve in the estimation procedure
beta_star=np.array([0.2,-0.8])



# The logistic function
def logistic(x):
    return 1 / (1 + np.exp(-x))


# Simulate the labels
def logistic_simulation(features,beta):    
    signal = np.dot(features, beta)
    p=logistic(signal)
    y= np.array([np.random.binomial(1, p[i] ) for i in range(n)])
    return y

simulated_labels = logistic_simulation(simulated_features, beta_star)

#### Scatter plot of the features and corresponding labels
plt.figure(figsize=(12,8))
scatter = plt.scatter(simulated_features[:, 0], simulated_features[:, 1], c=simulated_labels, alpha=0.5)

handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.cmap(scatter.norm(0)), markersize=10, label='0'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.cmap(scatter.norm(1)), markersize=10, label='1')
]

plt.legend(handles=handles, title="Labels")
plt.show()



# Newton-Raphson for logistic regression

def logistic_regression_NR(features, target, num_steps, tolerance):
    #initialization of beta
    beta = np.zeros(features.shape[1])

    for step in range(num_steps):
        # compute the weights matrix using the current beta
        p=logistic(np.dot(features, beta)) # probabilities
        W = np.diag(p*(1-p)) # weight matrix
        # computing the gradient of log-likelihood: X.T (y-p)
        gradient = features.T @ (target - p)
        if np.linalg.norm(gradient) > tolerance :
            #compute hessian -X.T W X
            Hessian=-features.T@W@features
            # Update beta according to Newton-Raphson procedure
            beta=beta - np.linalg.solve(Hessian, gradient)
        else:
            break
    return beta

# computing beta estimate
beta=logistic_regression_NR(simulated_features, simulated_labels,1000, 1e-10)
print(beta) # [ 0.16787627 -0.82899279]
print(np.linalg.norm(beta-beta_star)) # 0.043272577648446775

# MONTECARLO SIMULATION
## Simulation study
for n in {100, 1000}:
    # n denotes the sample size
    # we simulate two types of features
    np.random.seed(1) # imposing seed for reproducibility
    half_sample=int(n/2)
    x1 = np.random.multivariate_normal([-0.5, 1], [[1, 0.7],[0.7, 1]], half_sample)
    x2 = np.random.multivariate_normal([2, -1], [[1, 0.7],[0.7, 1]], half_sample)
    simulated_features = np.vstack((x1, x2)).astype(np.float64)

    S=1000
    mle_list=np.zeros((S,2))
    for i in range(S):
        #generate labels y for every simulation
        simulated_labels = logistic_simulation(simulated_features, beta_star)
        #compute the MLE for every simulation
        mle_list[i,:]=logistic_regression_NR(simulated_features, simulated_labels, 1000, 1e-10)

    #compute the means of estimated parameters beta_1 and beta_2
    beta_mean=np.mean(mle_list, axis=0)
    print(beta_mean)
    print(np.linalg.norm(beta_mean-beta_star))
    #make a histogram for the MLE of beta_1 and beta_2
    plt.hist(mle_list, bins=100)
    plt.title( n)
    plt.show()
# n=1000
# beta=[ 0.19882683 -0.80498953]
# norm_diff=0.005125595266666119
# n=100
# beta=[ 0.19726609 -0.82673037]
# norm_diff=0.026869819196556278

# The following code is to plot the log likelihood as a function of beta components
# and to show the sequence of NR iterations until convergence for a single label simulation

# log-likelihood as function of features, labels and beta
def log_likelihood(features, target, beta):
    p = logistic(features @ beta)
    return np.sum(target * np.log(p) + (1 - target) * np.log(1 - p))

# logistic regression saving NR steps
def logistic_regression_NR_beta_seq(features, target, num_steps, tolerance):
    #initialization of beta
    beta = np.zeros(features.shape[1])
    beta_seq=[beta]
    for step in range(num_steps):
        # compute the weights matrix using the current beta
        p=logistic(np.dot(features, beta))
        W = np.diag(p*(1-p))
        # computing the gradient of log-likelihood: X.T (y-p)
        gradient = features.T @ (target - p)
        if np.linalg.norm(gradient) > tolerance :
            #compute hessian -X.T W X
            Hessian=-features.T@W@features
            # Update beta according to Newton-Raphson procedure
            beta=beta - np.linalg.solve(Hessian, gradient)
            beta_seq.append(beta)
        else:
            break
    return np.array(beta_seq)

# sequence of beta
beta_seq=logistic_regression_NR_beta_seq(simulated_features, simulated_labels, 1000, 1e-10)

# grid of points for log-likelihood plot
beta_1_vals = np.linspace(-0.05, 0.3, 100)
beta_2_vals = np.linspace(-1, 0.1, 100)
B1, B2 = np.meshgrid(beta_1_vals, beta_2_vals)
Z = np.array([log_likelihood(simulated_features, simulated_labels, np.array([b1, b2]), ) for b1, b2 in zip(np.ravel(B1), np.ravel(B2))])
Z = Z.reshape(B1.shape)

# Contour plot of log-likelihood as a function of beta
plt.figure(figsize=(8, 6))
plt.contourf(B1, B2, Z, levels=50, cmap='viridis')
plt.colorbar(label='Log-Likelihood')

# Overlay the Newton-Raphson sequence
beta_sequence = beta_seq[:, 0], beta_seq[:, 1]
plt.plot(beta_sequence[0], beta_sequence[1], marker='.', color='red', label='Newton-Raphson Path')
plt.plot(beta_star[0], beta_star[1], marker='*', color='blue', label='beta*')
plt.legend()
plt.show()

# We observe that the optimal value found through NR method is the actual minimum of the log likelihood if we consider the simulated data
# The value beta* is the value we want to retrieve, the minimum of log likelihood for n->inf.
# What happens is that by using a Montecarlo approach this noise is reduced and the mean of MLE values approaches the value beta*