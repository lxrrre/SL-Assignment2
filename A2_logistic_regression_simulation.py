#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 12:11:28 2021

@author: Your names and student numbers
"""

#import packages

import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

 
# n denotes the sample size 
n = 1000


# we simulate two types of features
np.random.seed(0) # imposing seed for reproducibility
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
        p=logistic(np.dot(features, beta))
        W = np.diag(p*(1-p))
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

beta=logistic_regression_NR(simulated_features, simulated_labels,1000, 1e-5)
print(beta)

## Simulation study
S=100
mle_list=np.zeros((S,2))
for i in range(S):
    #generate labels y for every simulation
    simulated_labels = logistic_simulation(simulated_features, beta_star)
    #compute the MLE for every simulation
    mle_list[i,:]=logistic_regression_NR(simulated_features, simulated_labels, 1000, 1e-3)

#compute the means of estimated parameters beta_1 and beta_2
beta_mean=np.mean(mle_list, axis=0)
print(beta_mean)
#make a histogram for the MLE of beta_1 and beta_2
plt.hist(mle_list, bins=100)
plt.show()







# Graph experiments
plt.figure(figsize=(12,8))
p_pred = logistic(simulated_features @ beta)
plt.hist(p_pred[simulated_labels == 0], bins=30, alpha=0.5, label="Class 0")
plt.hist(p_pred[simulated_labels == 1], bins=30, alpha=0.5, label="Class 1")
plt.xlabel("predicted probability")
plt.ylabel("frequences")
plt.legend()
plt.show()

plt.figure(figsize=(12,8))
plt.scatter(simulated_features[:, 0], simulated_features[:, 1], c=simulated_labels, alpha=0.5, label="Dati")

# Decision boundary
x_vals = np.linspace(np.min(simulated_features[:, 0]), np.max(simulated_features[:, 0]), 100)
y_vals = - (beta[0] / beta[1]) * x_vals

plt.plot(x_vals, y_vals, 'r-', label="Decision boundary")
plt.legend()
plt.show()

# confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# predicted probabilities
p_pred = logistic(simulated_features @ beta)

# predicted classes
def logistic_forecast(features,beta):
    signal_hat = np.dot(features, beta)
    y_hat=np.sign(signal_hat)
    y_hat[y_hat<0] = 0
    return y_hat

y_pred = logistic_forecast(simulated_features, beta)

cm = confusion_matrix(simulated_labels, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix")
plt.show()

def log_likelihood(features, target, beta):
    p = logistic(features @ beta)
    return np.sum(target * np.log(p) + (1 - target) * np.log(1 - p))

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

beta_seq=logistic_regression_NR_beta_seq(simulated_features, simulated_labels, 1000, 1e-5)

beta_1_vals = np.linspace(-0.05, 0.3, 100)
beta_2_vals = np.linspace(-1, 0.1, 100)
B1, B2 = np.meshgrid(beta_1_vals, beta_2_vals)
Z = np.array([log_likelihood(simulated_features, simulated_labels, np.array([b1, b2]), ) for b1, b2 in zip(np.ravel(B1), np.ravel(B2))])
Z = Z.reshape(B1.shape)

# Plot the contour plot
plt.figure(figsize=(8, 6))
plt.contourf(B1, B2, Z, levels=50, cmap='viridis')
plt.colorbar(label='Log-Likelihood')

# Overlay the Newton-Raphson sequence
beta_sequence = beta_seq[:, 0], beta_seq[:, 1]
plt.plot(beta_sequence[0], beta_sequence[1], marker='.', color='red', label='Newton-Raphson Path')
plt.plot(beta_star[0], beta_star[1], marker='*', color='blue', label='True minimum')
plt.legend()
plt.show()

# We observe that the optimal value found through NR method is the actual minimum of the log likelihood if we consider the simulated data
# The value beta* is the value we want to retrieve, the minimum of log likelihood with no noisy data.
# What happens is that by using a Montecarlo approach this noise is reduced and the mean of simulated values approaches the value beta*