#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 12:11:28 2021

@author: Your names and student numbers
"""

#import packages

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

 
# n denotes the sample size 
n = 1000


# we simulate two types of features
half_sample=int(n/2)
x1 = np.random.multivariate_normal([-0.5, 1], [[1, 0.7],[0.7, 1]], half_sample)
x2 = np.random.multivariate_normal([2, -1], [[1, 0.7],[0.7, 1]], half_sample)
simulated_features = np.vstack((x1, x2)).astype(np.float64)


# the underlying value of beta in the simulation; the value we want to retrieve in the estimation procedure
beta_star=np.array([0.2,-0.8])



#The logistic function
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
plt.scatter(simulated_features[:, 0], simulated_features[:, 1], c = simulated_labels, alpha = .5)
plt.show()




#Skeleton for function Newton-Raphson for logistic regression

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
#from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# predicted probabilities
p_pred = logistic(simulated_features @ beta)

# predicted classes
#y_pred = (p_pred >= 0.5).astype(int)

#cm = confusion_matrix(simulated_labels, y_pred)

#disp = ConfusionMatrixDisplay(confusion_matrix=cm)
#disp.plot(cmap='Blues', values_format='d')
#plt.title("Confusion Matrix")
#plt.show()


## Simulation study
S=1000
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







