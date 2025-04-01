"""
@author:  Bracci Lorenzo - Federica Musazzi - Schiavi Francesco
"""

#import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#read the csv file
#the must be in the same directory of the script
df = pd.read_csv('mnist.csv')

#make dataframe into numpy array
df.to_numpy()
y_labels_data=df['label'].to_numpy()
df_xdata=df.drop(columns='label')
x_features_data=df_xdata.to_numpy()
print("\nShape of the feature data (samples, pixels):", x_features_data.shape)

# we will only use the zeros and ones in this empirical study
# so we filter only zeros and ones
y_labels_01 = y_labels_data[np.where(y_labels_data <=1)[0]]
x_features_01 = x_features_data[np.where(y_labels_data <=1)[0]]
print(f"Number of 0/1 samples: {len(y_labels_01)}")

# create training set
n_train=100
y_train=y_labels_01 [0:n_train]
x_train=x_features_01[0:n_train]
print(f"\nTraining set size: {n_train} samples")

#create test set
n_total=y_labels_01.size
y_test=y_labels_01 [n_train:n_total]
x_test=x_features_01[n_train:n_total]
print(f"Test set size: {n_total - n_train} samples")

## Here we plot some handwritten digits
plt.figure(figsize=(25,5))
for index, (image, label) in enumerate(zip(x_train[5:10], y_train[5:10])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)
    plt.title('Label: %i\n' % label, fontsize = 20)
    
plt.show()

#The logistic function
def logistic(x):
    return 1 / (1 + np.exp(-x))


#given an estimated parameter for the logistic model we can obtain forecasts 
#for our test set with the following function

def logistic_forecast(features,beta):    
    signal_hat = np.dot(features, beta)
    y_hat=np.sign(signal_hat)
    y_hat[y_hat<0] = 0
    return y_hat
 

 
#computes the prediction error by comparing the predicted and the observed labels in the test set
def prediction_accuracy(y_predicted,y_observed):
    errors= np.abs(y_predicted-y_observed)
    total_errors=sum(errors)
    acc=1-total_errors/len(y_predicted)    
    return acc



#dimension of the problem
p=x_train.shape[1]
print(f"\nNumber of features (pixels per image): {p}")

#Compute the ranks of the matrices X and X^T
print("Rank of X_train matrix:", np.linalg.matrix_rank(x_train))
print("Rank of X_train transposed:", np.linalg.matrix_rank(x_train.T))
# we see that the rank of the matrix X and X^T is 100 (that is the number of samoles)
 
def logistic_regression_NR(features, target, num_steps, tolerance):
    #initialization of beta
    beta = np.zeros(features.shape[1])
    for step in range(num_steps): #repeat the iterative procedure
        # compute the weights matrix using the current beta
        p=logistic(features@beta) #computing of probabilities
        W = np.diag(p*(1-p)) #weight matrix
        print(np.linalg.matrix_rank(W))
        # computing the gradient of log-likelihood: X.T (y-p)
        gradient = features.T @ (target - p)
        if np.linalg.norm(gradient) > tolerance :
            #computing the hessian of log-likelihood -X.T W X
            Hessian=-features.T@W@features
            # Update beta according to Newton-Raphson procedure
            beta=beta - np.linalg.solve(Hessian, gradient)
            # solving a linear sistem instead of matrix inversion for improving computational efficiency
        else:
            break
    return beta

# beta=logistic_regression_NR(x_train, y_train, 1000, 1e-5)
# This line of code gives an error as the Hessian matrix is singular if we don't introduce any regularization

# Regularization parameter
lambda_0=1


def logistic_regression_NR_penalized(features, target, num_steps, tolerance, l):
    beta = np.zeros(features.shape[1]) # inizialize the parameter
    for step in range(num_steps):
        # compute the weights matrix using the current beta
        p = logistic(features @ beta) # probabilities
        W = np.diag(p * (1 - p)) # weight matrix
        # computing the gradient of regularized log-likelihood: X.T (y-p)+2*lambda*beta
        gradient = -features.T @ (target - p)+2*l*beta
        if np.linalg.norm(gradient) > tolerance:
            # compute hessian -X.T W X +2*lambda*I
            Hessian = features.T @ W @ features+2*l*np.eye(features.shape[1])
            # Update beta according to Newton-Raphson procedure
            beta = beta - np.linalg.solve(Hessian, gradient)
        else:
            break
    return beta

# Estimator for beta
print("\nTraining model with Newton-Raphson (regularized)...")
beta=logistic_regression_NR_penalized(x_train, y_train, 1000, 1e-5, lambda_0)
print("Model training completed.")

# Predictions for labels    # prediction accuracy (99.48%)
y_hat=logistic_forecast(x_test,beta)
acc = prediction_accuracy(y_test,y_hat)
print(f"\nPrediction Accuracy: {acc*100:.2f}%")

# Plotting a confusion matrix that visually shows misclassifications
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_hat)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', values_format='d')
disp.ax_.set_xlabel('Predicted Label', fontsize=12)  
disp.ax_.set_ylabel('True Label', fontsize=12)     
plt.title("Confusion Matrix", fontsize=15)
plt.show()

# Plotting probability bars for 20 pictures in the dataset.
# They show the probability of being in each class given by the model
offset = 200
n_images = 20

images_per_row = 10
probabilities = logistic(np.dot(x_test, beta))


def draw_bars(ax, probabilities, y_hat, label):
    myplot = ax.bar(range(2), (1-probabilities, probabilities))
    ax.set_ylim([0, 1])
    ax.set_xticks(range(2))
    ax.set_xticklabels(['0', '1'])                 
    ax.set_xlabel('Class', fontsize=10)              
    ax.set_ylabel('Probability', fontsize=10)       

    label_predicted = y_hat
    color = "green" if label == label_predicted else "red"
    myplot[int(y_hat)].set_color(color)


import math

n_rows = 2 * math.ceil(n_images / images_per_row)
_, axs = plt.subplots(n_rows, images_per_row, figsize=(3 * images_per_row, 3 * n_rows))
row = 0
col = 0
for i in range(n_images):
    axs[2 * row, col].imshow(x_test[offset + i, :].reshape((28, 28)), cmap="gray")
    axs[2 * row, col].set_title(int(y_test[offset + i]))
    axs[2 * row, col].axis("off")

    draw_bars(axs[2 * row + 1, col],probabilities[offset+i], y_hat[offset+i], y_test[offset + i])

    col += 1
    if col == images_per_row:
        col = 0
        row += 1
plt.show()

