import os
print(os.getcwd())

# %%

import numpy as np

import time

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("always", ConvergenceWarning) #to understand better the convergence

from sklearn.preprocessing import StandardScaler  #not required but useful, we can think about this


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

dict1 = unpickle("./SL-Assignment2/data/cifar-10-batches-py/data_batch_1")
dict2 = unpickle("./SL-Assignment2/data/cifar-10-batches-py/data_batch_2")
dict3 = unpickle("./SL-Assignment2/data/cifar-10-batches-py/data_batch_3")
dict4 = unpickle("./SL-Assignment2/data/cifar-10-batches-py/data_batch_4")
dict5 = unpickle("./SL-Assignment2/data/cifar-10-batches-py/data_batch_5")
test = unpickle("./SL-Assignment2/data/cifar-10-batches-py/test_batch")
meta_data = unpickle("./SL-Assignment2/data/cifar-10-batches-py/batches.meta")
label_names = meta_data["label_names"]


X_train = np.concatenate((dict1["data"],dict2["data"],dict3["data"],dict4["data"],dict5["data"]))
y_train = np.concatenate((dict1["labels"],dict2["labels"],dict3["labels"],dict4["labels"],dict5["labels"]))
X_test = test["data"]
y_test = test["labels"]

print("X_train dimensions: {}".format(X_train.shape))
print("y_train dimensions: {}".format(y_train.shape))
print("X_test dimensions: {}".format(X_test.shape))
print("y_test dimensions: {}".format(np.array(y_test).shape))

def data_to_image(x):
    return(x.reshape(3,32,32).transpose(1,2,0))

def plot_image(image, title=""):
    fig = plt.imshow(data_to_image(image))
    plt.title(title)
    fig.axes.set_axis_off()
    plt.show()

def plot_scores(C_values, scores, loss_type, img_name):
    plt.figure(figsize=(8, 5))
    plt.plot(C_values, mean_scores, marker='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Regularization parameter C (log scale)')
    plt.ylabel('Cross-validated '+loss_type+' accuracy (log scale)')
    plt.title('Log-log plot of CV '+loss_type+' vs. C')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(img_name)


# as a verification that everything is working correctly, plot an image
#image_nr = 320
#plot_image(X_train[image_nr,:],label_names[y_train[image_nr]])

# %%
#the next 4 lines are to work with a smaller dataset if we want to avoid computational overhead
X_train = X_train[:8000]
y_train = y_train[:8000]
X_test = X_test[:2000]
y_test = y_test[:2000]

# %%

#scaling is very useful to improve the overall performances

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)

time_start = time.time()
model_saga = LogisticRegression(max_iter=5000, C=1e-3, verbose=1, solver='saga') #in scikit-learn the regularization parameter is set as the inverse of the lambda we use normally;
#saga is the minimization algorithm (which should be) more adapt for large datasets, such as this one (a lot features and samples)
model_saga.fit(X_train, y_train)
time_end = time.time()
time_saga = time_end - time_start

print("saga time elapsed: {}".format(time_saga))

time_start = time.time()
model_lbgfs = LogisticRegression(max_iter=5000, C=1e-3, verbose=1) #here we use lbgfs instead of saga
model_lbgfs.fit(X_train, y_train)
time_end = time.time()
time_LBGFS = time_end - time_start

print("lbgfs time elapsed: {}".format(time_LBGFS))


print("lbgfs accuracy: {}".format(model_lbgfs.score(X_test, y_test)))
print("saga accuracy: {}".format(model_saga.score(X_test, y_test)))

print("Number of iterations, lbgfs: {}".format(model_lbgfs.n_iter_))
print("Number of iterations, saga: {}".format(model_saga.n_iter_))#it's useful to check the number of iterations


# %%
#we discard SAGA as it's considerably slower; first we use a larger range of Cs
cv_model = LogisticRegressionCV(max_iter=5000, verbose=1, Cs = np.linspace(1e-7, 1e-2, 10), cv=4, n_jobs=-1) #considering high values of C bring the optimization to not converge --> strong regularization required
cv_model.fit(X_train, y_train)



print("CV accuracy: {}".format(cv_model.score(X_test, y_test)))


print("Number of iterations, cv: {}".format(cv_model.n_iter_))
#n_iter is of dimensions n_fold x n_Cs


print("Best value of C established: {}".format(cv_model.C_[0]))

mean_scores = np.mean(cv_model.scores_[np.unique(y_train)[0]], axis=0) # Average accuracy on the 4 folds for each C value; #np.unique(y_train) takes all diverse values of y_train and generates an np-array
#since the score of the model doesn't depend on the class for multiclass='multinomial', we just take the first)
C_values = cv_model.Cs_

plot_scores(C_values, mean_scores, loss_type='accuracy', img_name='accuracy_C_large')

print("\nAverage accuracy over the folds for each C:")
for C, score in zip(C_values, mean_scores):
    print(f"C = {C:.2e} | Accuracy: {score:.4f}")

# %%
#here we refine the search for the best C by considering a shorter range of values

cv_model_2 = LogisticRegressionCV(max_iter=5000, verbose=1, Cs = np.linspace(1e-4, 1e-3, 10), cv=4, n_jobs=-1)
cv_model_2.fit(X_train, y_train)
y_pred_cv_2 = cv_model_2.predict(X_test)

print("CV accuracy: {}".format(cv_model_2.score(X_test, y_test)))

print("Number of iterations, cv: {}".format(cv_model_2.n_iter_))

print("Best value of C established: {}".format(cv_model_2.C_[0]))

mean_scores = np.mean(cv_model_2.scores_[np.unique(y_train)[0]], axis=0)

C_values = cv_model_2.Cs_

plot_scores(C_values, mean_scores, loss_type='accuracy', img_name='accuracy_C_short')

print("\nAverage accuracy over the folds for each C:")
for C, score in zip(C_values, mean_scores):
    print(f"C = {C:.2e} | Accuracy: {score:.4f}")



# %%
#here we consider the neg-log-likelihood
cv_model_nll = LogisticRegressionCV(max_iter=5000, verbose=1, Cs = np.linspace(1e-4, 1e-3, 10), cv=4, n_jobs=-1, scoring='neg_log_loss')
cv_model_nll.fit(X_train, y_train)

print("CV accuracy: {}".format(cv_model_nll.score(X_test, y_test)))

print("Number of iterations, cv: {}".format(cv_model_nll.n_iter_))

print("Best value of C established: {}".format(cv_model_nll.C_[0]))

mean_scores = np.mean(cv_model_nll.scores_[np.unique(y_train)[0]], axis=0)

C_values = cv_model_nll.Cs_

plot_scores(C_values, mean_scores, loss_type='logarithmic score', img_name='nll_C')

print("\nAverage accuracy over the folds for each C:")
for C, score in zip(C_values, mean_scores):
    print(f"C = {C:.2e} | Accuracy.: {score:.4f}")


# %% Use the entire dataset with C=7e-4

X_train_full = np.concatenate((dict1["data"], dict2["data"], dict3["data"], dict4["data"], dict5["data"]))
y_train_full = np.concatenate((dict1["labels"], dict2["labels"], dict3["labels"], dict4["labels"], dict5["labels"]))
X_test_full = test["data"]
y_test_full = test["labels"]


scaler = StandardScaler()
X_train_full = scaler.fit_transform(X_train_full)
X_test_full = scaler.transform(X_test_full)

# Train logistic regression on full data with best C
final_model = LogisticRegression(max_iter=2000, C=7e-4, solver='saga', verbose=1)
final_model.fit(X_train_full, y_train_full)

#
y_pred_full = final_model.predict(X_test_full)
accuracy_full = np.mean(np.array(y_pred_full) == np.array(y_test_full))

print("Final model test accuracy on full dataset: {:.3f}".format(accuracy_full))
print("Number of iterations: {}".format(final_model.n_iter_))


#%%
from sklearn.metrics import accuracy_score, log_loss


train_accuracy = accuracy_score(y_train, final_model.predict(X_train))
test_accuracy = accuracy_score(y_test, final_model.predict(X_test))


train_log_loss = log_loss(y_train, final_model.predict_proba(X_train))
test_log_loss = log_loss(y_test, final_model.predict_proba(X_test))

print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Train Log-Loss: {train_log_loss:.4f}")
print(f"Test Log-Loss: {test_log_loss:.4f}")


