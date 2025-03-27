import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("always", ConvergenceWarning) #to understand better the convergence

from sklearn.preprocessing import StandardScaler  #not required but useful, we can think about this

from sklearn.model_selection import StratifiedKFold

import pandas as pd




def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

dict1 = unpickle("./data/cifar-10-batches-py/data_batch_1")
dict2 = unpickle("./data/cifar-10-batches-py/data_batch_2")
dict3 = unpickle("./data/cifar-10-batches-py/data_batch_3")
dict4 = unpickle("./data/cifar-10-batches-py/data_batch_4")
dict5 = unpickle("./data/cifar-10-batches-py/data_batch_5")
test = unpickle("./data/cifar-10-batches-py/test_batch")
meta_data = unpickle("./data/cifar-10-batches-py/batches.meta")
label_names = meta_data["label_names"]


X_train = np.concatenate((dict1["data"],dict2["data"],dict3["data"],dict4["data"],dict5["data"]))
y_train = np.concatenate((dict1["labels"],dict2["labels"],dict3["labels"],dict4["labels"],dict5["labels"]))
X_test = test["data"]
y_test = test["labels"]

def data_to_image(x):
    return(x.reshape(3,32,32).transpose(1,2,0))

def plot_image(image, title=""):
    fig = plt.imshow(data_to_image(image))
    plt.title(title)
    fig.axes.set_axis_off()
    plt.show()

# as a verification that everything is working correctly, plot an image
#image_nr = 320
#plot_image(X_train[image_nr,:],label_names[y_train[image_nr]])

#the next 4 lines are to work with a smaller dataset since my (Lorenzo) pc is not so quick with these computations
X_train = X_train[:5000]
y_train = y_train[:5000]
X_test = X_test[:1000]
y_test = y_test[:1000]


#da inserire o meno, si può valutare --> ho notato che mettendola migliora il problema delle iterazioni nulle nella cross-validation:
#potrebbe esulare da quanto richiesto nell'assignment

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)



model = LogisticRegression(max_iter=2000, C=1e-3, verbose=1, solver='saga') #in scikit-learn the regularization parameter is set as the inverse of the lambda we use normally;
#note that we can set the parameter multiclass='ovr' or multiclass='multinomial', which distinguishes between a one vs the rest model and a truly multinomial one.
#here it's set to auto, but since the accuracy of the model is low i tried to set multinomial: no signficant improvements noted, probably auto already sets it to multinomial
#saga is the minimization algorithm more adapt for large datasets, such as this one (a lot features and samples)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#plot_image(X_test_small[250,:], label_names[y_pred[250]])

print("accuracy: {}".format(sum(y_pred==y_test)/len(y_test))) #quick assessment of the quality of the predictions

print("Number of iterations: {}".format(model.n_iter_))#it's useful to check the number of iterations

cv_model = LogisticRegressionCV(max_iter=2000, verbose=1, Cs = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2], cv=4, n_jobs=-1, solver='saga') #considering high values of C bring the optimization to not converge --> strong regularization required
cv_model.fit(X_train, y_train)
y_pred_cv = cv_model.predict(X_test)

print("CV accuracy: {}".format(sum(y_pred_cv==y_test)/len(y_test)))


print("Number of iterations, cv: {}".format(cv_model.n_iter_))
#n_iter è di dimensione n_fold x n_Cs
#anche con lo scaling continuo ad avere 0 (che non hanno senso) per valori di C grandi (ossia regolarizzazione limitata)
#per giunta questo problema compare solo con la cv: per lo stesso valore di C la regressione logistica normale va (ha un certo numero di iterazioni)

print("Best value of C established: {}".format(cv_model.C_))

"""
cv = StratifiedKFold(n_splits=4)

for i, (_, val_idx) in enumerate(cv.split(X_train, y_train)):
    print(f"Fold {i}: classi presenti = {np.unique(y_train[val_idx])}")

#questo for serve per assicurarsi che n_iter=0 NON dipenda dai fold, ma soltanto da alcuni valori di C

"""








