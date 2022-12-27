import pickle # To load data from CIPHAR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import random,  sample
from sklearn.decomposition import PCA # To visualize the data and see feature distinction 

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

def load_cifar10(batches):
	datadict = [None]*batches

	# Load each file over a loop
	for i in range(batches):
		datadict[i] = unpickle('cifar-10-batches-py/data_batch_'+str(i+1))
	datadic_test = unpickle('cifar-10-batches-py/test_batch')
	#datadict = unpickle('/home/kamarain/Data/cifar-10-batches-py/test_batch')

	# Concatenate the dataset into training and test sets
	X = np.concatenate([datadict[i]["data"] for i in range(batches)])
	Y = np.concatenate([datadict[i]["labels"] for i in range(batches)])
	X_test = datadic_test["data"]
	Y_test = datadic_test["labels"] 

	print(X.shape)

	# Reshape the dataset to be able to view the images
	labeldict = unpickle('cifar-10-batches-py/batches.meta')
	label_names = labeldict["label_names"]
	X = X.reshape(batches*10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8").astype('int')
	Y = np.array(Y)

	X_test = X_test.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
	Y_test = np.array(Y_test)

	return (X,Y, X_test, Y_test, labeldict, label_names)

# PCA graphing to try and visualize any boundary between X and Y
def nd_image_2d(X, Y, f=PCA, randomize=0):
    n = 2
    X_ = pd.DataFrame(X)
    Y_ = pd.DataFrame(data=Y, columns=['label'])
    if randomize>0:
        indx = sample(range(X.shape[0]), randomize)
        X_ = X_.loc[indx].reset_index()
        Y_ = Y_.loc[indx].reset_index()

    pca = f(n_components = n)
    pc = pca.fit_transform(X_)
    
    df = pd.concat([pd.DataFrame(data=pc, columns=['pc_'+str(i+1) for i in range(n)]), Y_], axis = 1).loc[sample(range(len(X_)), 500)].reset_index()
    
    fig = plt.figure()
    ax = fig.add_subplot()
    groups = df.groupby('label')
    for name, group in groups:
        ax.scatter(group.pc_1, group.pc_2, marker = 'o', label=name)
    ax.legend()
    return df

def nd_image_3d(X, Y, f=PCA, randomize=0):
    n = 3
    X_ = pd.DataFrame(X)
    Y_ = pd.DataFrame(data=Y, columns=['label'])

    if randomize > 0:
        indx = sample(range(X.shape[0]), randomize)
        X_ = X_.loc[indx].reset_index()
        Y_ = Y_.loc[indx].reset_index()
        
    pca = f(n_components = n)
    pc = pca.fit_transform(X_)
    
    df = pd.concat([pd.DataFrame(data=pc, columns=['pc_'+str(i+1) for i in range(n)]), Y_], axis = 1).loc[sample(range(len(X_)), 500)].reset_index()
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    groups = df.groupby('label')
    for name, group in groups:
        ax.scatter(group.pc_1, group.pc_2, group.pc_3, marker = 'o', label=name)
    ax.legend()
    return df
    
def class_acc(pred, gt, is_int=False):
    """Returns the accuracy between categorical lists"""
    acc_mat = [i==j for i,j in zip(pred, gt)]
    if is_int:
        return round(np.mean(acc_mat)*100,2)
    return f'{round(np.mean(acc_mat)*100,2)}%'

def cifar10_classifier_random(x):
    """Returns a random class between 1 and 9 for any input"""
    return int(random()*10)

