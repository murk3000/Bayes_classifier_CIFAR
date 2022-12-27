import sys
import os

# this is the directory to cifar10.py (which is in a folder called muneeb_libs)
sys.path.append(os.path.abspath('D:\\Grad\\Period 1\\Intro to PR and ML'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal # to compute pdf of normal and multivariate normal dist
from scipy.stats import bootstrap
from random import random
from tqdm import tqdm # Measure time taken / esitmated time 

from muneeb_libs import cifar10 # Created a library to load data / compute accuracy / etc



def cifar10_color(X, size):
    """Returns the input X as downsized feature vector"""
    from skimage.transform import resize

    return resize(X, [X.shape[0], 3, *size], anti_aliasing=True, preserve_range=True)
    # return(np.mean(np.mean(X, axis=2), axis=2))
    

def cifar10_naivebayes_learn(Xp, Y):
    """
    Learn the parameters of a Naive Bayes estimate given the data Xp

    Parameters
    ----------
    Xp : List of ndarry 
        Feature vector to compute mean and sigma for.
    Y : List of Labels
        Vector to compute priors from.

    Returns
    -------
    Tuple of ndarray
        Returns Mean and Standard Deviation for each column in Xp and the Prior probability of Xp being Y.

    """
    y_unique = set(Y)
    
    mu_X = [None]*len(y_unique)
    sigma_X = [None]*len(y_unique)
    p_X = [None]*len(y_unique)
    
    for y in y_unique:
        mu_X[y] = np.mean([Xp[i] for i in range(Xp.shape[0]) if Y[i]==y], axis=0)
        # # The Cov function returns a covariance variance matrix. Since the assumption in Naive Bayes is that each
        # # parameter is independent, you can just take the variances of each feature i.e. the diagonal elements
        # sigma_X[y] = np.sqrt(np.diagonal(np.cov([Xp[i] for i in range(Xp.shape[0]) if Y[i]==y], rowvar = False)))
        sigma_X[y] = np.std([Xp[i] for i in range(Xp.shape[0]) if Y[i]==y])
        p_X[y] = np.mean([y==i for i in Y])
        
    # return mu_X, sigma_X, p_X
    return (np.array(mu_X), np.array(sigma_X), np.array(p_X))

def calc_norm_prob(x, mu, sigma, f=norm):
    """Returns the pdf value (likelihood) at x for a distribution f having parameters mu and sigma"""
    
    return f(mu,sigma).pdf(x)


def cifar10_classifier_naivebayes(x, mu, sigma, p): 
    """Returns the index (i.e. the class) with the maximum likelihood*prior estimate"""
    norm = [calc_norm_prob(x, mu[i], sigma[i]) for i in range(mu.shape[0])]
    lkhd = np.prod(norm, axis=1) 

    return np.argmax(lkhd*p) # you don't need to divide by the normalizing factor as we're just interested 
                             # at the value that maximizes the posterior, and the posterior is proportional 
                             # to the likelihood * prior

def cifar10_bayes_learn(Xf, Y):
    """
    Learn the parameters of a Bayes estimate given the data Xp

    Parameters
    ----------
    Xf : List of ndarry 
        Feature vector to compute mean and sigma for.
    Y : List of Labels
        Vector to compute priors from.

    Returns
    -------
    Tuple of ndarray
        Returns Mean and Covariance-Variance Matrix for each column in Xf and the Prior probability of Xf being Y.
    """
    
    y_unique = set(Y)
    
    mu_X = [None]*len(y_unique)
    sigma_X = [None]*len(y_unique)
    p_X = [None]*len(y_unique)
    
    for y in y_unique:
        mu_X[y] = np.mean([Xf[i] for i in range(Xf.shape[0]) if Y[i]==y], axis=0)
        sigma_X[y] = np.cov([Xf[i] for i in range(Xf.shape[0]) if Y[i]==y], rowvar=False)
        p_X[y] = np.mean([y==i for i in Y])
        
    return (np.array(mu_X), np.array(sigma_X), np.array(p_X))
    
def cifar10_classifier_bayes(x, mu, sigma, p): 
    """Returns the index (i.e. the class) with the maximum likelihood*prior estimate"""
    norm = [calc_norm_prob(x, mu[i], sigma[i], f=multivariate_normal) for i in range(mu.shape[0])]
    lkhd = np.array(norm)
    
    # return lkhd
    return np.argmax(lkhd*p) # you don't need to divide by the normalizing factor as we're just interested 
                             # at the value that mazimizes the posterior, and the posterior is proportional 
                             # to the likelihood * prior

if __name__ == '__main__':

    X,Y, X_test, Y_test, labeldict, label_names = cifar10.load_cifar10(5)
    
    # Randomly show some images   
    for i in range(X.shape[0]):
        # Show some images randomly
        if random() > 0.99999:
            plt.figure(1);
            plt.clf()
            plt.imshow(X[i])
            plt.title(f"Image {i} label={label_names[Y[i]]} (num {Y[i]})")
            plt.pause(1)
            
    X = X.transpose(0,3,1,2) # transpoe the array to have the RGB channel first and then the 32x32 matrix to easily reduce to RGB newsize x newsize
    X_test = X_test.transpose(0,3,1,2)
    
    img_sizes = [1,2,3,4,5,6,7,8,9,10,11,12] # the length of the square matrix the images should be reduced to (the RGB channel stays as is)
    nb_acc = [None]*len(img_sizes)
    b_acc = [None]*len(img_sizes)
    
    for indx, i in enumerate(img_sizes):
        X_ = cifar10_color(X, [i,i]).reshape(X.shape[0], 3*i*i)
        X_test_ = cifar10_color(X_test, [i,i]).reshape(X_test.shape[0], 3*i*i)
        
        
        mu_X, sigma_X, p_X = cifar10_naivebayes_learn(X_, Y)
        Y_pred = [cifar10_classifier_naivebayes(X_test_[i], mu_X, sigma_X, p_X) for i in tqdm(range(10000))]
        print(f'\nNaive Bayes Accuracy {i}: {cifar10.class_acc(Y_pred, Y_test)}\n\n')
        
        nb_acc[indx] = cifar10.class_acc(Y_pred, Y_test, is_int=True)
        
        mu_X, sigma_X, p_X = cifar10_bayes_learn(X_, Y)
        Y_pred = [cifar10_classifier_bayes(X_test_[i], mu_X, sigma_X, p_X) for i in tqdm(range(10000))]
        print(f'\nBayes Accuracy {i}: {cifar10.class_acc(Y_pred, Y_test)}\n\n')    
        
        b_acc[indx] = cifar10.class_acc(Y_pred, Y_test, is_int=True)
        
    plt.figure(1)
    plt.clf()
    plt.plot(img_sizes, b_acc, label='bayes')
    plt.plot(img_sizes, nb_acc, label='naive bayes')
    plt.legend()
    plt.show()
    
    
    
    
    