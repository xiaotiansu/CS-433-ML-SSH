# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = 0
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def standardize(tx):
    """Standardize the original data set."""
    # replace abnormal data with nan
    tx[tx==-999] = np.nan
    # replace nan with median
    np.nan_to_num(tx, copy=False, nan=np.nanmedian(tx, axis=0))
    # use Z-score to standardize the data
    mu = np.mean(tx, axis=0)
    tx -= mu
    sigma = np.std(tx, axis=0)
    return tx / sigma


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    """
    data_size = len(y)
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def accuracy(y_pred, y):
    """Give the accuracy of the prediction."""
    return sum([y_pred[i] == y[i] for i in range(len(y))]) / len(y)


def predict_labels(data, w):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, w)
    y_pred[np.where(y_pred <= 0)] = 0
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred



def predict_labels_logistic(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = sigmoid(data@weights)
    y_pred[y_pred < 0.5] = 0
    y_pred[y_pred > 0.5] = 1
    
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})