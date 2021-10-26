import numpy as np
from proj1_helpers import batch_iter, accuracy


def compute_mse(e):
    """Calculate mse loss."""
    return np.mean(e**2)


def compute_loss(y, tx, w):
    """Calculate the loss using mse"""
    e = y - tx.dot(w)
    return compute_mse(e)


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y - tx.dot(w)
    grad = tx.T.dot(e) / len(e)
    return grad


def least_squares(y, tx):
    """Least squares regression using normal equations."""
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    loss = compute_loss(y, tx, w)
    return w, loss


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent."""
    w = initial_w
    for i in range(max_iters):
        grad = compute_gradient(y, tx, w)
        w += gamma * grad
    loss = compute_loss(y, tx, w)
    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent."""
    w = initial_w
    for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=max_iters):
        grad = compute_gradient(y_batch, tx_batch, w)
        w += gamma * grad
    loss = compute_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations."""
    lambda_I = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    w = np.linalg.solve(tx.T.dot(tx) + lambda_I, tx.T.dot(y))
    e = y - tx.dot(w)
    loss = compute_mse(e)
    return w, loss


def build_poly(x, degree):
    """Polynomial basis functions for input data x, for 0 up to a given degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


def sig(x):
    """Sigmoid function for a scalar - preventing overflow"""
    if x > 0:
        return 1.0 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (1.0 + np.exp(x))
sigmoid = np.vectorize(sig)


def compute_sigmoid_loss(y, tx, w):
    """Compute loss with mse."""
    sig = sigmoid(tx.dot(w))
    loss = compute_mse(y-sig)
    return np.squeeze(-loss)


def compute_sigmoid_gradient(y, tx, w):
    """Compute gradient with sigmoid function."""
    sig = sigmoid(tx.dot(w))
    grad = tx.T.dot(sig - y)
    return grad


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression."""
    w = initial_w
    for i in range(max_iters):
        grad = compute_sigmoid_gradient(y, tx, w)
        w -= gamma * grad
    loss = compute_sigmoid_loss(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w=None, max_iters=50, gamma=0.00005):
    """Regularized logistic regression."""
    if np.all(initial_w == None): 
        initial_w = np.zeros(tx.shape[1])
    w = initial_w
    for i in range(max_iters):
        grad = compute_sigmoid_gradient(y, tx, w) + 2 * lambda_ * w
        w -= gamma * grad
    loss = compute_sigmoid_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    return w, loss
    

def build_k_indices(y, k_fold, seed):
    """Build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, fun, k_indices, k, degree, alpha, lamb=None, log=False, **kwargs):
    """
    Completes k-fold cross-validation for Least Squares with GD, SGD, Normal Equations, Logistic and Regularized Logistic 
    Regression with SGD
    """
    # get k'th subgroup in test, others in train
    msk_test = k_indices[k]
    msk_train = np.delete(k_indices, (k), axis=0).ravel()

    x_train = x[msk_train, :]
    x_test = x[msk_test, :]
    y_train = y[msk_train]
    y_test = y[msk_test]

    # initialize output vectors
    y_train_pred = np.zeros(len(y_train))
    y_test_pred = np.zeros(len(y_test))
 
    # data pre-processing
    # x_train, x_test = process_data(x_train, x_test, alpha)
            
    # transformation
    x_train, x_test = phi(x_train, x_test, degree)
        
    # compute weights using given method
    if lamb == None:
        w, loss = fun(y_train, x_train, **kwargs)
    else:
        w, loss = fun(y_train, x_train, lamb, **kwargs)
       
    # predict
    if log == True:
        y_train_pred = predict_labels_logistic(w, x_train)
        y_test_pred = predict_labels_logistic(w, x_test)
    else:
        y_train_pred = predict_labels(w, x_train)
        y_test_pred = predict_labels(w, x_test)
        

    # compute accuracy for train and test data
    acc_train = accuracy(y_train_pred, y_train)
    acc_test = accuracy(y_test_pred, y_test)
    
    return acc_train, acc_test, loss

def predict_labels_logistic(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = sigmoid(data@weights)
    y_pred[y_pred < 0.5] = -1
    y_pred[y_pred > 0.5] = 1
    
    return y_pred

def phi(x_train, x_test, degree):
    """
    Transformation of X matrix: polynomial expansion and coupling
    """
    # Polynomial expansion and coupling
    x_train = build_poly(x_train, degree)
    x_test = build_poly(x_test, degree)
    
    return x_train, x_test


def build_poly2(x, degree):
    """ Polynomial expansion: add an intecept
                             for each feature polynomial expansion from 1 to degree
                             for each feature create a new feature equal to the root and cubic square 
                             for each couple of feature create a new feature equal to the product """
    N, D = x.shape    
    # couples
    temp_dict2 = {}
    count2 = 0
    for i in range(D):
        for j in range(i+1,D):
            temp = x[:,i] * x[:,j]
            temp_dict2[count2] = [temp]
            count2 += 1
    
    poly = np.zeros(shape=(N, 1+D*(degree+2)+count2))
    
    # intercept
    poly[:,0] = np.ones(N)
    # powers
    for deg in range(1,degree+1):
        for i in range(D):
            poly[:, 1+D*(deg-1)+i ] = np.power(x[:,i],deg)      
    # coupling     
    for i in range(count2):
        poly[:, 1+D*degree+i ] = temp_dict2[i][0]     
    # roots   
    for i in range(D):
        poly[:, 1+D*degree+count2+i] = np.abs(x[:,i])**0.5
    poly[:, 1+D*degree+count2+D:] = rad(x, 3)
    
    return poly
