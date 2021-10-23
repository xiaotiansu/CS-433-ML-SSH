import numpy as np

def compute_error(y, tx, w):
    """Calculate the error vector."""
    return y - tx.dot(w)
    

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
    gradient = -tx.T.dot(e) / len(e)
    return gradient


def least_squares(y, tx):
    """Least squares regression using normal equations."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    return w, loss


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent."""
    w = initial_w
    for i in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        w = w - gamma * gradient
        print("Least Squares GD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=i, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    loss = compute_loss(y, tx, w)
    return w, loss


def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """Linear regression using stochastic gradient descent."""
    w = initial_w
    for i in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            gradient = compute_gradient(y, tx, w)
            w = w - gamma * gradient
        print("Least Squares SGD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=i, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    loss = compute_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations."""
    lambda_I = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + lambda_I
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    e = y - tx.dot(w)
    loss = compute_mse(e)
    return w, loss


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


def sigmoid(x):
    """Define sigmoid function."""
    return 1.0 / (1 + np.exp(-x))


def compute_logistic_loss(y, tx, w):
    """Compute the cost by negative log likelihood."""
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)


def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)
    return grad


def calculate_hessian(y, tx, w):
    """return the Hessian of the loss function."""
    pred = sigmoid(tx.dot(w))
    pred = np.diag(pred.T[0])
    r = np.multiply(pred, (1-pred))
    return tx.T.dot(r).dot(tx)


def logistic_regression(y, tx, w):
    """Logistic regression using gradient descent or SGD."""
    loss = compute_logistic_loss(y, tx, w)
    gradient = calculate_gradient(y, tx, w)
    hessian = calculate_hessian(y, tx, w)
    return loss, gradient, hessian


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD."""
    w = initial_w
    for i in range(max_iters):
        grad = np.zeros(tx.shape[1])
        sigma = sigmoid(tx.dot(w))
        SX = tx * (sigma - sigma*sigma).reshape(-1,1)
        XSX = tx.transpose().dot(SX)
        for aw in range(tx.shape[0]):
            grad = grad + (-1 / tx.shape[0]) * (y[aw] * tx[aw,:] * sigmoid(-y[aw] * np.dot(tx[aw,:], w)))
        w = w - gamma * np.linalg.solve(XSX, grad)
        if i % 5 == 0 and i != 0:
            gamma = gamma * 0.55
    loss = compute_logistic_loss(y, tx, w)

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent"""
    w = initial_w
    for i in range(max_iters):
        grad = np.zeros(tx.shape[1])
        sigma = sigmoid(tx.dot(w))
        SX = tx * (sigma - sigma*sigma).reshape(-1,1)
        XSX = tx.transpose().dot(SX) + lambda_*np.eye((tx.shape[1]))
        for aw in range(tx.shape[0]):
            grad = grad + (-1 / tx.shape[0]) * (y[aw] * tx[aw,:] * sigmoid(-y[aw] * np.dot(tx[aw,:],w)))
        w = w - gamma * np.linalg.solve(XSX, grad) - gamma * lambda_*w
        if i % 5 == 0 and i != 0:
            gamma = gamma * 0.55
    loss = compute_logistic_loss(y, tx, w)

    return w, loss
