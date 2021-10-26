import numpy as np
import matplotlib.pyplot as plt

from implementations import *
from proj1_helpers import *


print("Loading the training and testing data...")
y, trainx, train_id = load_csv_data('data/train.csv')
_, testx, test_id = load_csv_data('data/test.csv')

print("Cleanning the abnormal values and standardizing dataset with Z-score...")
standardized_trainx = standardize(trainx)
standardized_testx = standardize(testx)
'''
seed=10
degrees = [5, 5, 5]
alphas = [4, 4, 5]

# Model parameters
lambdas=0.1
max_iters = 500
gamma = 0.00001

# Split data in k-fold
k_fold = 5
k_indices = build_k_indices(y, k_fold, seed)


accs_train = []
accs_test = []
lambdas_ = np.linspace(10**-9, 4*10**-9, 4)
logistic_weights = []
gammas_ = np.linspace(4.5*10**5, 5.5*10**5, 3)
losses = np.zeros((len(lambdas_), len(gammas_)))
for i in range(len(lambdas_)):
    for j in range(len(gammas_)):
        for k in range(k_fold):
            acc_train, acc_test, loss = cross_validation(y, standardized_trainx, reg_logistic_regression, k_indices, k, 12, alphas, lambdas_[i], log=True,
                                                initial_w = None, max_iters=10, gamma=gammas_[j])
            accs_train.append(acc_train)
            accs_test.append(acc_test)
            losses[i,j] = loss
    
for i in range(len(accs_train)):
    print("Iter %d: Training accuracy: %f / Test accuracy : %f" % (i, accs_train[i], accs_test[i]))

min_losses = np.amin(losses)
idx_min = np.argmin(losses)
print("\nAverage test accuracy: %f" % np.mean(accs_test))
print("Variance test accuracy: %f" % np.var(accs_test))
print("loss: %f" % min_losses)
print("i, j: %f" % idx_min)

# print("Making predctions for the test file...")
# y_pred = predict_labels(standardized_testx, reg_logit_w)
# create_csv_submission(test_id, y_pred, 'predictions.csv')


initial_w = np.zeros((len(standardized_trainx[0])))
print("Training using least_squares_GD...")
gd_w, gd_loss = least_squares_GD(y, standardized_trainx, initial_w, 25, 1000)
gd_y_pred = predict_labels(standardized_trainx, gd_w)
print ('Accuracy', accuracy(gd_y_pred, y))

initial_w = np.zeros((len(standardized_trainx[0])))
print("Training using least_squares_SGD...")
sgd_w, sgd_loss = least_squares_SGD(y, standardized_trainx, initial_w, 10, 0.5)
sgd_y_pred = predict_labels(standardized_trainx, sgd_w)
print('loss', sgd_loss)
print ('Accuracy', accuracy(sgd_y_pred, y))
'''

initial_w = np.zeros((len(standardized_trainx[0])))
print("Training using logistic_regression...")
logit_w, logit_loss = logistic_regression(y, standardized_trainx, initial_w, 10, 2)
logit_y_pred = predict_labels(standardized_trainx, logit_w)
print('loss', logit_loss)
print ('Accuracy', accuracy(logit_y_pred, y))

initial_w = np.zeros((len(standardized_trainx[0])))
print("Training using reg_logistic_regression...")
reg_logit_w, reg_logit_loss = reg_logistic_regression(y, standardized_trainx, 1e-8, initial_w, 50, 0.1)
reg_logit_y_pred = predict_labels(standardized_trainx, reg_logit_w)
print('loss', reg_logit_loss)
print ('Accuracy', accuracy(reg_logit_y_pred, y))

'''
print("Making predctions for the test file...")
y_pred = predict_labels(standardized_testx, reg_logit_w)
create_csv_submission(test_id, y_pred, 'predictions.csv')
print("The prediction has been stored in the predictions.csv file")


print("Building polynomial..")
# Best found degree is 15 (from 10 to 25)
poly_train = build_poly(standardized_trainx, 15)
poly_train = np.c_[np.ones(poly_train.shape[0]), poly_train]

print("Training using ridge regression with polynomial without PCA...")
# Best found lambda is 1e-8
# lambdas_ = np.linspace(10e-9, 10e-4, 20)
poly_w, poly_loss = ridge_regression(y, poly_train, 1e-8)
poly_y_pred = predict_labels(poly_train, poly_w)
print ('Accuracy', accuracy(poly_y_pred, y))


eig_val, eig_vec, k = PCA(standardized_trainx, threshold=0.7)
standardized_trainx = standardized_trainx.dot(eig_vec)
print("Building polynomial again with a different degree...")
# Best found degree is 20 (from 10 to 30)
# for i in range(8, 20):
poly_train = build_poly(standardized_trainx, 20)
poly_train = np.c_[np.ones(poly_train.shape[0]), poly_train]

print("Training using ridge regression with polynomial with PCA...")
poly_w, poly_loss = ridge_regression(y, poly_train, 1e-8)
poly_y_pred = predict_labels(poly_train, poly_w)
print ('Accuracy', accuracy(poly_y_pred, y))

print("Making predctions for the test file...")
y_pred = predict_labels(standardized_testx, gd_w)
create_csv_submission(test_id, y_pred, 'predictions.csv')
print("The prediction has been stored in the predictions.csv file")

poly_test = build_poly(standardized_testx, 15)
poly_test = np.c_[np.ones(poly_test.shape[0]), poly_test]
y_pred = predict_labels(poly_test, poly_w)
create_csv_submission(test_id, y_pred, 'predictions.csv')
print("The prediction has been stored in the predictions.csv file")
'''