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

print("Building polynomial..")
# Best found degree is 15 (from 10 to 25)
poly_train = build_poly(standardized_trainx, 15)
poly_train = np.c_[np.ones(poly_train.shape[0]), poly_train]

print("Training with ridge regression...")
# Best found lambda is 1e-8
# lambdas_ = np.linspace(10e-9, 10e-4, 20)
poly_w, poly_loss = ridge_regression(y, poly_train, 1e-8)
poly_y_pred = predict_labels(poly_train, poly_w)
print ('Accuracy', accuracy(poly_y_pred, y))

print("Making predctions for the test file...")
y_pred = predict_labels(standardized_testx, poly_w)
create_csv_submission(test_id, y_pred, 'predictions.csv')
print("The prediction has been stored in the predictions.csv file")