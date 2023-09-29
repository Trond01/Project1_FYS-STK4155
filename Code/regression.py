import numpy as np
from sklearn import linear_model

from Code.plot import plot_model_prediction
from Code.utilities import MSELoss, random_dataset_split_preparation, r2_sampling

def fit_OLS(data, plot_or_not=False):

    # Get data
    train_X,train_Z, test_X, test_Z  = data['train_X'], data['train_Z'], data['test_X'], data['test_Z']

    # Compute estimator for beta and prediction of test dataset
    beta = np.dot(np.dot(np.linalg.inv(np.dot(train_X.T, train_X)), train_X.T), train_Z)
    y_pred = np.dot(test_X, beta)

    # Plot results
    if plot_or_not:

        # Variables needed for plotting
        num_features = data['num_features']
        means, var = data['means'], data['var']

        # Do the plotting
        plot_model_prediction(lambda X : np.dot(X, beta), num_features, means, var)
    
    # Return results
    return {'test_loss'  : MSELoss(test_Z, y_pred),
            'train_loss' : MSELoss(np.dot(train_X, beta),train_Z),
            'weights'    : beta,
            'predictor'  : lambda X : np.dot(X, beta)} | data


def fit_LASSO(data, lam, plot_or_not=False):

    # Get data
    train_X,train_Z, test_X, test_Z  = data['train_X'], data['train_Z'], data['test_X'], data['test_Z']
    
    # Do the fitting and evaluate errors
    regressor = linear_model.Lasso(lam, fit_intercept=False)
    regressor.fit(train_X, train_Z)

    # Find test and train loss
    test_pred = regressor.predict(test_X)
    train_pred = regressor.predict(train_X)

    # Do the plotting
    if plot_or_not:
        num_features = data['num_features']
        means,    var      = data['means'],    data['var']

        plot_model_prediction(lambda X : regressor.predict(X), num_features, means, var)
    
    return {'test_loss' : MSELoss(test_Z, test_pred),
            'train_loss': MSELoss(train_Z, train_pred),
            'predictor' : (lambda X : regressor.predict(X)),
            'lam'       : lam} | data


def fit_RIDGE(data, lam, plot_or_not=False):

    # Get data
    train_X,train_Z, test_X, test_Z  = data['train_X'], data['train_Z'], data['test_X'], data['test_Z']
    num_features = data['num_features']
    means, var = data['means'], data['var']

    # Compute the estimated beta and make prediction on test set
    beta = np.dot(np.dot(np.linalg.inv(np.dot(train_X.T, train_X) + np.identity(train_X.shape[1])*lam), train_X.T), train_Z)
    y_pred = np.dot(test_X, beta)

    # Do the plotting
    if plot_or_not:
        plot_model_prediction(lambda X : np.dot(X, beta), num_features, means, var)
    
    return {'test_loss':MSELoss(test_Z, y_pred),
            'train_loss':MSELoss(np.dot(train_X, beta), train_Z),
            'weights':beta,
            'lam':lam,
            'predictor': (lambda X : np.dot(X, beta))} | data # Merging Dictionaries


def train_OLS(data, num_features, scale=True, test_index=None, plot_or_not=False):
    data = random_dataset_split_preparation(data, num_features, scale=scale, test_index=test_index)
    data = fit_OLS(data, plot_or_not=plot_or_not)

    return data


def train_RIDGE(data, num_features, lam, scale=True, plot_or_not=False, test_index=None):
    data = random_dataset_split_preparation(data, num_features, scale=scale, test_index=test_index)
    data = fit_RIDGE(data, lam, plot_or_not=plot_or_not)

    return data


def train_LASSO(data, num_features, lam, scale=True, plot_or_not=False, test_index=None):
    data = random_dataset_split_preparation(data, num_features, scale=scale, test_index=test_index)
    data = fit_LASSO(data, lam, plot_or_not=plot_or_not)

    return data
