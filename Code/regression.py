import numpy as np
from sklearn import linear_model
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

from Code.plot import plot_model_prediction
from Code.utilities import (
    MSELoss,
    random_dataset_split_preparation,
    r2_sampling,
    R_squared,
)


def fit_OLS(data, plot_or_not=False, **kwargs):
    """Takes in data dictionary containing train and test sets in feature matrix form and fits an OLS model to it. Returns data dictionary with model and performance metrics.

    Args:
        data (dict): Data dictionary containing train and test set.
        plot_or_not (bool, optional): Plots the predicted surface of the model when true. Defaults to False.

    Returns:
        dict: Dictionary with original data dictionary and train/test loss metrics as well as model weights.
    """
    # Get data
    train_X, train_Z, test_X, test_Z = (
        data["train_X"],
        data["train_Z"],
        data["test_X"],
        data["test_Z"],
    )

    # Compute estimator for beta and prediction of test dataset
    beta = np.dot(np.dot(np.linalg.inv(np.dot(train_X.T, train_X)), train_X.T), train_Z)
    y_pred = np.dot(test_X, beta)

    # Plot results
    if plot_or_not:
        # Variables needed for plotting
        num_features = data["num_features"]
        means, var = data["means"], data["var"]

        # Do the plotting
        plot_model_prediction(lambda X: np.dot(X, beta), num_features, means, var)

    # Return results
    return {
        "test_loss": MSELoss(test_Z, y_pred),
        "train_loss": MSELoss(train_Z, np.dot(train_X, beta)),
        "test_loss_R2": R_squared(test_Z, y_pred),
        "train_loss_R2": R_squared(train_Z, np.dot(train_X, beta)),
        "weights": beta,
        "predictor": lambda X: np.dot(X, beta),
    } | data


def train_OLS(
    data, num_features, scale=True, test_index=None, plot_or_not=False, **kwargs
):
    """Takes in sampled data points in data dictionary and creates feature matrix and trains OLS model on it.

    Args:
        data (dict): Data dictionary containing x, y and z arrays of datapoints.
        num_features (int): Number of features.
        scale (bool, optional): Scales feature matrix when true. Defaults to True.
        test_index (ndarray, optional): Optional indexing array used to partition train and test set. Defaults to None.
        plot_or_not (bool, optional): Plots predicted model surface when true. Defaults to False.

    Returns:
        dict: Dictionary with original data dictionary and train/test loss metrics as well as model weights.
    """
    data = random_dataset_split_preparation(
        data, num_features, scale=scale, test_index=test_index
    )

    data = fit_OLS(data, plot_or_not=plot_or_not)

    return data


# Define the ridge regression trainer
def fit_RIDGE(data, lam, plot_or_not=False):
    """Takes in data dictionary containing train and test sets in feature matrix form and fits a Ridge regression model to it. Returns data dictionary with model and performance metrics.

    Args:
        data (dict): Data dictionary containing train and test set.
        lam (float): Lambda to use in the Ridge regression.
        plot_or_not (bool, optional): Plots the predicted surface of the model when true. Defaults to False.

    Returns:
        dict: Dictionary with original data dictionary and train/test loss metrics as well as model weights.
    """
    # Get data
    train_X, train_Z, test_X, test_Z = (
        data["train_X"],
        data["train_Z"],
        data["test_X"],
        data["test_Z"],
    )
    num_features = data["num_features"]
    means, var = data["means"], data["var"]

    # Compute the estimated beta and make prediction on test set
    beta = np.dot(
        np.dot(
            np.linalg.inv(
                np.dot(train_X.T, train_X) + np.identity(train_X.shape[1]) * lam
            ),
            train_X.T,
        ),
        train_Z,
    )
    y_pred = np.dot(test_X, beta)

    # Do the plotting
    if plot_or_not:
        plot_model_prediction(lambda X: np.dot(X, beta), num_features, means, var)

    return {
        "test_loss": MSELoss(test_Z, y_pred),
        "train_loss": MSELoss(train_Z, np.dot(train_X, beta)),
        "test_loss_R2": R_squared(test_Z, y_pred),
        "train_loss_R2": R_squared(train_Z, np.dot(train_X, beta)),
        "weights": beta,
        "lam": lam,
        "predictor": (lambda X: np.dot(X, beta)),
    } | data  # Merging Dictionaries


def train_RIDGE(
    data, num_features, lam, scale=True, plot_or_not=False, test_index=None
):
    """Takes in sampled data points in data dictionary and creates feature matrix and trains a Ridge regression model on it.

    Args:
        data (dict): Data dictionary containing x, y and z arrays of datapoints.
        num_features (int): Number of features.
        lam (float): Lambda to use in the Ridge regression.
        scale (bool, optional): Scales feature matrix when true. Defaults to True.
        test_index (ndarray, optional): Optional indexing array used to partition train and test set. Defaults to None.
        plot_or_not (bool, optional): Plots predicted model surface when true. Defaults to False.

    Returns:
        dict: Dictionary with original data dictionary and train/test loss metrics as well as model weights.
    """
    data = random_dataset_split_preparation(
        data, num_features, scale=scale, test_index=test_index
    )

    data = fit_RIDGE(data, lam, plot_or_not=plot_or_not)

    return data


def fit_LASSO(data, lam, plot_or_not=False, max_iter=1000):
    """Takes in data dictionary containing train and test sets in feature matrix form and fits a Lasso regression model to it. Returns data dictionary with model and performance metrics.

    Args:
        data (dict): Data dictionary containing train and test set.
        lam (float): Lambda to use in the Lasso regression.
        plot_or_not (bool, optional): Plots the predicted surface of the model when true. Defaults to False.

    Returns:
        dict: Dictionary with original data dictionary and train/test loss metrics as well as model weights.
    """
    
    
    """
    Scikit optimises this:
        (1 / (2 * n_samples)) * ||y - Xw||^2_2 + lambda * ||w||_1
    """

    # Get data
    train_X, train_Z, test_X, test_Z = (
        data["train_X"],
        data["train_Z"],
        data["test_X"],
        data["test_Z"],
    )

    # Do the fitting and evaluate errors
    regressor = linear_model.Lasso(
        lam, fit_intercept=False, max_iter=max_iter
    )  # We include intercept with False
    regressor.fit(train_X, train_Z)

    # Find test and train loss
    test_pred = regressor.predict(test_X)
    train_pred = regressor.predict(train_X)

    # Do the plotting
    if plot_or_not:
        num_features = data["num_features"]
        means, var = data["means"], data["var"]

        plot_model_prediction(lambda X: regressor.predict(X), num_features, means, var)

    train_Z = train_Z.reshape((train_Z.shape[0], 1))
    train_pred = train_pred.reshape((train_pred.shape[0], 1))
    test_Z = test_Z.reshape((test_Z.shape[0], 1))
    test_pred = test_pred.reshape((test_pred.shape[0], 1))

    return {
        "test_loss": MSELoss(test_Z, test_pred),
        "train_loss": MSELoss(train_Z, train_pred),
        "test_loss_R2": R_squared(test_Z, test_pred),
        "train_loss_R2": R_squared(train_Z, train_pred),
        "weights": regressor.coef_,  # Note that these are not comparable to the previous in scale...
        "predictor": (lambda X: regressor.predict(X)),
        "lam": lam,
    } | data


def train_LASSO(
    data,
    num_features,
    lam,
    scale=True,
    plot_or_not=False,
    test_index=None,
    max_iter=1000,
):
    """Takes in sampled data points in data dictionary and creates feature matrix and trains a Lasso regression model on it.

    Args:
        data (dict): Data dictionary containing x, y and z arrays of datapoints.
        num_features (int): Number of features.
        lam (float): Lambda to use in the Lasso regression.
        scale (bool, optional): Scales feature matrix when true. Defaults to True.
        test_index (ndarray, optional): Optional indexing array used to partition train and test set. Defaults to None.
        plot_or_not (bool, optional): Plots predicted model surface when true. Defaults to False.
        max_iter (int): Maximum number of iterations used in sklearn fitting of model. Defaults to 1000

    Returns:
        dict: Dictionary with original data dictionary and train/test loss metrics as well as model weights.
    """
    data = random_dataset_split_preparation(
        data, num_features, scale=scale, test_index=test_index
    )

    data = fit_LASSO(data, lam, plot_or_not=plot_or_not, max_iter=max_iter)

    return data
