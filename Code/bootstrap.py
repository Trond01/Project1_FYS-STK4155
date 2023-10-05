import numpy as np
import matplotlib.pyplot as plt

from Code.utilities import (
    r2_sampling,
    random_dataset_split_preparation,
    maximal_degree,
    nn_deg_indeces,
)
from Code.regression import fit_OLS

# Code inspired by https://compphysics.github.io/MachineLearning/doc/LectureNotes/_build/html/chapter3.html#the-bias-variance-tradeoff


def resample(X, Z):
    # Index range
    index_range = np.arange(0, X.shape[0])

    # Generate random numbers without replacement
    indeces = np.random.choice(index_range, size=X.shape[0], replace=True)

    return X[indeces, :], Z[indeces, :]


# Code inspired by https://compphysics.github.io/MachineLearning/doc/LectureNotes/_build/html/chapter3.html#the-bias-variance-tradeoff
def bootstrap(data, n_bootstraps=10, plot_first=False):
    """ """
    # Get data
    train_X, train_Z, test_X, test_Z = (
        data["train_X"],
        data["train_Z"],
        data["test_X"],
        data["test_Z"],
    )

    # Array for storing finds
    Z_pred = np.empty((test_Z.shape[0], n_bootstraps))


    bootstraps_train_loss = []
    bootstraps_test_loss = []

    # Run the bootstrap
    for i in range(n_bootstraps):
        # Resample
        data["train_X"], data["train_Z"] = resample(train_X, train_Z)

        # Run model
        if i > 0:
            plot_first = False
        run_data = fit_OLS(data, plot_or_not=plot_first)

        bootstraps_train_loss.append(run_data['train_loss'])
        bootstraps_test_loss.append(run_data['test_loss'])

        # Make prediction
        Z_pred[:, i] = run_data["predictor"](test_X).ravel()

    # Compute error, bias and variance estimators
    error = np.mean(np.mean((test_Z - Z_pred) ** 2, axis=1, keepdims=True))
    bias = np.mean((test_Z - np.mean(Z_pred, axis=1, keepdims=True)) ** 2)
    variance = np.mean(np.var(Z_pred, axis=1, keepdims=True))

    return error, bias, variance, sum(bootstraps_train_loss)/len(bootstraps_train_loss), sum(bootstraps_test_loss)/len(bootstraps_test_loss)


def tradeoff_experiment(
    num_points=100,
    num_features=100,
    n_bootstraps=50,
    nth=10,
    data=None,
    seed=42,
    mark_deg_nn=True,
    filename=False,
    sigma2=0.0,
):
    """
    If using custom data, the data variable needs to be a dictionary with the fields {'x', 'y', 'z'}
    containing arrays with their respective data points. x, y, z should be of shape (number_points, 1)
    """
    """
    nth: plot each nth figure
    """

    # Set seed
    np.random.seed(seed)

    # Initialise data storage
    feature_numbers = np.arange(1, num_features + 1, 1)
    errors = np.zeros(num_features)
    biases = np.zeros(num_features)
    variances = np.zeros(num_features)

    train_losses = np.zeros(num_features)
    test_losses = np.zeros(num_features)

    # If no data is given, sample from Franke.
    if data is None:
        data = r2_sampling(num_points, sigma2=sigma2)

    # Start with test_index = None to make random_dat... generate test_indeces.
    # These same indeces is used throughout
    test_index = None

    # Run experiment for each feature number
    for num in feature_numbers:
        # Create feature matrix
        new_data = random_dataset_split_preparation(
            data, num, scale=True, test_index=test_index
        )
        test_index = new_data["test_index"]  # Ensure same test split

        # Compute error, bias and variance
        plot = False
        if num % nth == 0:
            plot = True
        error, bias, variance, train_mean, test_mean = bootstrap(
            new_data, n_bootstraps=n_bootstraps, plot_first=plot
        )


        # We need to scale bias and variance to get a meaningfull result
        bias, variance = bias, variance

        # Add the result
        errors[num - 1] = error
        biases[num - 1] = bias
        variances[num - 1] = variance

        train_losses[num - 1] = train_mean
        test_losses[num - 1] = test_mean


    ## Make plot of bias and variance

    # Highlight p(x)=a0+...+an0 x^n +...+a0n y^n (whole number final), plot biases and variances
    if mark_deg_nn:
        max_deg = maximal_degree(num_features)
        xy_deg_indeces = nn_deg_indeces(max_deg)
        plt.plot(feature_numbers[xy_deg_indeces], biases[xy_deg_indeces], "o", c="b")

    plt.plot(feature_numbers, errors, label="error", c="r")
    plt.plot(feature_numbers, biases, label="bias", c="b")
    plt.plot(feature_numbers, variances, label="variance", c="g")

    # Add legend and labels
    plt.legend()
    plt.xlabel("Number of features")
    plt.title("The Bias Variance Tradeoff")

    # Save figure if filename given
    if filename:
        plt.savefig(filename)
    plt.show()

    plt.plot(feature_numbers, train_losses, label="Bootstrap Train Loss", c="b")
    plt.plot(feature_numbers, test_losses, label="Bootstrap Test Loss", c="r")
    
    # Add legend and labels
    plt.legend()
    plt.xlabel("Number of features")
    plt.ylabel("MSE")
    plt.title("Loss")

    plt.show()

    return feature_numbers, errors, biases, variances
