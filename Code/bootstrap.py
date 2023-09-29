import numpy as np
import matplotlib.pyplot as plt

from Code.utilities import r2_sampling, random_dataset_split_preparation
from Code.regression import fit_OLS

def resample(X, Z):

    # Index range
    index_range = np.arange(0, X.shape[0])

    # Generate random numbers without replacement
    indeces = np.random.choice(index_range, size=X.shape[0], replace=True)

    return X[indeces, :], Z[indeces, :]

# Code inspired by https://compphysics.github.io/MachineLearning/doc/LectureNotes/_build/html/chapter3.html#the-bias-variance-tradeoff
def bootstrap(data, n_bootstraps = 10, plot_first = False):
    """
    
    """
    # Get data
    train_X, train_Z, test_X, test_Z  = data['train_X'], data['train_Z'], data['test_X'], data['test_Z']

    # Array for storing finds
    Z_pred = np.empty((test_Z.shape[0], n_bootstraps))    

    # Run the bootstrap
    for i in range(n_bootstraps):
        
        # Resample
        data["train_X"], data["train_Z"] = resample(train_X, train_Z)
        
        # Run model
        if i > 0:
            plot_first = False
        run_data = fit_OLS(data, plot_or_not=plot_first)

        # Make prediction
        Z_pred[:, i] = run_data["predictor"](test_X[0])

    # Compute error, bias and variance
    error    = np.mean( np.mean((test_Z - Z_pred)**2, axis=1, keepdims=True) )
    bias     = np.mean( (test_Z - np.mean(Z_pred, axis=1, keepdims=True))**2 )
    variance = np.mean( np.var(Z_pred, axis=1, keepdims=True) )

    return error, bias, variance


def tradeoff_experiment(num_points = 100 ,num_features = 100, n_bootstraps = 50, nth=10):    
    """
    nth: plot each nth figure
    """

    # Initialise data storage
    feature_numbers = np.arange(1, num_features, 1)
    errors    = []
    biases    = []
    variances = []
    
    # Initial sample of data, so far no index for test
    data = r2_sampling(num_points)
    test_index = None

    # Run experiment for each feature number
    for num in feature_numbers:

        # Create feature matrix        
        new_data = random_dataset_split_preparation(data, num, scale=True, test_index=test_index)
        test_index = new_data["test_index"] # Ensure same test split

        # Compute error, bias and variance
        plot = False
        if num % nth == 0:
            plot = True
        error, bias, variance = bootstrap(new_data, n_bootstraps=n_bootstraps, plot_first=plot)

        # We need to scale bias and variance to get a meaningfull result
        bias, variance = bias/error, variance/error

        # Add the result
        errors.append(error) # unscaled...
        biases.append(bias)
        variances.append(variance)

    fig, ax = plt.subplots(figsize=(7, 7))

    plt.plot(feature_numbers, biases,    label = "bias",     c="b")
    plt.plot(feature_numbers, variances, label = "variance", c="g")
    plt.legend()
    plt.savefig("bias_var_trade2.png")
    plt.xlabel("Number of features")
    plt.ylabel("Quantity divided by total error")
    plt.show()

    fig, ax = plt.subplots(figsize=(7, 7))

    plt.plot(feature_numbers, errors, c="r")
    plt.legend()
    # plt.savefig("bias_var_trade2.err.png")
    plt.xlabel("Number of features")
    plt.ylabel("Error")
    plt.show()

    return feature_numbers, errors

feature_numbers, errors = tradeoff_experiment(num_points=50,num_features=50,nth=100)


