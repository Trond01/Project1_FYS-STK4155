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

    # Compute error, bias and variance estimators
    error = np.mean(np.mean((test_Z - Z_pred) ** 2, axis=1, keepdims=True))
    bias = np.mean((test_Z - np.mean(Z_pred, axis=1, keepdims=True)) ** 2)
    variance = np.mean(np.var(Z_pred, axis=1, keepdims=True))

    return error, bias, variance


def cross_validation_experiment(n_folds, n_points=100, num_features=100, data = None, mark_deg_nn = True, filename = None, seed=42, start=1):
    """
    If using custom data, the data variable needs to be a dictionary with the fields {'x', 'y', 'z'}
    containing arrays with their respective data points. x, y, z should be of shape (number_points, 1)
    """

    # Set seed
    np.random.seed(seed)

    # Feature numbers and data storage init
    feature_numbers  = np.arange(start, num_features+1, 1)
    all_errors_test  = np.zeros(num_features-start+1)
    all_errors_train = np.zeros(num_features-start+1)

    # Initial sample of data, so far no index for test
    if data is None:
        data = r2_sampling(n_points)

    # Run experiment for each feature number
    for num in feature_numbers:

        X = feature_matrix(data['x'], data['y'], num)
        X, means, var = scale_feature_matrix(X)

        splits = k_fold_split(X, data['z'], n_folds)
        

        single_fold_errors_test  = []     
        single_fold_errors_train = []        
        for train_X, train_Z, test_X, test_Y in splits:
            fold_data = {'train_X':train_X, 'train_Z':train_Z, 'test_X':test_X, 'test_Z':test_Y,
                        'feature_matrix':X, 'means':means, 'var':var, 'num_features':num,
                        'x':data['x'], 'y':data['y'], 'z':data['z']}
            results_data = fit_OLS(data=fold_data, plot_or_not=False)
            single_fold_errors_test.append(results_data['test_loss'])
            single_fold_errors_train.append(results_data['train_loss'])
        
        all_errors_test[num-start]  = np.mean(single_fold_errors_test)
        all_errors_train[num-start] = np.mean(single_fold_errors_train)

    # Make plot of test and train errors
    plt.plot(feature_numbers, all_errors_test,    label = "Average test error",    c="g")
    plt.plot(feature_numbers, all_errors_train,    label = "Average train error",    c="r")

    # Mark finished degrees
    if mark_deg_nn:
        max_deg = maximal_degree(num_features)
        xy_deg_indeces = nn_deg_indeces(max_deg)
        xy_deg_indeces = xy_deg_indeces[xy_deg_indeces >= start-1] - start + 1
        plt.plot(feature_numbers[xy_deg_indeces], all_errors_train[xy_deg_indeces], "o", c="r")

    # Add legend, x label and save
    plt.legend()
    plt.xlabel("Number of features")
    if filename:
        plt.savefig(filename)
    plt.show()

    return all_errors_test, all_errors_train
    

    return feature_numbers, errors, biases, variances
