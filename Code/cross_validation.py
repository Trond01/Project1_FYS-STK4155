from matplotlib import pyplot as plt
import numpy as np
from Code.utilities import r2_sampling, feature_matrix, scale_feature_matrix, maximal_degree, nn_deg_indeces
from Code.regression import fit_OLS

def k_fold_split(X, Z, k):
    """
    returns list of touples (train, test) of length k
    
    Let * be train and _ be test. The 5-fold split can be understood by:

        ****_
        ***_*
        **_**
        *_***
        _****

    """
    # Find number of data and split size   
    n = X.shape[0]
    s = n // k # Split size

    # Remove extra data (need divisible by k)
    X, Z = X[0:s*k], Z[0:s*k]

    # List for storing splits
    splits = []

    # Make splits
    for i in range(k):

        # Find indeces
        test_indeces  = np.arange(i*s, (i+1)*s, 1)
        train_indeces = np.delete(np.arange(0, k*s), test_indeces)

        # Append the fold
        splits.append((X[train_indeces, :], Z[train_indeces, :], X[test_indeces, :], Z[test_indeces, :]))

    return splits



def cross_validation_experiment(n_folds, n_points=100, num_features=100, data = None, mark_deg_nn = True, filename = None, seed=42, start=0):
    """
    If using custom data, the data variable needs to be a dictionary with the fields {'x', 'y', 'z'}
    containing arrays with their respective data points. x, y, z should be of shape (number_points, 1)
    """

    # Set seed
    np.random.seed(seed)

    # Feature numbers and data storage init
    feature_numbers  = np.arange(1, num_features+1, 1)
    all_errors_test  = np.zeros(num_features)
    all_errors_train = np.zeros(num_features)

    # Initial sample of data, so far no index for test
    if data is None:
        data = r2_sampling(n_points)
    test_index = None

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
        
        all_errors_test[num-1]  = np.mean(single_fold_errors_test)
        all_errors_train[num-1] = np.mean(single_fold_errors_train)

    # Make plot of test and train errors
    plt.plot(feature_numbers[start:], all_errors_test[start:],    label = "Average test error",    c="g")
    plt.plot(feature_numbers[start:], all_errors_train[start:],    label = "Average train error",    c="r")

    # Mark finished degrees
    if mark_deg_nn:
        max_deg = maximal_degree(num_features)
        xy_deg_indeces = np.setdiff1d(nn_deg_indeces(max_deg), nn_deg_indeces(start))
        plt.plot(feature_numbers[xy_deg_indeces], all_errors_train[xy_deg_indeces], "o", c="r")

    # Add legend, x label and save
    plt.legend()
    plt.xlabel("Number of features")
    if filename:
        plt.savefig(filename)
    plt.show()

    return all_errors_test, all_errors_train


if __name__=="__main__":
    cross_validation_experiment(5, n_points=100, num_features=50)