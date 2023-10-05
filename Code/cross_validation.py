from matplotlib import pyplot as plt
import numpy as np
from Code.utilities import (
    r2_sampling,
    feature_matrix,
    scale_feature_matrix,
    maximal_degree,
    nn_deg_indeces,
)
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
    s = n // k  # Split size

    # Remove extra data (need divisible by k)
    X, Z = X[0 : s * k], Z[0 : s * k]

    # List for storing splits
    splits = []

    # Make splits
    for i in range(k):
        # Find indeces
        test_indeces = np.arange(i * s, (i + 1) * s, 1)
        train_indeces = np.delete(np.arange(0, k * s), test_indeces)

        # Append the fold
        splits.append(
            (
                X[train_indeces, :],
                Z[train_indeces, :],
                X[test_indeces, :],
                Z[test_indeces, :],
            )
        )

    return splits


def cross_validation_experiment(
    method_dict_list,
    n_folds,
    n_points=100,
    num_features=100,
    data=None,
    mark_deg_nn=True,
    filename=None,
    seed=42,
    start=1,
):
    """
    If using custom data, the data variable needs to be a dictionary with the fields {'x', 'y', 'z'}
    containing arrays with their respective data points. x, y, z should be of shape (number_points, 1)
    """

    # Set seed
    np.random.seed(seed)

    for meth in method_dict_list:
        fit_func = meth["fit_func"]
        lam = meth["lam"]
        name = meth["name"]
        styles = meth["styles"]

        # Feature numbers and data storage init
        feature_numbers = np.arange(start, num_features + 1, 1)
        all_errors_test = np.zeros(num_features - start + 1)
        all_errors_train = np.zeros(num_features - start + 1)

        # Initial sample of data, so far no index for test
        if data is None:
            data = r2_sampling(n_points)

        # Run experiment for each feature number
        for num in feature_numbers:
            X = feature_matrix(data["x"], data["y"], num)
            X, means, var = scale_feature_matrix(X)

            splits = k_fold_split(X, data["z"], n_folds)

            single_fold_errors_test = []
            single_fold_errors_train = []
            for train_X, train_Z, test_X, test_Y in splits:
                fold_data = {
                    "train_X": train_X,
                    "train_Z": train_Z,
                    "test_X": test_X,
                    "test_Z": test_Y,
                    "feature_matrix": X,
                    "means": means,
                    "var": var,
                    "num_features": num,
                    "x": data["x"],
                    "y": data["y"],
                    "z": data["z"],
                }
                results_data = fit_func(data=fold_data, plot_or_not=False, lam=lam)
                single_fold_errors_test.append(results_data["test_loss"])
                single_fold_errors_train.append(results_data["train_loss"])

            all_errors_test[num - start] = np.mean(single_fold_errors_test)
            all_errors_train[num - start] = np.mean(single_fold_errors_train)

        # Make plot of test and train errors
        plt.plot(
            feature_numbers,
            all_errors_test,
            styles[0],
            label=f"Average test error {name}",
        )
        plt.plot(
            feature_numbers,
            all_errors_train,
            styles[1],
            label=f"Average train error {name}",
        )

    # Mark finished degrees
    if mark_deg_nn:
        max_deg = maximal_degree(num_features)
        xy_deg_indeces = nn_deg_indeces(max_deg)
        xy_deg_indeces = xy_deg_indeces[xy_deg_indeces >= start - 1] - start + 1
        plt.plot(
            feature_numbers[xy_deg_indeces],
            all_errors_test[xy_deg_indeces],
            "o",
            c=styles[0][0],
        )

    # Add legend, x label and save
    plt.legend()
    plt.xlabel("Number of features")
    plt.title(f"Cross-validation with {n_folds} folds")
    if filename:
        plt.savefig(filename)
    plt.show()

    return all_errors_test, all_errors_train
