from matplotlib import pyplot as plt
import numpy as np
from Code.utilities import (
    r2_sampling,
    feature_matrix,
    scale_feature_matrix,
    maximal_degree,
    nn_deg_indeces,
)
from Code.regression import fit_OLS, fit_LASSO, fit_RIDGE


def k_fold_split(X, Z, k):
    """Splits the feature matrix and target vector into k partitions.

    Args:
        X (Matrix): Feature matrix of dataset
        Z (ndarray): Target vector of dataset
        k (int): Number of folds to use for the k fold split.

    Returns:
        List: List of touples containing the different train and test splits.


    Notes:
        Let * be train and _ be test. The 5-fold split can be understood by:

            ****_ \n
            ***_* \n
            **_** \n
            *_*** \n
            _**** \n

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


methods_list = [
    {"fit_func": fit_OLS, "lam": 0, "name": "OLS", "styles": ["r", "r--"]},
    {"fit_func": fit_RIDGE, "lam": 0.01, "name": "RIDGE", "styles": ["b", "b--"]},
    {"fit_func": fit_LASSO, "lam": 0.01, "name": "LASSO", "styles": ["g", "g--"]},
]


def cross_validation_experiment(
    n_folds,
    method_dict_list=methods_list,
    n_points=100,
    num_features=100,
    data=None,
    mark_deg_nn=True,
    filename=None,
    seed=42,
    start=1,
):
    """Performs a cross validation experiment on all of the fitting methods in method_dict_list

    Args:
        n_folds (function): The function used to fit the data(train_RIDGE, train_LASSO)
        method_dict_list (List): List containing dictionaries of different methods with their names and potential lambda values.
        n_points (int, optional): Number of points to generate for the Franke function dataset.
        num_features (int): Maximum number of features to use for the models.
        data (dict, optional): Optional data dictionary containing custom data to use instead of Franke function. Needs to contain x, y and z arrays for datapoints. Defaults to None.
        mark_deg_nn (boolean, optional): Marks the points where the number of features containing x and number of features containing y are the same.
        filename (str, optional): Optional filename for where to save figure. Defaults to None.
        seed (int, optional): Seed to use for experiment. Defaults to 42.

    Returns:
        (ndarray, ndarray): ndarrays containing the test and train loss for different feature numbers respectively.

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
            label=f"Test error {name}",
        )
        plt.plot(
            feature_numbers,
            all_errors_train,
            styles[1],
            label=f"Train error {name}",
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
