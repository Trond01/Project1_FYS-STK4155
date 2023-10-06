import numpy as np
from imageio import imread
import os


def path_to_results(filename):
    PATH_TO_RESULTS = "."
    PATH_TO_RESULTS = os.path.join(PATH_TO_RESULTS, "..")
    PATH_TO_RESULTS = os.path.join(PATH_TO_RESULTS, "runs")
    PATH_TO_RESULTS = os.path.join(PATH_TO_RESULTS, "results")
    PATH_TO_RESULTS = os.path.join(PATH_TO_RESULTS, filename)
    return PATH_TO_RESULTS


def feature_matrix(x, y, num_features):
    """
    Generates a feature matrix based on x, y and the number of features.

    The function constructs features by taking combinations of powers of x and y,
    starting with the highest power of x.

    x: A 2D array with a single column of input values.
    y: A 2D array with a single column of input values.
    num_features: The number of feature columns.

    X: The feature matrix,a 2D numpy array with a column for each feature
    """

    X = np.zeros((len(x), num_features))

    deg = 0
    n = 0
    while True:
        for i in range(0, deg + 1):
            # Add feature row
            X[:, n] = x[:, 0] ** (deg - i) * y[:, 0] ** i

            # Increment number of points done, and end if needed
            n += 1
            if n == num_features:
                return X

        deg += 1


def f(x, y, beta: list, num_features) -> float:
    """
    x: float
    y: list of coefficients, [b00, b01, b02, ..., b0n, b10, b11, ..., b1n, ..., b2n, ...]
    """

    X = feature_matrix(x, y, num_features)
    return X @ beta


def FrankeFunction(x, y):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-((9 * x - 7) ** 2) / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-((9 * x - 4) ** 2) - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4  # + np.random.normal(scale=0.1,size=x.shape)


def train_test_split(X, Y, percentage, test_index=None):
    """
    X: Feature matrix
    Y: Label vector(size=(n, 1))
    Percentage: How much of the dataset should be used as a test set.
    """

    n = X.shape[0]
    if test_index is None:
        test_index = np.random.choice(n, round(n * percentage), replace=False)
    test_X = X[test_index]
    test_Y = Y[test_index]

    train_X = np.delete(X, test_index, axis=0)
    train_Y = np.delete(Y, test_index, axis=0)

    return train_X, train_Y, test_X, test_Y, test_index


def scale_feature_matrix(X):
    """Scales feature matrix using mean/variance scaling.

    Args:
        X (Matrix): Feature Matrix

    Returns:
        Matrix: Scaled feature matrix
    """
    means = np.mean(X, axis=0).reshape((1, X.shape[1]))
    means[0, 0] = 0
    var = np.var(X, axis=0).reshape((1, X.shape[1]))
    var[0, 0] = 1
    X_copy = np.copy(X - means)
    X_copy = X_copy / np.sqrt(var)
    return X_copy, means, var


def MSELoss(y, y_pred):
    """MSE loss of prediction array.

    Args:
        y (ndarray): Target values
        y_pred (ndarray): Predicted values

    Returns:
        float: MSE loss
    """
    return np.power(y - y_pred, 2).sum() / y.shape[0]


def R_squared(y, y_pred):
    """R squared value of prediction array.

    Args:
        y (ndarray): Target values
        y_pred (ndarray): Predicted values

    Returns:
        float: R squared
    """
    return 1 - ((np.sum(np.power(y - y_pred, 2))) / (np.sum(np.power(y - y.mean(), 2))))


# Functions to initialise data dictionary
def r2_sampling(num_points, sigma2=0.0):
    """Samples datapoints in the range (0, 1)x(0, 1) from the Franke function. Optionally adds noise to the sample.

    Args:
        num_points (int): Number of points to sample
        sigma2 (float, optional): Variance of the Gaussian where the noise is sampled. Defaults to 0.

    Returns:
        dict: Data dictionairy containing three arrays for the x, y and z coordinates of the points respectively.
    """
    x = np.random.random((num_points, 1))
    y = np.random.random((num_points, 1))

    z = FrankeFunction(x, y) + np.random.normal(
        0, np.sqrt(sigma2), size=(num_points, 1)
    )

    return {"x": x, "y": y, "z": z}


def prepare_feature_matrix(data, num_features, scale=True):
    """Takes in data dictionary and creates scaled feature matrix with given number of features.

    Args:
        data (dict): Data dictionary containing x and y arrays of datapoints.
        num_features (int): Number of features to use in the feature matrix
        scale (bool, optional): Scales feature matrix when true. Defaults to True.

    Returns:
        dict: Data dictionary containing feature matrix as well as means and variance used when scaling. Also contains number of features in the feature matrix.
    """
    x, y = data["x"], data["y"]

    X = feature_matrix(x, y, num_features)
    means = 0
    var = 1
    if scale:
        X, means, var = scale_feature_matrix(X)

    return {
        "feature_matrix": X,
        "means": means,
        "var": var,
        "num_features": num_features,
    } | data


def random_dataset_split_preparation(data, num_features, scale=True, test_index=None):
    """Takes data dictionary with x, y and z arrays. Creates feature matrix and splits it into train and test sets.

    Args:
        data (dict): Data dictionary containing x, y and z arrays of datapoints.
        num_features (int): Number of features
        scale (bool, optional): Scales feature matrix when true. Defaults to True.
        test_index (ndarray, optional): Optional indexing array used to partition into test and train sets. Defaults to None.

    Returns:
        dict: Data array containing train and test sets as well as index used to split.
    """
    x, y, z = data["x"], data["y"], data["z"]

    data = prepare_feature_matrix(data, num_features, scale=scale)

    X = data["feature_matrix"]

    train_X, train_Z, test_X, test_Z, test_index_new = train_test_split(
        X, z, 0.2, test_index=test_index
    )

    return {
        "train_X": train_X,
        "train_Z": train_Z,
        "test_X": test_X,
        "test_Z": test_Z,
        "test_index": test_index_new,
    } | data


def nn_deg_indeces(max_deg):
    """
    input:   total number of features
    returns: the indeces correspondining to feature number where the final term is of the form y^n:
                p(x) = a00 + a10 x + a01 y + ... + an0 x^n + ... + a0n y^n
             these are the indeces where we "complete a degree". See figure in report for more details.
    """
    return np.array(
        [(deg + 1) * (deg + 2) // 2 - 1 for deg in range(max_deg + 1)], dtype="int"
    )


def maximal_degree(num_features):
    """
    If N features we have N = 1+2+...+d+_ where (d-1) is the highest completed degree.

    returns: solution d-1 of the above equation
    """
    return (
        int((-1 + np.sqrt(1 + 8 * num_features)) / 2) - 1
    )  # solution of d^2+d<=2*feature_num


def number_of_features(max_deg):
    """
    Given max_deg = d, the number of features is 1 + 2 + 3 + ... + (d+1) = (d+1)(d+2)/2, by figure in the report
    """
    return (max_deg + 1) * (max_deg + 2) // 2


def load_terrain(filename):
    """
    Returns larges square of terrain
    """
    z = imread(filename).T  # To get ocean near...
    (ny, nx) = z.shape

    # Make grid.
    x = np.linspace(
        0, 1 * nx / ny, nx
    )  # Scale x axis to have same distance between points
    y = np.linspace(0, 1, ny)  # Scale y axis to 0,1
    x, y = np.meshgrid(x, y)

    return x, y, z / z.max()


def sample_terrain_data(
    n_points, filename="data_source/SRTM_data_Norway_1.tif", seed=42
):
    """Samples terrain data from USGS dataset.

    Args:
        n_points (int): Number of data points to sample.
        filename (str, optional): File from which to sample data from. Defaults to "data_source/SRTM_data_Norway_1.tif".
        seed (int, optional): Seed used to sample datapoints. Defaults to 42.

    Returns:
        dict: Data dictionary containing sampled x, y and z arrays.
    """
    # Set seed
    np.random.seed(seed)

    x, y, z = load_terrain(filename)

    z = z.astype(np.float32)

    x_array = x[0, :]
    y_array = y[:, 0]

    x_index = np.random.randint(0, x_array.shape[0], size=(n_points, 1))
    y_index = np.random.randint(0, y.shape[0], size=(n_points, 1))

    final_x = x_array[x_index]
    final_y = y_array[y_index]
    final_z = z[y_index, x_index]

    return {"x": final_x, "y": final_y, "z": final_z}
