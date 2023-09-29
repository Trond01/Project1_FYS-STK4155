import numpy as np

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
    n   = 0
    while True:
        for i in range(0, deg+1):

            # Add feature row
            X[:,n] = x[:, 0]**(deg-i) * y[:, 0]**i

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


def FrankeFunction(x,y):
  term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
  term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
  term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
  term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
  return term1 + term2 + term3 + term4# + np.random.normal(scale=0.1,size=x.shape)


def train_test_split(X, Y, percentage, test_index=None):
  """
  X: Feature matrix
  Y: Label vector(size=(n, 1))
  Percentage: How much of the dataset should be used as a test set.
  """
  n = X.shape[0]
  if test_index is None:
    test_index = np.random.choice(n, round(n*percentage), replace=False)
  test_X = X[test_index]
  test_Y = Y[test_index]

  train_X = np.delete(X, test_index, axis=0)
  train_Y = np.delete(Y, test_index, axis=0)

  return train_X, train_Y, test_X, test_Y, test_index


def scale_feature_matrix(X):
    means = np.mean(X, axis=0).reshape((1, X.shape[1]))
    means[0, 0] = 0
    var = np.var(X, axis=0).reshape((1, X.shape[1]))
    var[0, 0] = 1
    X_copy = np.copy(X - means)
    X_copy = X_copy/var
    return X_copy, means, var


def MSELoss(y, y_pred):
    return np.power(y - y_pred, 2).sum()/len(y)


def R_squared(y, y_pred):
    return 1 - MSELoss(y, y_pred)/MSELoss(y, y.mean())

def r2_sampling(num_points, seed=42):
    np.random.seed(42)
    x = np.random.random(( num_points, 1))
    y = np.random.random((num_points, 1))

    z = FrankeFunction(x, y)

    return {'x':x, 'y':y, 'z':z}


def prepare_feature_matrix(data, num_features, scale=True):
    x, y = data['x'], data['y']

    X = feature_matrix(x, y, num_features)
    means = 0
    var = 1
    if scale:
        X, means, var = scale_feature_matrix(X)

    return {'feature_matrix'  : X, 
            'means' : means, 
            'var'   : var,
            'num_features' : num_features} | data


def random_dataset_split_preparation(data, num_features, scale=True, test_index=None):
    x, y, z = data['x'], data['y'], data['z']
    
    data = prepare_feature_matrix(data, num_features, scale=scale)

    X = data['feature_matrix']

    train_X, train_Z, test_X, test_Z, test_index_new = train_test_split(X, z, 0.2, test_index=test_index)

    return {'train_X'  : train_X,
            'train_Z'  : train_Z,
            'test_X'   : test_X,
            'test_Z'   : test_Z,
            'test_index':test_index_new} | data
