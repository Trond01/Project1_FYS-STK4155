import numpy as np
from Code.utilities import FrankeFunction, feature_matrix
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def plot_Franke(resolution=101, filename=None):
    """Plots surface of Franke function.

    Args:
        resolution (int, optional): How granular the surface plot is. Defaults to 101.
    """
    # Make data.
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    x, y = np.meshgrid(x, y)

    z = FrankeFunction(x, y)

    plot_surface(x, y, z, filename=filename)


def plot_surface(x, y, z, filename=None):
    """Plots surface with a

    Args:
        x (meshgrid): x values for surface plot.
        y (meshgrid): y values for surface plot.
        z (meshgrid): z values for surface plot.
        filename (str, optional): Optional filename to save surface plot to. Defaults to None.
    """
    # Init figure
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    if filename:
        plt.savefig(filename)

    plt.show()


def plot_model_prediction(predict_func, num_features, means, var, filename=None):
    """Takes in function predicting z values for a given place in x and y grid.

    Args:
        predict_func (function): Function used to create z values for surface.
        num_features (int): Number of features used in model.
        means (ndarray): Means used to scale feature matrix in model training.
        var (ndarray): Variance used to scale feature matrix in model training.
        filename (str, optional): Optional filename to save plot to. Defaults to None.
    """
    fig = plt.figure()

    num_plot = 101

    # Make data.
    x = np.linspace(0, 1, num_plot).reshape((num_plot, 1))
    y = np.linspace(0, 1, num_plot).reshape((num_plot, 1))

    x_mesh = np.zeros(shape=(num_plot * num_plot, 1))
    y_mesh = np.zeros(shape=(num_plot * num_plot, 1))

    for i in range(num_plot):
        for j in range(num_plot):
            x_mesh[j * num_plot + i] = x[i, 0]
            y_mesh[j * num_plot + i] = y[j, 0]

    plot_X = feature_matrix(x_mesh, y_mesh, num_features)

    plot_X = plot_X - means
    plot_X = plot_X / var

    xm, ym = np.meshgrid(x, y)

    z = predict_func(plot_X).reshape(xm.shape)  # Must reshape (prediction is vector...)

    # Plot the surface.
    plot_surface(xm, ym, z, filename=filename)
