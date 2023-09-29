def plot_surface(x,y,z, filename=None):

    # Init figure
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')

    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
    if filename:
        plt.savefig(filename)


def plot_Franke(resolution = 101):
    # Make data.
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    x, y = np.meshgrid(x,y)

    z = FrankeFunction(x, y)

    plot_surface(x,y,z)


def plot_model_prediction(predict_func, num_features, means, var, filename=None):
    fig = plt.figure()

    num_plot = 101

    # Make data.
    x = np.linspace(0, 1, num_plot).reshape((num_plot, 1))
    y = np.linspace(0, 1, num_plot).reshape((num_plot, 1))

    x_mesh = np.zeros(shape=(num_plot*num_plot, 1))
    y_mesh = np.zeros(shape=(num_plot*num_plot, 1))

    for i in range(num_plot):
        for j in range(num_plot):
            x_mesh[j*num_plot + i] = x[i, 0]
            y_mesh[j*num_plot + i] = y[j, 0]
    
    plot_X = feature_matrix(x_mesh, y_mesh, num_features)
    
    plot_X = (plot_X - means)
    plot_X = plot_X/var

    xm, ym = np.meshgrid(x,y)

    z = predict_func(plot_X).reshape(xm.shape) # Must reshape (prediction is vector...)

    # Plot the surface.
    plot_surface(xm,ym,z, filename=filename)