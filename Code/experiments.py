from Code.utilities import r2_sampling, random_dataset_split_preparation, maximal_degree, nn_deg_indeces
from Code.regression import train_OLS
import numpy as np
from matplotlib import pyplot as plt




def plot_MSE_R2_experiment(train_method, num_features = 100, num_points = 151, lam=0, seed = 42, mark_deg_nn = True, filename=False, data = None, sigma2=0):
    
    # """
    # If using custom data, the data variable needs to be a dictionary with the fields {'x', 'y', 'z'}
    # containing arrays with their respective data points. x, y, z should be of shape (number_points, 1)
    # """
    # Set seed of choice
    np.random.seed(seed)

    # Initialise data storage
    feature_numbers = np.arange(1, num_features+1, 1)
    
    MRS_test  = np.zeros(num_features)
    MRS_train = np.zeros(num_features)
    R2_test   = np.zeros(num_features)
    R2_train  = np.zeros(num_features)
    
    # Sample data
    if data is None:
        data = r2_sampling(num_points, sigma2=sigma2)
    test_index = None # First preparation will make test index

    # Run experiment for each feature number
    for num in feature_numbers:

        # Create feature matrix        
        new_data = random_dataset_split_preparation(data, num, scale=True, test_index=test_index)
        test_index = new_data["test_index"] # Ensure same test split every time!

        # Do OLS fitting
        new_data = train_method(new_data, num_features, lam=lam)

        # Add the result
        MRS_test[num-1]  = new_data['test_loss']
        MRS_train[num-1] = new_data['train_loss']
        R2_test[num-1]   = new_data['test_loss_R2']
        R2_train[num-1]  = new_data['train_loss_R2']

    # Initialise plot
    fig, ax = plt.subplots(1,2, figsize=(10, 5))

    # Plot the MSE error
    ax[0].set_title("MSE error")
    ax[0].plot(feature_numbers, MRS_test,  label = "bias",     c="b")
    ax[0].plot(feature_numbers, MRS_train, label = "bias",     c="r")
    ax[0].set_xlabel("Number of features")

    # Plot R2 error
    ax[1].set_title("R2 error")    
    ax[1].plot(feature_numbers[1:], R2_test[1:],     label = "Test",     c="b")
    ax[1].plot(feature_numbers[1:], R2_train[1:],    label = "Train",    c="r") # Ignore degree 0 as RD=10^30...
    ax[1].set_xlabel("Number of features")

    # Highlight p(x)=a0+...+ann x^n y^n (whole number final)
    if mark_deg_nn:
        max_deg = maximal_degree(num_features)
        xy_deg_indeces = nn_deg_indeces(max_deg)
        ax[0].plot(feature_numbers[xy_deg_indeces], MRS_test[xy_deg_indeces], "o", c="b")
        ax[1].plot(feature_numbers[xy_deg_indeces[1:]], R2_test[xy_deg_indeces[1:]], "o", c="b")

    # Add legend, save if given filename. Show
    ax[1].legend()
    if filename:
        plt.savefig(filename)
    plt.show()

    return MRS_test, MRS_train,R2_test, R2_train


def plot_beta_experiment(train_method, beta_comp_indeces, num_features = 100, num_points = 151, lam=0, seed = 42, mark_deg_nn = True, filename=False, data = None, sigma2=0.0):
    """
    If using custom data, the data variable needs to be a dictionary with the fields {'x', 'y', 'z'}
    containing arrays with their respective data points. x, y, z should be of shape (number_points, 1)
    """
    # Set seed of choice
    np.random.seed(seed)

    # Initialise data storage
    feature_numbers = np.arange(1, num_features+1, 1)
    
    beta_vectors = []

    # Sample data
    if data is None:
        data = r2_sampling(num_points, sigma2=sigma2)
    test_index = None # First preparation will make test index

    # Run experiment for each feature number
    for num in feature_numbers:

        # Create feature matrix        
        new_data = random_dataset_split_preparation(data, num, scale=True, test_index=test_index)
        test_index = new_data["test_index"] # Ensure same test split every time!

        # Do OLS fitting
        new_data = train_method(new_data, num_features, lam=lam)

        # Add the new beta vector
        beta_vectors.append(new_data["weights"])


    # Make beta dictionary
    beta_dict = {}
    for i in range(len(beta_vectors)):
        beta_dict[f"{i}"] = {  "fn" : feature_numbers[i:], 
                                "bi" : [beta[i] for beta in beta_vectors[i:]]}


    for i in beta_comp_indeces:
        fn = beta_dict[f"{i}"]["fn"]
        bi = beta_dict[f"{i}"]["bi"]
        subscript = ''.join(chr(0x2080 + int(digit)) for digit in str(i)) # Convert 15 to _{15}
        plt.plot(fn, bi, label = rf"$\beta${subscript}")
    

    # Add legend, save if given filename. Show
    plt.legend()
    plt.title("Magnitude of parameters")
    plt.xlabel("Number of features")
    if filename:
        plt.savefig(filename)
    plt.show()

    return beta_vectors





def analyze_lambda_range(train_method,  high_deg, lam_low, lam_high, num_lam, num_points=101, seed=42, nth=3, data=None, filename=None, sigma2=0.0):
    """_summary_

    Args:
        method (function): The function used to fit the data(train_RIDGE, train_LASSO)
        num_points (_type_): _description_
        high_deg (_type_): _description_
        lam_low (_type_): _description_
        lam_high (_type_): _description_
        num_lam (_type_): _description_
        seed (int, optional): _description_. Defaults to 42.
        nth (int, optional): _description_. Defaults to 3.
        data (_type_, optional): _description_. Defaults to None.
        filename (str, optional): _description_. Defaults to "lambda_range.png".
    """
    
    # Set seed and sample
    np.random.seed(seed)
    data = r2_sampling(num_points, sigma2=sigma2) if data is None else data
    
    # Initialise values for experiment
    all_lambdas = np.logspace(lam_low, lam_high, num_lam)
    all_degrees = np.array(range(2, high_deg))
    lams, degs = np.meshgrid(all_lambdas, all_degrees)
    
    # Initialise storage for errors
    train_errors = np.zeros_like(lams)
    test_errors = np.zeros_like(lams)
    test_index = None

    # Initialise figure
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Loop over all lambdas and degrees
    for i in range(lams.shape[0]):
        for j in range(lams.shape[1]):
            fit_data = train_method(data, degs[i, j], lams[i, j], plot_or_not=False, test_index=test_index)
            test_index = fit_data.get('test_index')
            train_errors[i, j] = fit_data['train_loss']
            test_errors[i, j] = fit_data['test_loss']
        
        if i%nth == 0:
            axs[0].plot(lams[i, :], train_errors[i, :], label=f"{degs[i,0]}")
            axs[1].plot(lams[i, :], test_errors[i, :], label=f"{degs[i,0]}")

    # Customize plot
    axs[0].set(title="Train", xlabel="Lambda", ylabel="MSE", xscale="log")
    axs[1].set(title="Test", xlabel="Lambda", xscale="log")
    axs[0].legend(title="Features")
    axs[1].legend(title="Features")
    
    # Ensure no overlap, save and show
    plt.tight_layout() 
    if filename is not None:
        plt.savefig(filename)
    plt.show()