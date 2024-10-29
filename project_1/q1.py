import numpy as np
import scipy.stats as stats
import seaborn as sea
import matplotlib.pyplot as plt
from os import mkdir, path

def calculate_ks(cdf_1, cdf_2):
    return np.max(np.abs(cdf_1 - cdf_2))


def truncated_norm(a, b, mu, sigma, N):
    """
    Obtain empirical cdf of truncated normal distribution (a, b) and plot it against the true cdf

    Parameters
    ----------
    a : float
        Lower bound of the truncated normal distribution
    b : float
        Upper bound of the truncated normal distribution
    mu : float
        Mean of the normal distribution
    sigma : float   
        Standard deviation of the normal distribution
    N : int
        Number of samples

    Returns
    -------
    X : array
        Random samples from truncated normal distribution
    F_empirical : array
        Empirical CDF
    F_exact : array
        True CDF  
    """
    # Generate random samples from truncated normal distribution
    a_truncated, b_truncated = (a - mu) / sigma, (b - mu) / sigma
    X = stats.truncnorm.rvs(a_truncated, b_truncated, loc=mu, scale=sigma, size=N)
    
    # Empirical CDF
    X = np.sort(X)
    F_empirical = np.arange(1, N+1) / N
    
    # True CDF  
    F_exact_tn = stats.truncnorm.cdf(X, a_truncated, b_truncated, loc=mu, scale=sigma)
    F_exact_as = stats.arcsine.cdf(X, loc=a, scale=b-a)
    F_exact_uni = stats.uniform.cdf(X, loc=a, scale=b-a)
    F_exact = [F_exact_tn, F_exact_as, F_exact_uni]

    return X, F_empirical, F_exact


def arcsine_dist(a, b, N, mu=0, sigma=0):
    """
    Obtain empirical cdf of arcsine distribution (a, b) and plot it against the true cdf

    Parameters
    ----------
    a : float
        Lower bound of the arcsine distribution
    b : float
        Upper bound of the arcsine distribution
    N : int
        Number of samples

    Returns
    -------
    X : array
        Random samples from arcsine distribution
    F_empirical : array
        Empirical CDF
    F_exact : array
        True CDF   
    """
    # Generate random samples from arcsine distribution
    X = stats.arcsine.rvs(loc=a, scale=b-a, size=N)
    
    # Empirical CDF
    X = np.sort(X)
    F_empirical = np.arange(1, N+1) / N
    
    # True CDF 
    a_truncated, b_truncated = (a - mu) / sigma, (b - mu) / sigma 
    F_exact_tn = stats.truncnorm.cdf(X, a_truncated, b_truncated, loc=mu, scale=sigma)
    F_exact_as = stats.arcsine.cdf(X, loc=a, scale=b-a)
    F_exact_uni = stats.uniform.cdf(X, loc=a, scale=b-a)    
    F_exact = [F_exact_tn, F_exact_as, F_exact_uni]
    
    return X, F_empirical, F_exact


def uniform_dist(a, b, N, mu=0, sigma=0):
    """
    Obtain empirical cdf of uniform distribution (a, b) and plot it against the true cdf

    Parameters
    ----------
    a : float
        Lower bound of the uniform distribution
    b : float
        Upper bound of the uniform distribution
    N : int
        Number of samples

    Returns
    -------
    X : array
        Random samples from uniform distribution
    F_empirical : array
        Empirical CDF
    F_exact : array
        True CDF   
    """
    # Generate random samples from uniform distribution
    X = stats.uniform.rvs(loc=a, scale=b-a, size=N)
    
    # Empirical CDF
    X = np.sort(X)
    F_empirical = []
    for i in range(N):
        F_empirical.append((i+1) / N)
    
    # True CDF  
    a_truncated, b_truncated = (a - mu) / sigma, (b - mu) / sigma
    F_exact_tn = stats.truncnorm.cdf(X, a_truncated, b_truncated, loc=mu, scale=sigma)
    F_exact_as = stats.arcsine.cdf(X, loc=a, scale=b-a)
    F_exact_uni = stats.uniform.cdf(X, loc=a, scale=b-a)
    F_exact = [F_exact_tn, F_exact_as, F_exact_uni]
   
    return X, F_empirical, F_exact


def plot_cdfs(dist_name, X, F_empirical, F_exact, ks_value=None):
    """
    Plot empirical and true cdfs
    
    Parameters
    ----------
    dist_name : str
        Name of the distribution
    X : array
        Random samples
    F_empirical : array
        Empirical CDF
    F_exact : array
        True CDF
    """
    # Calculate KS value 
    if ks_value is None:
        ks_value = np.max(np.abs(F_empirical - F_exact))
    
    diff = F_empirical - F_exact
    
    # plot empirical and true cdfs in one subplot and the difference in another
    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    plt.suptitle(f'Empirical and True CDFs of {dist_name} with N = 10^{int(np.log10(N))} samples')
    ax[0].plot(X, F_empirical, label='Empirical CDF')
    ax[0].plot(X, F_exact, label='True CDF')
    # add a box
    ax[0].text(0.75, 0.01, f'KS value: {ks_value:.6f}', fontsize=12, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))


    ax[0].set_title(f'Empirical and True CDFs of {dist_name}')
    ax[0].set_xlabel('X')

    ax[0].legend()

    ax[1].plot(X, diff, label='Difference')
    ax[1].set_title('Difference between Empirical and True CDFs')


    ax[1].set_xlabel('X')
    ax[1].legend()
    
    plt.tight_layout()
    plt.savefig(rf'Images/Q1/N=10^{int(np.log10(N))}_{dist_name}.png')
    plt.close()
    return None


def plot_ks(n_samples, ks_values, trials=1):
    """
    Plot KS values against number of samples

    Parameters
    ----------
    n_samples : array
        Number of samples
    ks_values : array
        KS values
    
    Returns
    -------
    None
    """

    fig, ax = plt.subplots()
    slope, intercept = np.polyfit(np.log(n_samples), np.log(ks_values[0]), 1)
    ax.plot(n_samples, ks_values[0], label=f'Truncated Normal: Slope {slope:.2f}')

    slope, intercept = np.polyfit(np.log(n_samples), np.log(ks_values[1]), 1)  
    ax.plot(n_samples, ks_values[1], label=f'Arcsine: Slope {slope:.2f}')   

    slope, intercept = np.polyfit(np.log(n_samples), np.log(ks_values[2]), 1)
    ax.plot(n_samples, ks_values[2], label=f'Uniform: Slope {slope:.2f}')

    ax.set_xscale('log')    
    ax.set_yscale('log')    

    ax.set_xlabel('Number of samples')
    ax.set_ylabel('KS value')   
    ax.set_title(f'KS value vs Number of samples for {trials} trials')   
    ax.legend()

    plt.tight_layout()
    plt.savefig(rf'Images/Q1/KS_values.png')
    plt.close()
    return None


def confusion_matrix(empirical_cdf, exact_cdfs):
    """
    Calculate confusion matrix for KS values

    Parameters
    ----------
    empirical_cdf : array
        Empirical CDFs
    exact_cdfs : array
        True CDFs
    
    Returns
    -------
    confusion_matrix_row : array
    """

    confusion_matrix_row = np.zeros(3)
    for j in range(3):
            confusion_matrix_row[j] = calculate_ks(empirical_cdf, exact_cdfs[j])
    return confusion_matrix_row


def print_confusion_matrix(ks_values, N, trials=1):
    print("\n")
    print(f"Mean Confusion Matrix for N = {N} & {trials} trials")
    print(f"\t\tTN_true      AS_true      UNI_true")
    print(f"TN_emp\t\t{ks_values[0][0]:.6f}      {ks_values[0][1]:.6f}      {ks_values[2][0]:.6f}")
    print(f"AS_emp\t\t{ks_values[1][0]:.6f}      {ks_values[1][1]:.6f}      {ks_values[2][1]:.6f}")
    print(f"UNI_emp\t\t{ks_values[2][0]:.6f}      {ks_values[2][1]:.6f}      {ks_values[2][2]:.6f}")


# Create a directory to save images
if not path.exists('Images'):
    mkdir('Images')
    mkdir('Images/Q1')
elif not path.exists('Images/Q1'):
    mkdir('Images/Q1')

# Parameters
max_trials = 100
ks_values_in_trial = []
ks_values_for_n = []
n_samples = [10**i for i in range(2, 5)]


for N in n_samples:
    trial = 1
    
    while trial <= max_trials:
        ks_values = [[] for _ in range(3)]

        # Generate random samples from truncated normal distribution  and plot empirical and true cdfs
        mu, sigma = 1/2, 1/6
        X1, F_empirical_tn, F_exact_1 = truncated_norm(a=0, b=1, mu=mu, sigma=sigma, N=N)
        

        # Generate random samples from arcsine distribution  and plot empirical and true cdfs
        X2, F_empirical_as, F_exact_2 = arcsine_dist(a=0, b=1, mu=mu, sigma=sigma, N=N) 
          

        # Generate random samples from uniform distribution  and plot empirical and true cdfs
        X3, F_empirical_uni, F_exact_3 = uniform_dist(a=0, b=1, mu=mu, sigma=sigma, N=N)
          

        # Calculate Confusion matrix for KS values
        ks_values[0] =  confusion_matrix(F_empirical_tn, F_exact_1)
        ks_values[1] =  confusion_matrix(F_empirical_as, F_exact_2)
        ks_values[2] =  confusion_matrix(F_empirical_uni, F_exact_3)

        # Append for each trial
        ks_values_in_trial.append(ks_values)
        trial += 1
    
    # Mean KS values for N
    ks_values_for_n.append(np.mean(ks_values_in_trial, axis=0))
    plot_cdfs('truncated_normal', X1, F_empirical_tn, F_exact_1[0], ks_values_for_n[-1][0][0])
    plot_cdfs('arcsine', X2, F_empirical_as, F_exact_2[1], ks_values_for_n[-1][1][1]) 
    plot_cdfs('uniform', X3, F_empirical_uni, F_exact_3[2], ks_values_for_n[-1][2][2]) 
    print_confusion_matrix(ks_values_for_n[-1], N, trials=max_trials)

    # Confusion of KS values
    sea.heatmap(ks_values_for_n[-1], annot=True, fmt=".4f", cmap='Blues', xticklabels=['TN_true', 'AS_true', 'UNI_true'], yticklabels=['TN_emp', 'AS_emp', 'UNI_emp'])
    plt.title(f'KS values for N = 10^{int(np.log10(N))} & {max_trials} trials')
    plt.savefig(rf'Images/Q1/confusion_matrix_N=10^{int(np.log10(N))}_{max_trials}trials.png')
    plt.close()


# Plot KS values against number of samples
ks_trunc = [ks_values_for_n[i][0][0] for i in range(len(n_samples))]
ks_arcsine = [ks_values_for_n[i][1][1] for i in range(len(n_samples))]
ks_uniform = [ks_values_for_n[i][2][2] for i in range(len(n_samples))]

plot_ks(n_samples, [ks_trunc, ks_arcsine, ks_uniform], trials=max_trials)