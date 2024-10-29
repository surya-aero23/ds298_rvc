import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from os import mkdir, path, makedirs
from time import time


'Create a directory to save images. Create folder "Images" and subfolder "q1" if they dont exist'
def create_directory():
    """Create a directory to save images. Create folder "Images" and subfolder "q1" if they dont exist

    Returns:
    None
    """
    # if not path.exists('Images'):
    #     mkdir('Images')
    # if not path.exists('Images/q1'):
    #     mkdir('Images/q1')
    
    class_num = [1, 2, 3]
    algo = [2, 3]
    folder_names = ['rejection sampling', 'pmf', 'error']

    for i in class_num:
        for alg in algo:
            for folder in folder_names:
                directory = f'Images/q1/class{i}/algo_{alg}/{folder}'
                if not path.exists(directory):
                    makedirs(directory)
    return None


'Number of entries of matrix and number of samples for matrix multiplication'
def size_and_samples(n_start: int = 50, n_end: int = 1600,  do_print: bool = False):
    """Number of entries of matrix and number of samples for matrix multiplication

    Args:
    n_start (int, optional): starting number of entries of matrix. Defaults to 50.
    n_end (int, optional): ending number of entries of matrix. Defaults to 1600.
    do_print (bool, optional): print the values. Defaults to True.

    Returns:
    n_list: number of entries of matrix
    c_list: number of samples for matrix multiplication
    tuple: (n_list, c_list)
    """
    value = n_start
    n_list = []
    while value <= n_end:
        n_list.append(value)
        value *= 2
    
    c_list = []
    for i in range(len(n_list)):
        # round to the nearest integer
        entry_1 = int(np.log2(n_list[i]) + 0.5)
        entry_2 = int((np.log2(n_list[i]) ** 2) + 0.5)
        entry_3 = int((0.2 * n_list[i]) + 0.5)

        c_list.append([entry_1, entry_2, entry_3])
    
    if do_print:
        print(f'n_list: {n_list}\nc_list: {c_list}')
    
    return n_list, c_list


'Function to obtain singular values for class 1'
def class_1_sv(n: int, k: list):
    """Class 1 singular values
    
    Args:
    n (int): number of singular values
    k (list): indices of singular values

    Returns:
    list: singular values
    """
    return np.exp(-10 * (np.add(k, -1))/ n)


'Function to obtain singular values for class 2'
def class_2_sv(n: int, k: list):
    """Class 2 singular values

    Args:
    n (int): number of singular values
    k (list): indices of singular values

    Returns:
    list: singular values
    """
    return -1 * np.add(-n-1, k) / n


'Function to obtain singular values for class 3'
def class_3_sv(n: int, k: list):
    """Class 3 singular values

    Args:
    n (int): number of singular values
    k (list): indices of singular values

    Returns:
    list: singular values
    """
    return np.log(-1 * np.add(-n-1, k))/np.log(n)


'Compare the singular values of the matrices as a plot'
def plot_singular_vals(n_list: list, k_factor: int = 2, savefig: bool = True):
    """Compare singular values of the matrices for different classes

    Args:
    n_list (list): list of number of singular values
    k_factor (int, optional): factor to divide n to calculate k. Defaults to 2.
    savefig (bool, optional): save the figure. Defaults to True.

    Returns:
    None    
    """
    fig, axs = plt.subplots(2, 3, figsize=(10, 10))
    ax_list = axs.ravel()
    fig.suptitle('Singular Values of Matrix of Different Classes', fontsize=16)

    for i in range(len(n_list)):
        n = n_list[i]
        k = int(n / k_factor)
        k_list = np.arange(1, k+1)

        sv_1 = class_1_sv(n, k_list)
        sv_2 = class_2_sv(n, k_list)
        sv_3 = class_3_sv(n, k_list)

        ax_list[i].plot(k_list, sv_1, label='Class 1')
        ax_list[i].plot(k_list, sv_2, label='Class 2')
        ax_list[i].plot(k_list, sv_3, label='Class 3')
        ax_list[i].set_title(f'n = {n}')
        
        # label x only for the last row
        if i >= 3:
            ax_list[i].set_xlabel('k', fontsize=12)
        
        # label y only for the first column
        if i % 3 == 0:
            ax_list[i].set_ylabel('Singular Value', fontsize=12)
        
        # fix the legend at right top for each axis
        ax_list[i].legend(loc='upper right')
        ax_list[i].grid()

    

    if savefig:
        plt.savefig('Images/q1/singular_values_all_n_all_classes.png')
    
    if not savefig:
        plt.show()

    plt.close()
    return None


'Algorithm 1'
def algorithm_1(A, B, indices, pmf):
    """Matrix multiplication using algorithm 1

    Args:
    A (np.ndarray): matrix A
    B (np.ndarray): matrix B
    indices (list): list of indices    
    pmf (list): list of PMF

    Returns:
    np.ndarray: matrix multiplication result
    """
    n = A.shape[0]
    true_error, relative_error = [], []
    c_count = len(indices)

    for t in range(c_count):
        c = len(indices[t])
        M = np.zeros((n, n))

        for i in range(c):
            M = np.add(M, np.outer(A[:, indices[t][i]], B[indices[t][i], :]) / (c * pmf[indices[t][i]]))
        
        true_error.append(np.linalg.norm(A @ B - M, ord='fro'))
        relative_error.append(true_error[-1] / np.linalg.norm(A @ B, ord='fro'))
    
    return M, true_error, relative_error


'Algorithm 1 caller for a class'
def algo_1_caller(class_matrices_list: list, indices_list: list, pmf_list: list):
    """Given the list containing pairs of matrices, indices and the pmf, do the matrix multiplication using algorithm 1

    Args:
    class_matrices_list (list): list of matrices belonging to one class
    indices
    indices_list (list): list of indices
    pmf_list (list): list of PMF lists

    Returns:
    list: list of matrix multiplication results
    """
    true_error, rel_error = [], []
    
    for i in range(len(class_matrices_list)):
        _, true_error_for_matrix, rel_error_for_matrix = algorithm_1(class_matrices_list[i][0], class_matrices_list[i][1], indices_list[i], pmf_list[i])
        true_error.append(true_error_for_matrix)
        rel_error.append(rel_error_for_matrix)
    
    return true_error, rel_error


'Algorithm 2'
def algorithm_2(n: int, sv: list):
    """Generate random matrix using the singular values

    Args:
    n (int): number of singular values
    sv (list): singular values

    Returns:
    list: list of random matrices
    """
    matrices = []
    for i in range(2):
        # Use uniform distribution between 0 and 1 to generate matrices which you can QR to get U and V
        U, _ = np.linalg.qr(np.random.rand(n, n))
        V, _ = np.linalg.qr(np.random.rand(n, n))
        matrices.append(U @ np.diag(sv) @ V.T)
    return matrices


'Algorithm 3'
def algorithm_3(n: int, sv: list):
    """Generate random matrix using the singular values

    Args:
    n (int): number of singular values
    sv (list): singular values

    Returns:
    list: list of pair of random matrices
    """
    Q1, _ = np.linalg.qr(np.random.rand(n, n))
    Q2, _ = np.linalg.qr(np.random.rand(n, n))
    Q3, _ = np.linalg.qr(np.random.rand(n, n))
    s = np.diag(sv)

    A = Q1 @ s @ Q2.T
    B = Q2 @ s @ Q3.T
    
    return [A, B]


'Generate A and B matrices for different classes'
def generate_trial_matrices(n_list: list, do_print: bool = False, alg: int = 3):
    """Generate A and B matrices for different classes
    
    Args:
    n_list (list): list of number of singular values
    do_print (bool, optional): print the values. Defaults to True.
    alg (int, optional): algorithm to use (2 or 3). Defaults to 3.

    Returns:
    class1_matrices_list: list of A and B matrices for class 1
    class2_matrices_list: list of A and B matrices for class 2
    class3_matrices_list: list of A and B matrices for class 3
    tuple: (class1_matrices_list, class2_matrices_list, class3_matrices_list)
    """

    class1_matrices_list, class2_matrices_list, class3_matrices_list = [], [], []
    
    # start the timer
    start = time()
    if alg == 2:
        for i in range(len(n_list)):
            A, B = algorithm_2(n_list[i], sv=class_1_sv(n_list[i], [i for i in range(1, n_list[i]+1)]))
            class1_matrices_list.append([A, B])

            A, B = algorithm_2(n_list[i], sv=class_2_sv(n_list[i], [i for i in range(1, n_list[i]+1)]))
            class2_matrices_list.append([A, B])

            A, B = algorithm_2(n_list[i], sv=class_3_sv(n_list[i], [i for i in range(1, n_list[i]+1)]))
            class3_matrices_list.append([A, B])
    
    if alg == 3:
        for i in range(len(n_list)):
            A, B = algorithm_3(n_list[i], sv=class_1_sv(n_list[i], [i for i in range(1, n_list[i]+1)]))
            class1_matrices_list.append([A, B])

            A, B = algorithm_3(n_list[i], sv=class_2_sv(n_list[i], [i for i in range(1, n_list[i]+1)]))
            class2_matrices_list.append([A, B])

            A, B = algorithm_3(n_list[i], sv=class_3_sv(n_list[i], [i for i in range(1, n_list[i]+1)]))
            class3_matrices_list.append([A, B])
    
    # print the dimensions of the matrices and the number of sets matrices
    if do_print:
        print('- - '* 15)
        print(f'Class 1: {len(class1_matrices_list)} sets of matrices for n = {[class1_matrices_list[i][0].shape[0] for i in range(len(class1_matrices_list))]}')
        print(f'Class 2: {len(class2_matrices_list)} sets of matrices for n = {[class2_matrices_list[i][0].shape[0] for i in range(len(class2_matrices_list))]}')
        print(f'Class 3: {len(class3_matrices_list)} sets of matrices for n = {[class3_matrices_list[i][0].shape[0] for i in range(len(class3_matrices_list))]}')   
        print('- - '* 15)
        print(f'\tTime to generate all matrices: {time() - start:.2f} seconds')
        print('- - '* 15)

    return class1_matrices_list, class2_matrices_list, class3_matrices_list


'Generate PMF given two random matrices'
def generate_pmf_for_matrices(A: np.ndarray, B: np.ndarray):
    """Generate PMF given two random matrices

    Args:
    A (np.ndarray): matrix A
    B (np.ndarray): matrix B

    Returns:
    list: list of PMF
    """
    pmf = []
    for i in range(A.shape[1]):
        pmf.append(np.linalg.norm(A[:, i], ord=2) * np.linalg.norm(B[i, :], ord=2))
    
    # normalize it with sum of all elements
    pmf = np.array(pmf) / np.sum(pmf)
    return pmf


'Generate PMF for a given class list of matrices'
def generate_pmf_for_class(class_matrices_list: list, class_num: int, saveplot: bool = True, alg: int = 0):
    """Generate PMF for a given class list of matrices

    Args:
    class_matrices_list (list): list of matrices
    class_num (int): class number
    saveplot (bool, optional): save the plot. Defaults to True.

    Returns:
    list: list of PMFs for all pairs of matrices in class_matrices_list
    """
    pmf_list = []
    for i in range(len(class_matrices_list)):
        pmf = generate_pmf_for_matrices(class_matrices_list[i][0], class_matrices_list[i][1])
        
        if saveplot:
            directory = f'Images/q1/class{class_num}/algo_{alg}/pmf'
            if not path.exists(directory):
                makedirs(directory)
            
            plot_pmf(pmf, savefig=True, title=f'pmf_class{class_num}_n={class_matrices_list[i][0].shape[0]}', directory=directory)
        
        pmf_list.append(pmf)
    return pmf_list


'Plot a PMF'
def plot_pmf(pmf: list, savefig: bool = True, title: str = 'pmf', directory: str = 'Images/q1'):
    """Plot a PMF

    Args:
    pmf (list): list of PMF
    savefig (bool, optional): save the figure. Defaults to True.
    title (str, optional): title of the plot. Defaults to 'pmf'.
    directory (str, optional): directory to save the plot. Defaults to 'Images/q1'.

    Returns:
    None
    """
    plt.figure(figsize=(10, 5))
    plt.plot(pmf, label='PMF', marker='o', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel('Column Index')
    plt.ylabel('PMF Value')
    plt.yticks(np.arange(0, max(pmf) + 0.5 * max(pmf), 0.1 * max(pmf)))
    
    # Printing sum of all pmf inside a box
    plt.text(0.4 * len(pmf), max(pmf) + 0.3 * max(pmf), f'Sum of PMF: {np.sum(pmf):.3f}', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    plt.legend()

    if savefig:
        plt.savefig(f'{directory}/{title}.png')
    
    if not savefig:
        plt.show()

    plt.close()
    return None


'Given a list for pmf and c, pick c columns based on uniform rejection sampling'
def rejection_sampling(pmf: list, c: int, class_num: int, col_num: int, saveplot: bool = True, alg: int = 0):
    """Given a list for pmf and c, pick c columns based on uniform rejection sampling

    Args:
    pmf (list): list of PMF
    c (int): number of columns to pick
    class_num (int): class number
    col_num (int): column number (choice 1, 2, 3)
    saveplot (bool, optional): save the plot. Defaults to True.

    Returns:
    list: list of indices of columns
    """
    indices = []
    u2_list = []
    front_coeff = max(pmf)
    for i in range(c):
        while True:
            index = np.random.randint(0, len(pmf))
            u2 = np.random.rand()
            if u2 < pmf[index] / front_coeff:
                indices.append(index)
                u2_list.append(u2)
                break
    
    if saveplot:
        col_type = ['log2n', 'sq(log2n)', '0.2n']
        directory = f'Images/q1/class{class_num}/algo_{alg}/rejection sampling'
        
        # if not path.exists(directory):
        #     makedirs(directory)
        
        plt.plot(pmf / max(pmf), label='PMF ratio', marker='o', linestyle='-', color='b')
        plt.scatter(indices, u2_list, color='r', label='Selected Columns', marker='^')
        plt.title(f'Rejection Sampling for Class {class_num}, n={len(pmf)}, col={col_type[col_num-1]} = {c}')
        plt.xlabel('Column Index')
        plt.ylabel('PMF Value')
        # plt.xticks(range(0, len(pmf)+1, int(len(pmf)/10)))
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(f'{directory}/rejection_sampling_class{class_num}_n={len(pmf)}_c={col_type[col_num-1]}={c}.png')
        plt.close()
    return indices


'Do rejection sampling for all given pmf, class, ctype and c'
def rejection_sampling_all(pmf_list: list, c_list: list, class_num: int, saveplot: bool = True, alg: int = 0):
    """Do rejection sampling for all given lists of pmf, class, ctype and c

    Args:
    pmf_list (list): list of PMF lists
    c_list (list): list of c lists for all n
    class_num (int): class number
    saveplot (bool, optional): save the plot. Defaults to True.

    Returns:
    indices: list of indices of columns for each c in c_list
    """
    indices_for_class = []
    for j in range(len(pmf_list)):
        index_for_c_list = []
        for i in range(len(c_list[j])):
            index_for_c_list.append(rejection_sampling(pmf=pmf_list[j], c=c_list[j][i], class_num=class_num, col_num=i+1, saveplot=saveplot, alg=alg))
        
        indices_for_class.append(index_for_c_list)
    return indices_for_class


'Plot the trial average of the errors for all classes'
def plot_average_errors(class_error_list: list, n_list: list, trials: int, error_name: str = 'relative', saveplot: bool = True, directory: str = 'Images/q1', alg: int = 0):
    """Plot the trial average of the errors for all classes

    Args:
    class_error_list (list): list of true and relative errors for all classes.
    n_list (list): list of sizes of matrices.
    trials (int): number of trials.
    error_name (str, optional): name of the error. Defaults to 'relative'.  
    saveplot (bool, optional): save the plot. Defaults to True.

    Returns:
    None
    """
    
    class1, class2, class3 = class_error_list[0], class_error_list[1], class_error_list[2]

    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    ax = axs.ravel()

    # for class 1
    ax[0].plot(n_list, class1[0], label='log2(n)', marker='o', linestyle='-', color='b')
    ax[0].plot(n_list, class1[1], label='sq(log2(n))', marker='o', linestyle='-', color='r')
    ax[0].plot(n_list, class1[2], label='0.2n', marker='o', linestyle='-', color='g')
    ax[0].set_title(f'Class 1 {error_name} error')
    ax[0].set_xlabel('n')
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_ylabel(f'{error_name.capitalize()} Error')
    ax[0].legend()
    ax[0].grid()

    # for class 2
    ax[1].plot(n_list, class2[0], label='log2(n)', marker='o', linestyle='-', color='b')
    ax[1].plot(n_list, class2[1], label='sq(log2(n))', marker='o', linestyle='-', color='r')
    ax[1].plot(n_list, class2[2], label='0.2n', marker='o', linestyle='-', color='g')
    ax[1].set_title(f'Class 2 {error_name} error')
    ax[1].set_xlabel('n')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_ylabel(f'{error_name.capitalize()} Error')    
    ax[1].legend()
    ax[1].grid()

    # for class 3
    ax[2].plot(n_list, class3[0], label='log2(n)', marker='o', linestyle='-', color='b')
    ax[2].plot(n_list, class3[1], label='sq(log2(n))', marker='o', linestyle='-', color='r')
    ax[2].plot(n_list, class3[2], label='0.2n', marker='o', linestyle='-', color='g')   
    ax[2].set_title(f'Class 3 {error_name} error')
    ax[2].set_xlabel('n')
    ax[2].set_xscale('log')
    ax[2].set_yscale('log') 
    ax[2].set_ylabel(f'{error_name.capitalize()} Error')
    ax[2].legend()
    ax[2].grid()

    fig.suptitle(f'{trials} Trial Averaged {error_name.capitalize()} Error for All Classes', fontsize=16)
    plt.tight_layout()

    if saveplot:
        plt.savefig(f'{directory}/algo{alg}_{trials}trial_{error_name}_error.png') 
    
    if not saveplot:    
        plt.show()
    
    plt.close()
    return None



'This is the main function'
if __name__ == "__main__":
    
    start_main = time()
    print('\n---main Running...')

    print('\n---Creating directories...')
    create_directory()
    
    print('\n---Generating matrix and sample sizes...')
    n_start, n_end = 50, 1600
    algo = [2, 3]
    num_trials = 10
    n_list, c_list = size_and_samples(n_start=n_start, n_end=n_end, do_print=True)
    
    print('\n---Plotting singular values...')
    plot_singular_vals(n_list, k_factor=1, savefig=True)

    for alg in algo:
        class_1_true, class_1_rel = [], []
        class_2_true, class_2_rel = [], []
        class_3_true, class_3_rel = [], []

        for trial in range(num_trials):
            print(f'\n\n---------- TRIAL {trial+1} ----------')

            print(f'\n---Generating random trial matrices for all classes using Algo {alg}...')
            class1_matrices_list, class2_matrices_list, class3_matrices_list = generate_trial_matrices(n_list, do_print=False, alg=3)
            
            if trial == 0:  # Save these plots only for the first trial
                print('\n---Generating and Plotting PMF for all classes...')
                pmf_1, pmf_2, pmf_3 = generate_pmf_for_class(class1_matrices_list, class_num=1, saveplot=True, alg=alg), generate_pmf_for_class(class2_matrices_list, class_num=2, saveplot=True, alg=alg), generate_pmf_for_class(class3_matrices_list, class_num=3, saveplot=True, alg=alg)

                print('\n---Rejection Sampling for all classes...')
                indices_1, indices_2, indices_3 = rejection_sampling_all(pmf_1, c_list, class_num=1, saveplot=True, alg=alg), rejection_sampling_all(pmf_2, c_list, class_num=2, saveplot=True, alg=alg), rejection_sampling_all(pmf_3, c_list, class_num=3, saveplot=True, alg=alg)
                print(f'No of indices for class 1: {[(len(indices_1[i][0]), len(indices_1[i][1]), len(indices_1[i][2]) )for i in range(len(indices_1))]}')
            else:
                print('\n---Generating and Plotting PMF for all classes...')
                pmf_1, pmf_2, pmf_3 = generate_pmf_for_class(class1_matrices_list, class_num=1, saveplot=False), generate_pmf_for_class(class2_matrices_list, class_num=2, saveplot=False), generate_pmf_for_class(class3_matrices_list, class_num=3, saveplot=False)

                print('\n---Rejection Sampling for all classes...')
                indices_1, indices_2, indices_3 = rejection_sampling_all(pmf_1, c_list, class_num=1, saveplot=False), rejection_sampling_all(pmf_2, c_list, class_num=2, saveplot=False), rejection_sampling_all(pmf_3, c_list, class_num=3, saveplot=False)

            print('\n---Matrix Multiplication using Algorithm 1 for all classes...')
            true_error_1, rel_error_1 = algo_1_caller(class1_matrices_list, indices_1, pmf_1)
            true_error_2, rel_error_2 = algo_1_caller(class2_matrices_list, indices_2, pmf_2)
            true_error_3, rel_error_3 = algo_1_caller(class3_matrices_list, indices_3, pmf_3)

            class_1_true.append(true_error_1)
            class_1_rel.append(rel_error_1)

            class_2_true.append(true_error_2)
            class_2_rel.append(rel_error_2)

            class_3_true.append(true_error_3)
            class_3_rel.append(rel_error_3) 

            print()

        # Find the average of the true and relative errors
        print('\n---Finding the average of the true and relative errors...')
        class_1_true = np.mean(np.array(class_1_true), axis=0)
        class_1_rel = np.mean(np.array(class_1_rel), axis=0)

        class_2_true = np.mean(np.array(class_2_true), axis=0)
        class_2_rel = np.mean(np.array(class_2_rel), axis=0)

        class_3_true = np.mean(np.array(class_3_true), axis=0)
        class_3_rel = np.mean(np.array(class_3_rel), axis=0)

        print('\n---Plotting the trial averaged errors for all classes...')
        plot_average_errors([class_1_rel.T, class_2_rel.T, class_3_rel.T], n_list, trials=num_trials, error_name='relative', saveplot=True, alg=alg)
        plot_average_errors([class_1_true.T, class_2_true.T, class_3_true.T], n_list, trials=num_trials, error_name='true', saveplot=True, alg=alg)

        print(f'\n---End for algo {alg}...')

    
    print('\n---main Done!\n')

    print('- - '*15)
    print(f'\tTime taken for main function: {((time() - start_main) / 60):.1f} minutes')
    print('- - '*15)
