import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time


def get_eigen(n, class_type):

    Eigen = np.zeros((n,n))

    # Class I
    if class_type == 1:
        for k in range(n):
            Eigen[k,k] = 2 * np.exp(-10 * (k-1) / n)

    # Class II
    elif class_type == 2:
        for k in range(n):
            Eigen[k,k] = 1 + (k/n)

    return Eigen

# Function to create test matrix A using Algorithm 2

def generate_test_matrix_algo1(Eigen):
    n = Eigen.shape[0]
    
    # Initialize random matrices
    M = np.random.rand(n, n)
    
    # Generate singular vectors using QR factorization
    Q, R = np.linalg.qr(M)
    
    # Build test matrix
    A = np.matmul(np.matmul(Q, Eigen), Q.T)
    
    return A


def get_pk(A, type):
    n = A.shape[0]
    total = 0
    pk = np.zeros(n)

    if type == 'row':
        # Calculate the sum for the denominator of pk
        for k in range(n):
            total += (np.linalg.norm(A[k,:], ord=2) ** 2)

        # Calculate pk
        for k in range(n):
            pk[k] = (np.linalg.norm(A[k,:], ord=2) ** 2) / total

    elif type == 'column':
        # Calculate the sum for the denominator of pk
        for k in range(n):
            total += (np.linalg.norm(A[:,k], ord=2) ** 2)

        # Calculate pk
        for k in range(n):
            pk[k] = (np.linalg.norm(A[:,k], ord=2) ** 2) / total

    return pk

# Function to perform randomized linear solve using Kaczmarz algorithm

def randomized_kaczmarz(A, b, x0, x):
    n = A.shape[0]
    xk = x0
    pk = get_pk(A, 'row')
    max_pk = np.max(pk)
    iter_count = 0
    while iter_count < 10 * n:
        u = np.random.uniform(0, 1)
        k = np.random.randint(0, n)

        # acceptance criterion
        if u * max_pk < pk[k]:
            a = A[k,:]
            xk = xk - (a @ xk - b[k]) * a.T / (np.linalg.norm(a, ord=2) ** 2)

            # Calculate the residual
            residual = np.linalg.norm(x - xk, ord=2) / np.linalg.norm(x, ord=2)
            if residual < 0.1:
                break

        iter_count += 1

    return iter_count


def cd_lsq(A, b, x0, x):
    n = A.shape[0]
    xk = x0
    pk = get_pk(A, 'column')
    max_pk = np.max(pk)
    iter_count = 0
    ax_temp = np.zeros(n)
    alpha = 0
    while iter_count < 10 * n:
        u = np.random.uniform(0, 1)
        k = np.random.randint(0, n)

        # acceptance criterion
        if u * max_pk < pk[k]:
            a = A[:,k]
            # Canonical vector
            # e = np.zeros(n)
            # e[k] = 1
            alpha = (a.T @ (ax_temp - b)) / (np.linalg.norm(a, ord=2) ** 2)
            xk[k] = xk[k] - alpha 
            ax_temp -= alpha * A[:,k]

            # Calculate the residual
            residual = np.linalg.norm(x - xk, ord=2) / np.linalg.norm(x, ord=2)
            if residual < 0.1:
                break

        iter_count += 1

    return iter_count


def cd_spd(A, b, x0, x):
    n = A.shape[0]
    xk = x0
    pk = get_pk(A, 'row')
    max_pk = np.max(pk)
    iter_count = 0
    while iter_count < 10 * n:
        u = np.random.uniform(0, 1)
        k = np.random.randint(0, n)

        # acceptance criterion
        if u * max_pk < pk[k]:
            a = A[k,:]
            # Canonical vector
            e = np.zeros(n)
            e[k] = 1
            xk = xk - (np.dot(a, xk) - b[k]) * e / (np.linalg.norm(a, ord=2) ** 2)

            # Calculate the residual
            residual = np.linalg.norm(x - xk, ord=2) / np.linalg.norm(x, ord=2)
            if residual < 0.1:
                break

        iter_count += 1

    return iter_count

# Initializing the parameters

N = [100, 200, 400, 800, 1600]
category_type = [1, 2]
algos = ['Kaczmarz', 'CD-LSQ', 'CD-SPD']

num_trails = 20        # Number of trails for averaging

categories = ['Category 1', 'Category 2']
markers = {0: 'x', 1: 'o', 2: '*'}

mean_wall_time = np.zeros((2, 3, len(N)))


average_iterations = np.zeros((2, 3, len(N)))
computational_effort = np.zeros((2, 3, len(N)))


for i in range(len(category_type)):
    ct = category_type[i]
    time_class = []
    for j, a in enumerate(algos):
        times = []
        for n in N: 
            x0 = np.zeros(n)
            total_time = 0  # Initialize total time
            for _ in range(num_trails):   # Performing trails
                print(f"--- Category: {ct}, Algorithm: {a}, N: {n}, trial: {_} ---")
                Lambda = get_eigen(n, ct)
                A = generate_test_matrix_algo1(Lambda)
                x = np.random.randn(n)
                b = np.dot(A, x)
                start_time = time.time()    # Start time 
                if a == 'Kaczmarz':
                    iter_count = randomized_kaczmarz(A, b, x0, x)
                elif a == 'CD-LSQ':
                    iter_count = cd_lsq(A, b, x0, x)
                elif a == 'CD-SPD':
                    iter_count = cd_spd(A, b, x0, x)
                end_time = time.time()
                total_time += (end_time - start_time)
                average_iterations[i, j, N.index(n)] += iter_count        # Calculate average iterations

            average_time = total_time / num_trails  # Calculate average time
            mean_wall_time[i, j, N.index(n)] = average_time

average_iterations /= num_trails

# Calculate computational effort
for i in range(len(category_type)):
    for j, a in enumerate(algos):
        for k in range(len(N)):
            if a == 'Kaczmarz':
                computational_effort[i, j, k] = average_iterations[i, j, k] * (2 * N[k] + 1)
            elif a == 'CD-LSQ':
                computational_effort[i, j, k] = average_iterations[i, j, k] * (2 * N[k] + 1)
            elif a == 'CD-SPD':
                computational_effort[i, j, k] = average_iterations[i, j, k] * (N[k] + 1)


# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

for i in range(len(category_type)):
    ct = category_type[i]
    for j, a in enumerate(algos):            
        axs[i].plot(N, mean_wall_time[i, j], label=a, marker=markers[j])
    axs[i].set_xlabel('N (Matrix Size)')
    axs[i].set_ylabel('Time (s)')
    # axs[i].set_yscale('log')
    axs[i].set_title(f"{categories[i]}")
    axs[i].legend()

plt.suptitle(f'Average Time taken for different algorithms (Average over {num_trails} trials)')
plt.tight_layout()
plt.savefig('q1_time.png')
plt.close()
# plt.show()

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

for i in range(len(category_type)):
    ct = category_type[i]
    for j, a in enumerate(algos):
        axs[i].plot(N, computational_effort[i, j], label=a, marker=markers[j])
    axs[i].set_xlabel('N (Matrix Size)')
    axs[i].set_ylabel('Computational Effort')
    axs[i].set_yscale('log')
    axs[i].set_title(f"{categories[i]}")
    axs[i].grid()
    axs[i].legend()

plt.suptitle(f'Computational Effort for different algorithms (Average over {num_trails} trials)')
fig.tight_layout()
plt.savefig('q1_compute.png')
# plt.show()
