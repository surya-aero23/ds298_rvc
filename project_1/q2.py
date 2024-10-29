import os
import numpy as np
import matplotlib.pyplot as plt


# Create a directory to save images
if not os.path.exists('Images'):
    os.mkdir('Images')
    os.mkdir('Images/Q2')
elif not os.path.exists('Images/Q2'):
    os.makedirs('Images/Q2')


def plot_histograms(data, xlabel=None, ylabel=None, title=' ', bins=100, close=True, label=None):
    """
    Plot histogram of the data
    """
    plt.hist(data, label=label, bins=bins, density=True, alpha=0.5, color='r', edgecolor='black')
    if close:
        plt.title(title)
        plt.savefig(f'Images/Q2/{title}.png')
        plt.close()


def expected_income(Y, Z):
    return np.mean(Z[Y > 0])


def pdf_price(x_vals, a, b):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((-b * np.log(x_vals / a) - 3)**2)) * np.abs(-3 / x_vals)


# Sample size
n = 10 ** 5

# Yield Properties
mu = 3
sigma = 1

# Transformation Properties
a, b = 5000, 3

# Sampling Yield Values
Y = np.random.normal(mu, sigma, n)

# Calculating local prize using transformation
X = a * np.exp(-Y / b)

# Combination for notional income
Z = X * Y

plot_histograms(X, title='Density vs Local Prize')
plot_histograms(Y, title='Density vs Yield')
plot_histograms(Z, title='Density vs Notional Income')

print(f"Expected Income for valid Y: {expected_income(Y, Z):.6f}")

# Plot the pdf of the local prize vs pdf of X
x_vals = np.linspace(0.001, 10000, 1000)
pdf_x = pdf_price(x_vals, a, b)
plt.plot(x_vals, pdf_x, label='PDF of X')
plot_histograms(X, close=False, label='Histogram of X')
plt.title('PDF of X')
plt.legend()
plt.savefig('Images/Q2/pdf_of_X.png')
plt.close()