import numpy as np 
import sys, time, math, itertools
import matplotlib.pyplot as plt
from scipy.sparse.linalg import norm as sparse_norm
from skimage.transform import radon, iradon, resize
from skimage.data import shepp_logan_phantom
from matplotlib.lines import Line2D

#################################################################################################################
# To use the necessary functions from other files you can either change the path below 
# after downloading "Assignment_2_2.py", "Assignment3_2.py" and "Assignment_4.py"
# or copy paste the code of the functions into this file then remove the following four lines

sys.path.append('/Users/marabrandsen/Desktop/SCI Code/Part 2')
from Assignment_2_2 import apply_circular_mask, fourier_reconstruct
from Assignment_3_2 import RadonSparseMatrix
from Assignment_4 import run_fbp_reconstruction, ramp, SL, cos_filter

#################################################################################################################

'''
Lecture 5
Assignment 1

Compare the iterative methods: Richardson, Kaczmarz with the previously defined reconstruction methods  
in terms of computational efficiency and reconstruction quality.  

This code includes:  
------- Functions to apply different methods -------
- function to run the Richardson iteration and output the result, relative reconstruction error and the runtime
- function to run the Kaczmarz iteration and output the result, relative reconstruction error and the runtime
- function to run the iradon reconstruction and output the result, relative reconstruction error and the runtime

------- Functions to visualise and compare these methods -------
- function to plot the runtime versus the relative error for various methods 
- function to visualise the result of all different methods
- function to compare various methods and their sensitivity to na = number of angles
- function to plot the relative error versus the number of iterations 

------- Remaining code -------
- code which defines variables and runs the above functions to output the stated graphs 

'''


#################################################################################################################
# Richardson Iteration Method
#################################################################################################################
def richard(image, k, a, K, u0):
    start = time.time()

    u = image.flatten()
    f = K @ u 
    Kt = K.transpose()

    uk = u0 
    recon_error_list = []

    for i in range(k): 
        uk = uk - a * (Kt @ (K @ uk - f))
        recon_error = np.linalg.norm(uk - u)/np.linalg.norm(u)
        recon_error_list.append(recon_error)

    end = time.time()
    runtime = end - start
    return apply_circular_mask(uk.reshape(n, n)), recon_error_list, runtime

#################################################################################################################
# Kaczmarz Iteration Method
#################################################################################################################
def kaczmarz(image, passes, u0, K):
    start = time.time()

    u = image.flatten()
    f = K @ u 

    uk = u0
    recon_error_list = []

    m = K.shape[0]  # number of rows in sparse matrix
    k = passes * m

    for i in range(k):
        step = m // k if m > k else 1
        idx = (i * step) % m
        ki = K.getrow(idx)
        kit2 = ki.multiply(ki).sum()
        residual = f[idx] - ki @ uk
        uk = uk + (ki.transpose() @ (residual / kit2)).ravel()

        temp_uk = apply_circular_mask(uk.reshape(n, n))
        temp_uk = temp_uk.ravel()
        recon_error = np.linalg.norm(temp_uk - u) / np.linalg.norm(u)
        recon_error_list.append(recon_error)

    end = time.time()
    runtime = end - start

    return apply_circular_mask(uk.reshape(n, n)), recon_error_list, runtime

#################################################################################################################
# iradon Reconstruction Method
#################################################################################################################
def iradon_reconstruct(na, image, sigma):
    theta_deg = np.linspace(0., 180., na, endpoint=False)
    sinogram = radon(image, theta_deg)
    noisy_sinogram = sinogram + sigma * np.random.randn(*sinogram.shape)

    start = time.time()
    uk_iradon = iradon(noisy_sinogram, theta=theta_deg, circle=True)
    runtime = time.time() - start

    recon_error = np.linalg.norm(uk_iradon - image) / np.linalg.norm(image)

    return apply_circular_mask(uk_iradon), recon_error, runtime

#################################################################################################################
# Plot Runtime vs Reconstruction Error for Each Method
#################################################################################################################
def runtime_vs_accu(errors, names, runtimes):
    print("Plotting... runtime_vs_accu")
    plt.figure(figsize=(8, 6))

    # Use distinct colors and markers
    cmap = plt.cm.get_cmap('tab10', len(names))
    markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X', 'h', '<']

    for i, name in enumerate(names):
        plt.scatter(runtimes[i], errors[i], 
                    color=cmap(i), 
                    marker=markers[i % len(markers)], 
                    s=80, label=name)

    plt.xlabel("Runtime (seconds)")
    plt.ylabel("Reconstruction Error")
    plt.title("Runtime vs. Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return

#################################################################################################################
# Compare Original Image to Reconstructions
#################################################################################################################
def compare_images(u, uk_list, names):
    print("Plotting... compare_images")

    num_images = 1 + len(uk_list)
    ncols = 4
    nrows = math.ceil(num_images / ncols)

    plt.figure(figsize=(4 * ncols, 4 * nrows))

    # Original
    plt.subplot(nrows, ncols, 1)
    plt.imshow(u, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    # Reconstructions
    for i, (uk, name) in enumerate(zip(uk_list, names)):
        if uk.ndim == 1:
            uk = uk.reshape(u.shape)

        plt.subplot(nrows, ncols, i + 2)
        if "Filtered Back Projection" in name:
            plt.imshow(uk, cmap='gray', origin='lower',vmin=0, vmax = 1)
        else:
            plt.imshow(uk, cmap='gray', vmin=0, vmax = 1)
        plt.title(name)
        plt.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.show()
    return

#################################################################################################################
# Sweep over na values and Compare Methods (custom subplot layout 2x4)
#################################################################################################################
def sweep_na_and_plot(na_list, sigma, n, k, shepp):
    print('Plotting... sweep_na_and_plot')

    # Define methods in custom order matching the layout
    method_order = [
        "iradon Reconstruction",
        "Fourier Reconstruction",
        "Filtered Back Projection (Shepp-Logan)",
        "Richardson Iteration",
        "Legend",
        "Filtered Back Projection (Ramp)",
        "Filtered Back Projection (Cosine)",
        "Kaczmarz Iteration"
    ]

    # All actual methods (no 'Legend')
    method_names = [
        "iradon Reconstruction",
        "Filtered Back Projection (Ramp)",
        "Filtered Back Projection (Shepp-Logan)",
        "Filtered Back Projection (Cosine)",
        "Fourier Reconstruction",
        "Richardson Iteration",
        "Kaczmarz Iteration"
    ]

    all_runtimes = {name: [] for name in method_names}
    all_errors = {name: [] for name in method_names}

    for na in na_list:
        print(f'Calculating... n = {na}')
        theta_deg = np.linspace(0., 180., na, endpoint=False)
        s = np.linspace(-1, 1, n)
        K = RadonSparseMatrix(n, theta_deg, s, format='csr')
        u0 = np.zeros(K.shape[1])

        # Estimate step size for Richardson
        K1_norm = sparse_norm(K, 1)
        Kinf_norm = sparse_norm(K, np.inf)
        lambda_max_estimate = K1_norm * Kinf_norm
        a = 1 / lambda_max_estimate

        # iradon
        uk_iradon, recon_err_iradon, runtime_iradon = iradon_reconstruct(na, shepp, sigma)
        all_runtimes["iradon Reconstruction"].append(runtime_iradon)
        all_errors["iradon Reconstruction"].append(recon_err_iradon)

        # FBP
        uk_ramp, err_ramp, t_ramp = run_fbp_reconstruction(shepp, ramp, na, sigma, n)
        uk_sl, err_sl, t_sl = run_fbp_reconstruction(shepp, SL, na, sigma, n)
        uk_cos, err_cos, t_cos = run_fbp_reconstruction(shepp, cos_filter, na, sigma, n)
        uk_fourier, err_fourier, t_fourier = fourier_reconstruct(shepp, na, sigma)

        all_runtimes["Filtered Back Projection (Ramp)"].append(t_ramp)
        all_errors["Filtered Back Projection (Ramp)"].append(err_ramp)
        all_runtimes["Filtered Back Projection (Shepp-Logan)"].append(t_sl)
        all_errors["Filtered Back Projection (Shepp-Logan)"].append(err_sl)
        all_runtimes["Filtered Back Projection (Cosine)"].append(t_cos)
        all_errors["Filtered Back Projection (Cosine)"].append(err_cos)
        all_runtimes["Fourier Reconstruction"].append(t_fourier)
        all_errors["Fourier Reconstruction"].append(err_fourier)

        # Richardson
        richard_uk, richard_errors, t_richard = richard(shepp, k, a, K, u0)
        all_runtimes["Richardson Iteration"].append(t_richard)
        all_errors["Richardson Iteration"].append(richard_errors[-1])

        # Kaczmarz
        kacz_uk, kacz_errors, t_kacz = kaczmarz(shepp, k, u0, K)
        all_runtimes["Kaczmarz Iteration"].append(t_kacz)
        all_errors["Kaczmarz Iteration"].append(kacz_errors[-1])

    # Plotting
    fig, axs = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
    axs = axs.flatten()
    colors = plt.cm.viridis(np.linspace(0, 1, len(na_list)))

    for idx, name in enumerate(method_order):
        ax = axs[idx]
        if name == "Legend":
            # Create custom legend
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label=f'na = {na}',
                       markerfacecolor=colors[j], markersize=10)
                for j, na in enumerate(na_list)
            ]
            ax.legend(handles=legend_elements, loc='center', title='na values')
            ax.axis('off')
        else:
            ax.plot(all_runtimes[name], all_errors[name], linestyle=':', color='gray')
            for j, na in enumerate(na_list):
                ax.scatter(all_runtimes[name][j], all_errors[name][j], color=colors[j], s=80)
            ax.set_title(name, fontsize=12)

            # Bottom row gets x-axis
            if idx >= 4:
                ax.set_xlabel("Runtime (s)")
            else:
                ax.set_xticklabels([])

            # Left column gets y-axis
            if idx % 4 == 0:
                ax.set_ylabel("Reconstruction Error")
            else:
                ax.set_yticklabels([])

            ax.grid(True)

    fig.suptitle("Reconstruction Error vs Runtime based on number of angles", fontsize=16)
    plt.show()

#################################################################################################################
# Plot Reconstruction Error over Iterations for Iterative vs Non-Iterative Methods
#################################################################################################################
def plot_recon_error_vs_iterations(iterative_methods, non_iterative_methods, k):
    print("Plotting... plot_recon_error_vs_iterations")

    plt.figure(figsize=(10, 7))

    # Iterative methods: line plot
    for method_name, error_list in iterative_methods.items():
        plt.plot(range(1, len(error_list) + 1), error_list, label=method_name, linewidth=2)

    # Non-iterative methods: horizontal lines
    color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    for method_name, error_value in non_iterative_methods.items():
        color = next(color_cycle)
        plt.hlines(error_value, 1, k, linestyles='--', colors=color, linewidth=2, label=method_name)

    plt.xlabel("Iterations")
    plt.ylabel("Reconstruction Error")
    plt.title("Reconstruction Error vs Iterations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#################################################################################################################
# Main Execution Block
#################################################################################################################

#############################
# Define Parameters
na = 5                                                                      # Number of angles used in reconstruction
sigma = 0                                                                   # Noise level
passes = 10                                                                 # Number of passes over the data
na_list = [5, 10, 15, 20, 25, 50, 100, 150, 200, 300, 400]                  # For sweeping comparisons
n = 64                                                                      # Resolution to which phantom is resized (64x64)
N = 10                                                                      # Unused in current code, possibly reserved
s = np.linspace(-1, 1, n)                                                   # Detector positions

#############################
# Prepare Shepp–Logan phantom and sinogram

theta_deg = np.linspace(0., 180., na, endpoint=False)  
theta_rad = np.deg2rad(theta_deg)

shepp_orig = shepp_logan_phantom()                                          # Original high-res phantom
shepp = resize(shepp_orig, (n, n), mode='reflect', anti_aliasing=True)      # Resize to working resolution

#############################
# Generate Radon Sparse Matrix for iterative methods

K = RadonSparseMatrix(n, theta_deg, s, format='csr')
u0 = np.zeros(K.shape[1])                                                   # Initial guess for iterative methods (all zeros)

# Estimate step size for Richardson iteration
K1_norm = sparse_norm(K, 1)
Kinf_norm = sparse_norm(K, np.inf)
lambda_max_estimate = K1_norm * Kinf_norm
a = 1 / lambda_max_estimate                                                 # Step size α

#############################
# Run All Reconstruction Methods

print("Calculating... richard")
richard_uk, richard_recon_error, richard_runtime = richard(shepp, passes, a, K, u0)

print("Calculating... kazc")
kacz_uk, kacz_recon_error, kacz_runtime = kaczmarz(shepp, passes, u0, K)

print("Calculating... iradon")
uk_iradon, recon_err_iradon, runtime_iradon = iradon_reconstruct(na, shepp, sigma)

print("Calculating... ramp")
uk_ramp, recon_err_ramp, runtime_ramp = run_fbp_reconstruction(shepp, ramp, na, sigma, n)

print("Calculating... sl")
uk_sl, recon_err_sl, runtime_sl = run_fbp_reconstruction(shepp, SL, na, sigma, n)

print("Calculating... cos")
uk_cos, recon_err_cos, runtime_cos = run_fbp_reconstruction(shepp, cos_filter, na, sigma, n)

print("Calculating... fourier")
uk_fourier, recon_err_fourier, runtime_fourier = fourier_reconstruct(shepp, na, sigma)

#############################
# Collect Results for Plotting

name_list               = ["iradon Reconstruction", "Filtered Back Projection (Ramp)",  "Filtered Back Projection (Shepp-Logan)",   "Filtered Back Projection (Cosine)",    "Fourier Reconstruction",   "Richardson Iteration",     "Kaczmarz Iteration"]
recon_error_list        = [recon_err_iradon,        recon_err_ramp,                     recon_err_sl,                               recon_err_cos,                          recon_err_fourier,          richard_recon_error[-1],    kacz_recon_error[-1]]
runtime_list            = [runtime_iradon,          runtime_ramp,                       runtime_sl,                                 runtime_cos,                            runtime_fourier,            richard_runtime,            kacz_runtime]
uk_list                 = [uk_iradon,               uk_ramp,                            uk_sl,                                      uk_cos,                                 uk_fourier,                 richard_uk,                 kacz_uk]

iterative_methods       = {"Richardson Iteration": richard_recon_error,     "Kaczmarz Iteration": kacz_recon_error}
non_iterative_methods   = {"iradon Reconstruction": recon_err_iradon,       "FBP (Ramp)": recon_err_ramp,               "FBP (Shepp-Logan)": recon_err_sl,      "FBP (Cosine)": recon_err_cos,      "Fourier Reconstruction": recon_err_fourier}

#############################
# Plot All Results

runtime_vs_accu(recon_error_list, name_list, runtime_list)                          # Plot Runtime vs Accuracy
compare_images(shepp, uk_list, name_list)                                           # Show image reconstructions
sweep_na_and_plot(na_list, sigma, n, passes, shepp)                                 # Sweep over na and compare
plot_recon_error_vs_iterations(iterative_methods, non_iterative_methods, passes)    # Plot error vs iterations