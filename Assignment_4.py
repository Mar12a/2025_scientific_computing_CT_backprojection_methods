import sys
import numpy as np 
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from skimage.transform import radon, iradon, resize
from skimage.data import shepp_logan_phantom
import time

#################################################################################################################
# To use the apply_circular_mask function you can either change the path below after downloading "assignment_2_2_.py"
# or copy paste the code of apply_circular_mask into this file then remove the following two lines

sys.path.append('/Users/marabrandsen/Desktop/SCI Code/Part 2') 
# for windows: sys.path.append(r'c:\Users\Hugo\Documents\Blok 4\Inleiding Scientific Computing\Part 2')
from Assignment_2_2 import apply_circular_mask

#################################################################################################################

'''
Lecture 4
Assignment 1

Implement filtered back projection for various filters and test its performance on noisy data. In order to do this, 
we apply the following steps: 

         (u / f4) ────────── (1) Radon ──────────────→ (f1)
            │                                            │
            │                                            ↓
     (4) inverse Fourier                            (2) Fourier
            ↑                                            │
           (f3) ◄──────── (3) apply filter ──────────── (f2)

This code includes: 
- function to compute all variables shown in the diagram
- 3 functions to define different filters used in FBP
- function to evaluate performance for assignment 5
- plotting code for FBP and iRadon under varying noise levels
- plotting code to visualize the shape of each filter

'''

#################################################################################################################
# Perform Filtered Back Projection using a specified filter
#################################################################################################################
def FBP(image, filter_func, n_angles, noise):
    theta_deg = np.linspace(0., 180., n_angles, endpoint=False)  
    theta_rad = np.deg2rad(theta_deg)

    # Generate sinogram from Radon transform and add Gaussian noise
    f = radon(image, theta_deg)
    f = f + noise * np.random.randn(*f.shape) 
    ns = f.shape[0]

    # (1) Apply 1D FFT across detector axis (rows)
    f1 = fft(f, axis=0)

    # (2) Multiply FFT with filter in frequency domain
    freqs = np.fft.fftfreq(ns, d=1)
    norm_freqs = freqs / np.max(np.abs(freqs))  # now in [-1, 1]
    filt = filter_func(norm_freqs)
    f2 = f1 * filt[:, np.newaxis]

    # (3) Apply inverse FFT to return to spatial sinogram
    f3 = np.real(ifft(f2, axis=0))

    # (4) Backproject each filtered projection
    nx, ny = image.shape
    x = np.linspace(-nx/2, nx/2, nx)
    y = np.linspace(-ny/2, ny/2, ny)
    X, Y = np.meshgrid(x, y)

    f4 = np.zeros_like(X)
    s_grid =  np.linspace(-nx/2, nx/2, ns)

    for i, th in enumerate(theta_rad):
        # Compute projection line coordinates
        s = X * np.cos(th) + Y * np.sin(th)
        projection = f3[:, i]
        # Interpolate the projection data over spatial grid
        values = np.interp(s, s_grid, projection, left=0, right=0)
        f4 += values

    return apply_circular_mask(f4)

#################################################################################################################
# Filter definitions used in FBP
#################################################################################################################
def ramp(sigma):
    return np.abs(sigma)

def SL(sigma):
    val = (2/np.pi) * np.sin(sigma * np.pi / 2)
    return np.abs(val)

def cos_filter(sigma):
    val = sigma * np.cos(sigma * np.pi / 2)
    return np.abs(val)

#################################################################################################################
# Assignment 5 required function to run FBP and return reconstruction, errors, runtime
#################################################################################################################
def run_fbp_reconstruction(image, filter_func, n_angles, noise, n):
    start = time.time()

    # Resize and mask input
    image = resize(image, (n, n), mode='reflect', anti_aliasing=True)

    # Run FBP
    uk = FBP(image, filter_func, n_angles, noise)

    # Rescale to match input range
    uk = uk * (np.max(image) / (np.max(uk)))
    uk = apply_circular_mask(uk)

    # Compute error using masked inputs
    recon_error = np.linalg.norm(uk - image) / (np.linalg.norm(image))
    runtime = time.time() - start

    return uk, recon_error, runtime



#################################################################################################################
# Main code block for visualization
#################################################################################################################
if __name__ == "__main__":          

    #################################################################################################################
    # Plotting FBP reconstructions and iradon under varying noise
    #################################################################################################################
    na = 400
    shepp = shepp_logan_phantom()
    sigma_list = [0.005, 0.5, 5, 10, 25, 50, 500, 1000]

    filter_list = [ramp, SL, cos_filter]
    filter_names = ["Ramp", "Shepp-Logan", "Cosine"]

    n_rows = len(filter_list) + 1  # Add one row for iRadon comparison
    n_cols = len(sigma_list)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))

    for row_idx, (filter_func, filter_name) in enumerate(zip(filter_list, filter_names)):
        for col_idx, sigma in enumerate(sigma_list):
            # Reconstruct and plot
            reconstruction = FBP(shepp, filter_func, na, sigma)
            axs[row_idx, col_idx].imshow(apply_circular_mask(reconstruction), cmap='gray', origin='lower', vmin=0, vmax=1)
            axs[row_idx, col_idx].axis('off')
            if row_idx == 0:
                axs[row_idx, col_idx].set_title(f"σ={sigma}")

    for col_idx, sigma in enumerate(sigma_list):
        # iRadon computation and plotting 
        theta_deg = np.linspace(0., 180., na, endpoint=False)
        sinogram = radon(shepp, theta_deg)
        sinogram = sinogram + sigma * np.random.randn(*sinogram.shape)
        iradon_recon = iradon(sinogram, theta=theta_deg, circle=True)
        axs[-1, col_idx].imshow(apply_circular_mask(iradon_recon), cmap='gray', aspect='equal', vmin=0, vmax=1)
        axs[-1, col_idx].axis('off')

    plt.suptitle("FBP vs iRadon reconstruction with varying noise levels", fontsize=16)
    plt.tight_layout()
    plt.show()

    #################################################################################################################
    # Plotting the filter response functions
    #################################################################################################################
    sigma_vals = np.linspace(0, 1, 200)
    plt.plot(sigma_vals, ramp(sigma_vals), label="Ramp")
    plt.plot(sigma_vals, SL(sigma_vals), label="Shepp-Logan")
    plt.plot(sigma_vals, cos_filter(sigma_vals), label="Cosine")

    plt.xlabel('σ (frequency)')
    plt.ylabel('Filter Response')
    plt.title('Frequency Response of FBP Filters')

    plt.legend()
    plt.grid(True)
    plt.show()
