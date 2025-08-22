import numpy as np
import time
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon, resize
from skimage.data import shepp_logan_phantom


###############################################################
# To use the interpolate function you can either change the path below after downloading "assignment_2_1.py"
# or copy paste the code of interpolate into this file then remove the following three lines

import sys
sys.path.append('/Users/marabrandsen/Desktop/SCI Code/Part 2')
from Assignment_2_1 import interpolate

###############################################################

'''
Lecture 2
Assignment 2

Using 2.1, our interpolation method, to reconstruct the SL phantom and compare it with the iradon method

     (u / fi / uu) ────────── (1) Radon ──────────────→ (f)
            │                                            │
            │                                            ↓
     (4) inverse Fourier                            (2) Fourier
            ↑                                            │
 (fft_interpolated) ◄──── (3) interpolation ───── (fft_back_shifted)

This code includes: 
- function to find all of the above variables and returns these 
- function to compute all variables needed in assignment 5
- function to mask a square image with a circle 
- the code to graph all the different steps of the above diagram 
- the code to graph the original, iradon and our reconstruction 
- the code to graph the image with different sigma values
- the code to graph the image with different na values
- the code to graph the cross-section through vertical center at na = 50

'''

#################################################################################################################
# The function that will reconstruct according to the diagram above 
#################################################################################################################
def reconstruct_from_sinogram(nx, na, image, sigma):
    
    # Setting up the parameters 
    theta_deg = np.linspace(0., 180., na, endpoint=False)                                   # Range of angles in degrees
    theta_rad = np.deg2rad(theta_deg)                                                       # Range of angles in radians
   
    # Fourier grid set-up
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=1))                                           
    ky = np.fft.fftshift(np.fft.fftfreq(nx, d=1))  
    ks = np.fft.fftshift(np.fft.fftfreq(nx, d=1))

    # (1) Radon transform 
    sinogram_nonoise = radon(image, theta=theta_deg)                 
    sinogram = sinogram_nonoise + sigma * np.random.randn(*sinogram_nonoise.shape)          # Adding noise to the sinogram using sigma 

    # (reverse 1) iradon result
    uu = iradon(sinogram, theta=theta_deg, circle=True)

    # (2) Fourier transform
    f_shifted = np.fft.ifftshift(sinogram, axes=0)
    fft_shifted = np.fft.fft(f_shifted, axis=0)
    fft_back_shifted = np.fft.fftshift(fft_shifted, axes=0)

    # (3) Interpolate
    fft_interpolated = interpolate(fft_back_shifted, theta_rad, ks, kx, ky)                 # See Assignment 2.1

    # (4) Inverse Fourier
    fi_shifted = np.fft.ifftshift(fft_interpolated, axes=(0, 1))
    iffti_shifted = np.fft.ifft2(fi_shifted)
    fi_display = (np.fft.fftshift(iffti_shifted)).real

    # Fixing the orientation of the result to match with imshow
    f4 = apply_circular_mask(np.flipud(fi_display.T))

    return sinogram, f4, uu, fft_back_shifted, fft_interpolated

#################################################################################################################
# The function that retrieves the data needed in assignment 5 
#################################################################################################################
def fourier_reconstruct(image, na, sigma):
    nx = image.shape[0]
    
    start_time = time.time()
    uk_iradon = reconstruct_from_sinogram(nx, na, image, sigma)
    recon_error = np.linalg.norm(uk_iradon - image) / np.linalg.norm(image)
    runtime = time.time() - start_time

    return uk_iradon, recon_error, runtime

#################################################################################################################
#  The function that can mask the data with a circle
#################################################################################################################
def apply_circular_mask(image):
    h, w = image.shape
    Y, X = np.ogrid[:h, :w]
    center = (h // 2, w // 2)
    radius = min(center)
    mask = (X - center[1])**2 + (Y - center[0])**2 <= radius**2
    masked_image = np.zeros_like(image)
    masked_image[mask] = image[mask]
    return masked_image


if __name__ == "__main__":      
    
    #################################################################################################################
    # Setting up the main variables and parameters
    #################################################################################################################
    
    # Setting the variables 
    nx = 400                                                                                # Number of spatial steps
    na = 400                                                                                # Number of angular steps 

    # Phantom and sinogram
    u = shepp_logan_phantom()   

    # Applying function 1
    f, f4, uu, fft_back_shifted, fft_interpolated = reconstruct_from_sinogram(nx, na, u, sigma=0)

    # Computing the log magnitude spectrum of our data
    fft_mag = np.log1p(np.abs(fft_back_shifted))
    fft_interp_mag = np.log1p(np.abs(fft_interpolated))

    #################################################################################################################
    # Plotting the results  
    #################################################################################################################

    ###############################################################
    # Image 1 showing the steps as displayed in the diagram above
    fig1, axs1 = plt.subplots(2, 2, figsize=(12, 10))

    # Top row
    axs1[0, 0].imshow(u, cmap='gray')
    axs1[0, 0].set_title('Original SLP')
    axs1[0, 0].axis('off')

    axs1[0, 1].imshow(f, cmap='gray', aspect='equal')
    axs1[0, 1].set_title('Radon Transform')
    axs1[0, 1].axis('off')

    # Bottom row
    axs1[1, 0].imshow(fft_interp_mag, cmap='gray', aspect='equal')
    axs1[1, 0].set_title('Interpolated FFT')
    axs1[1, 0].axis('off')

    axs1[1, 1].imshow(fft_mag, cmap='gray', aspect='equal')
    axs1[1, 1].set_title('FFT of Sinogram')
    axs1[1, 1].axis('off')

    fig1.suptitle("Reconstruction Steps: Radon and Frequency Domain", fontsize=16, weight='bold')
    plt.tight_layout()
    plt.show()


    ###############################################################
    # Image 2 showing the three different images so comparsion is easier

    fig2, axs2 = plt.subplots(1, 3, figsize=(15, 5))

    axs2[0].imshow(u, cmap='gray')
    axs2[0].set_title('Original SLP')
    axs2[0].axis('off')

    axs2[1].imshow(f4, vmin=0, vmax=1, cmap='gray', aspect='equal', origin='lower')
    axs2[1].set_title('Fourier Reconstruction')
    axs2[1].axis('off')

    axs2[2].imshow(uu, cmap='gray')
    axs2[2].set_title('iRadon Reconstruction')
    axs2[2].axis('off')

    fig2.suptitle("Original vs Fourier vs iRadon", fontsize=16, weight='bold')
    plt.subplots_adjust(top=0.85, bottom=0.1, hspace=0.3)
    plt.show()

    
    ###############################################################
    # Image 3: Comparing the effect of sigma on iradon and Fourier reconstruction

    sigma_list = [0.005, 0.5, 5, 10, 25, 50, 500, 1000]
    fig, axs = plt.subplots(2, len(sigma_list), figsize=(3 * len(sigma_list), 6))

    for i, sigma in enumerate(sigma_list):
        f, f4, uu, fft_back_shifted, fft_interpolated = reconstruct_from_sinogram(nx, na, u, sigma)

        axs[0, i].imshow(f4, vmin=0, vmax=1, cmap='gray', aspect='equal', origin='lower')
        axs[0, i].set_title(f"σ={sigma}")
        axs[0, i].axis('off')

        axs[1, i].imshow(apply_circular_mask(uu), cmap='gray', vmin=0, vmax=1)
        axs[1, i].axis('off')

    axs[0, 0].text(-0.3, 0.5, "Fourier reconstruction", fontsize=12,
                   va='center', ha='right', rotation=90, transform=axs[0, 0].transAxes)

    axs[1, 0].text(-0.3, 0.5, "iRadon", fontsize=12,
                   va='center', ha='right', rotation=90, transform=axs[1, 0].transAxes)

    fig.suptitle("Effect of Noise Level (σ) on Reconstruction Quality", fontsize=16, weight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, left=0.15)
    plt.show()


    ###############################################################
    # Image 4: Comparing the effect of na (number of angles) on reconstruction

    na_list = [5, 10, 20, 30, 40, 50, 100, 500, 1000]
    fig, axs = plt.subplots(2, len(na_list), figsize=(3 * len(na_list), 6))

    for i, na in enumerate(na_list):
        f, f4, uu, fft_back_shifted, fft_interpolated = reconstruct_from_sinogram(nx, na, u, sigma=0)

        axs[0, i].imshow(f4, vmin=0, vmax=1, cmap='gray', origin='lower')
        axs[0, i].set_title(f"na={na}")
        axs[0, i].axis('off')

        axs[1, i].imshow(apply_circular_mask(uu), cmap='gray', vmin=0, vmax=1)
        axs[1, i].axis('off')

    axs[0, 0].text(-0.3, 0.5, "Fourier reconstruction", fontsize=12,
                   va='center', ha='right', rotation=90, transform=axs[0, 0].transAxes)

    axs[1, 0].text(-0.3, 0.5, "iRadon", fontsize=12,
                   va='center', ha='right', rotation=90, transform=axs[1, 0].transAxes)

    fig.suptitle("Effect of Angular Resolution (na) on Reconstruction Quality", fontsize=16, weight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, left=0.15)
    plt.show()

    ###############################################################
    # Image 5: vertical cross-section 
    na_cross = 10                                   # Number of angles to use
    x_cross = 200                                   # Column index to extract cross-section 

    # Reconstruct with chosen na
    f, f4, uu, _, _ = reconstruct_from_sinogram(nx, na_cross, u, sigma=0)

    # Clamp x index if needed
    x_cross = min(max(x_cross, 0), u.shape[1] - 1)

    # Extract vertical slice (same x across all)
    u_col = u[:, x_cross]
    f4_col = f4[:, x_cross]
    uu_col = uu[:, x_cross]

    # Plot the cross-sections
    plt.figure(figsize=(8, 5))
    plt.plot(u_col, label="Original", linewidth=2)
    plt.plot(f4_col, label="Fourier Recon", linestyle='--')
    plt.plot(uu_col, label="iRadon Recon", linestyle=':')
    plt.title(f"Vertical Cross-section at x = {x_cross} (na = {na_cross})")
    plt.xlabel("Pixel index (y-axis)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


