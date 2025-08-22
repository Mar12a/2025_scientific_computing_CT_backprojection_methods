import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon
from skimage.data import shepp_logan_phantom

'''
Lecture 1
Assignment 2

For the SL phantom inspect the influence of noise and angular sampling 

This code includes: 
- function which, for given theta and sigma produces data for the SL phantom, and returns the reconstructed image.
- graph noise vs error and anular sampling vs error to determine the influence of angular sampling and noise on the reconstruction error.

'''

#################################################################################################################
# The function to construct and reconstruct the SL phantom 
#################################################################################################################

def sheppardrun(sigma, na):
    # set the variables 
    nx = 400                                                        # number of spatial points   
    theta = np.linspace(0., 180., na, endpoint=False)               # angular grid

    # import the SL phantom 
    u = shepp_logan_phantom()                                       

    # create the sinogram
    f = radon(u, theta=theta)

    # add noise to the sinogram 
    f_noisy = f + sigma * np.random.randn(nx,na)

    # reconstruct the SL phantom 
    u_fbp = iradon(f_noisy,theta=theta)

    # compute the error
    error = np.linalg.norm(u_fbp - u)/np.linalg.norm(u)

    ###############################################################

    # Code that can be used to plot: 
    # [0] the original image 
    # [1] the reconstructed image 
    # [2] the difference between [0] and [1] to show where the images differ the most clearly 
    
    '''
    diff = abs(u_fbp - u)
    fig,ax = plt.subplots(1,3)

    ax[0].imshow(u,extent=(-1,1,1,-1),vmin=0)
    ax[0].set_xlabel(r'$x$')
    ax[0].set_ylabel(r'$y$')
    ax[0].set_aspect(1)

    ax[1].imshow(u_fbp,extent=(-1,1,1,-1),vmin=0)
    ax[1].set_xlabel(r'$x$')
    ax[1].set_ylabel(r'$y$')
    ax[1].set_aspect(1)

    ax[2].imshow(diff,extent=(-1,1,1,-1),vmin=0)
    ax[2].set_xlabel(r'$x$')
    ax[2].set_ylabel(r'$y$')
    ax[2].set_aspect(1)

    fig.tight_layout()
    fig.suptitle(f"Noise Level; σ = {sigma}, and number of angles; na = {na}.")
    plt.show()
    '''
    return error


#################################################################################################################
# Plotting the influence of sigma and the angular sampling
#################################################################################################################


###############################################################
# First plot: error vs. sigma

# Set the variables
sigma_list = np.logspace(-10, 10, 10)                                                           # values of sigma
na = 400                                                                                        # the number of angles sampled 
theta = np.linspace(0., 180., na)                                                               # the corresponding list of angles

# Running the function above for different sigma values and recording the error
error_list_sig = []
for i in range(len(sigma_list)):
    print("sigma =" + str(sigma_list[i]))
    error_list_sig.append(sheppardrun(sigma_list[i], na))

# Plotting the best fit line for the data for sigma > 1
sigma_fit = [s for s in sigma_list if s > 1]                                                    # Create filtered list for sigma > 1
error_fit = [e for s, e in zip(sigma_list, error_list_sig) if s > 1]                            # Create filtered list for sigma > 1

c_fit = sum(e * s for s, e in zip(sigma_fit, error_fit)) / sum(s**2 for s in sigma_fit)         # Compute best-fit coefficient c in e_r = c * sigma
a = np.linspace(1, sigma_list[-1], 100)                             
b = c_fit * a                                                                                   # Generate the best-fit line using this c

# Find the horizontal line at the plateau for sigma < 1
plateau_value = error_list_sig[0]  

###############################################################
# Second plot: error vs. number of angles

# Set the parameters
na_list = [5, 10, 50, 100, 200, 300, 400, 500, 600, 1000]
sigma = 1
error_list_na = []

# Running the function above for different sigma values and recording the error
for i in range(len(na_list)):
    print("na =" + str(na_list[i]))
    error_list_na.append(sheppardrun(sigma, na_list[i]))


###############################################################
# Actually plotting both of them together 

fig, ax = plt.subplots(1, 2, figsize=(12, 5)) 

# Left plot: error vs sigma
ax[0].axhline(plateau_value, color='blue', linestyle='--', label=f'$e_{{min}} \\approx {plateau_value:.2f}$')
ax[0].plot(sigma_list, error_list_sig, 'o:g')                                
ax[0].plot(a, b, 'r-', label=f'Best Fit: $e_r = {c_fit:.2f} \\, \\sigma$')
ax[0].set_xlabel('Log(Sigma (Noise Level))')
ax[0].set_ylabel('Log(Reconstruction Error)')
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_title('Error vs. Noise Level on a log-log plot')
ax[0].grid(True)
ax[0].legend()

# Right plot: error vs number of angles
ax[1].plot(na_list, error_list_na, 'o:g')
ax[1].set_xlabel('Number of Angles')
ax[1].set_ylabel('Reconstruction Error')
ax[1].set_title('Error vs. Number of Angles')
ax[1].grid(True)

plt.tight_layout()
plt.show()
