import numpy as np 
import matplotlib.pyplot as plt
from skimage.transform import radon

'''
Lecture 1
Assignment 1

We compute the exact radon transform of a uniform square of [-1,1]^2 and compare it to the result of the radon transform of the square

This code includes: 
- function to compute the exact radon
- the code to plot the exact radon next to the numerical 
- computing the relative error and plotting it vs the angle 

'''

#################################################################################################################
# Setting up the uniform square
#################################################################################################################

# Setting up the parameters  
nx = 400
na = 400
theta = np.linspace(0., 180., na)

# Creating the uniform square in [-1.5,1.5]x[-1.5,1.5] grid
x = np.linspace(-1.5, 1.5, nx)
y = np.linspace(-1.5, 1.5, nx)
X, Y = np.meshgrid(x, y)
u = np.logical_and(np.abs(X) <= 1, np.abs(Y) <= 1).astype(float) 

#################################################################################################################
# Computing the numerical radon transform
#################################################################################################################

# Computing the radon transform 
f_numerical = radon(u, theta=theta)

# Computing and matching the pixel size to that of the exact transform 
pixel_size = (x[1] - x[0])                                                                  
f_numerical_scaled = f_numerical * pixel_size


#################################################################################################################
# The function that comoutes the exact/analytical radon transform 
#################################################################################################################

def f_exact(th, s):
    # Convert to radians
    th = np.deg2rad(th)  

    # Set up a list with all the corner points
    corners = np.array([
        np.cos(th)-np.sin(th),
        np.cos(th)+np.sin(th),
        -np.cos(th)-np.sin(th),
        -np.cos(th)+np.sin(th)
    ])

    # Sort the corner points
    corners_sorted=np.sort(corners)
    s1,s2,s3,s4=corners_sorted                  

    # Find the maximum intersection length for this theta
    if np.cos(th)==0 or np.sin(th)==0:
        l=2
    else:
        l=2*min(1/np.abs(np.cos(th)),1/np.abs(np.sin(th)))

    # Compute value of f(theta,s)
    f=np.zeros(len(s))
    s1s2=(s>s1)&(s<=s2)
    s2s3=(s>s2)&(s<=s3)
    s3s4=(s>s3)&(s<=s4)

    f[s1s2]=l*(s[s1s2]-s1)/(s2-s1)
    f[s2s3]=l
    f[s3s4]=l*(s4-s[s3s4])/(s4-s3)
    
    return f


#################################################################################################################
# Plot the analytical vs the numerical radon transform 
#################################################################################################################

# Use the function above to compute the analytical radon transform 
f_analytical=np.zeros((len(x),len(theta)))
for i in range(len(theta)):
    f_analytical[:,i]=f_exact(theta[i],x)

# Plot both the numerical and the analytical radon transform  
plt.figure(figsize=(11, 5))
plt.subplot(1, 2, 1)
plt.imshow(f_numerical_scaled, extent=(0, 180, -nx//2, nx//2), aspect='auto')
plt.title("Numerical radon transform of unit square")
plt.xlabel(r"Angle $\theta$ in degrees")
plt.ylabel(r"Distance $s$ along the detector in pixels")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(f_analytical, extent=(0, 180, -nx//2, nx//2), aspect='auto')
plt.title("Analytical radon transform of unit square")
plt.xlabel(r"Angle $\theta$ in degrees")
plt.ylabel(r"Distance $s$ along the detector in pixels")
plt.colorbar()
plt.show()


#################################################################################################################
# Plot the relative error between the two methods and plot this vs the angle 
#################################################################################################################

errors = np.zeros(len(theta))
for i in range(len(theta)):
    errors[i] = np.linalg.norm(f_analytical[:,i]-f_numerical_scaled[:,i]) / np.linalg.norm(f_analytical[:,i])

plt.plot(theta, errors)
plt.xlabel("Angle (degrees)")
plt.ylabel("Relative error")
plt.title("Radon transform relative error across angles")
plt.grid(True)
plt.show()
