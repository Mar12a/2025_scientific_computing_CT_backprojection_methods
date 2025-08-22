import numpy as np
import matplotlib.pyplot as plt

"""

Lecture 2
Assignment 1

Bi-linear interpolation from radial grid (ks,theta) to cartesian grid (kx,ky), 
related via kx = ks*cos(theta), ky = -ks*sin(theta).

The radial coordinates are assumed to include negative radii and theta in [0,pi).

Input: 
    f_hat     - 2d array of size (ns,ntheta) containing values on the radial grid
    theta, ks - 1d arrays of size nt and ns containing the polar gridpoints 
    (We use nt for theta, and ns for ks)
    kx, ky    - 1d arrays of size nx and ny containing the cartesian gridpoints
    These are the points we find an interpolated value for

Output:
    u_hat     - 2d array of size (nx,ny) containing the interpolated values


This code includes: 
- the function to interpolate as described above 
- a test case created by chat gpt 

"""


#################################################################################################################
# The interpolation function 
#################################################################################################################


def interpolate(f_hat, theta, ks, kx, ky):
    nt=len(theta)                                                       #Get number of angles 
    ns=len(ks)                                                          #Get number of different ks
    nx=len(kx)                                                          #Get number of x-points
    ny=len(ky)                                                          #Get number of y-points

    dtheta=theta[1]-theta[0]                                            #Compute spacing in angles
    dks=ks[1]-ks[0]                                                     #Compute spacing in radius

    u_hat=np.zeros((nx,ny),dtype=type(f_hat[0,0]))                      #Initialize output u_hat, allowing for complex values
    for i in range(nx):
        for j in range(ny):
            k_x=kx[i]                                                   #Get point from Cartesian grid
            k_y=ky[j]                                                   #Different from theta, ks from input
            k_s=np.sqrt(k_x**2+k_y**2)                                  #Use subscript to emphasize this gives a point
            k_theta=np.arctan2(-k_y,k_x)                                #Convert to polar coordinates output in [-pi,pi]
            if k_theta<0:
                k_theta+=np.pi                                          #Should be in [0,pi]
                k_s=-k_s                                                #Also make the radius negative

            lindex_theta=int((k_theta-theta[0])//dtheta)                #Compute left index
            lindex_ks=int((k_s-ks[0])//dks)                             #Cast to integer as mentioned in lecture

            if 0<=lindex_theta<=nt-2 and 0<=lindex_ks<=ns-2:            #Can interpolate if inside polar grid
                #Compute theta and ks difference between gridpoint and point to interpolate, ensuring weights will add up to 1
                tdiff=(k_theta-theta[lindex_theta])/dtheta 
                ksdiff=(k_s-ks[lindex_ks])/dks
                #Apply bilinear interpolation in polar grid
                u_hat[i,j]=(
                    f_hat[lindex_ks,lindex_theta]*(1-ksdiff)*(1-tdiff)+
                    f_hat[lindex_ks+1,lindex_theta]*ksdiff*(1-tdiff)+
                    f_hat[lindex_ks,lindex_theta+1]*(1-ksdiff)*tdiff+
                    f_hat[lindex_ks+1,lindex_theta+1]*ksdiff*tdiff
                )
            else:                                                       #Unable to interpolate
                u_hat[i,j]=0                                            #Does not impact the program, but is clearer as mentioned in lecture

    return u_hat


#################################################################################################################
# Test case: creating a sinogram with two horizontal stripes
#################################################################################################################

if __name__ == "__main__":

    '''
    We used Chatgpt to create the original sinogram for this example.
    Then we apply our bilinear interpolation.
    We plot both the original sinogram and our result after interpolating.
    '''

    '''

    # Define polar grid
    ntheta = 180
    ns = 200
    theta = np.linspace(0, np.pi, ntheta)
    ks = np.linspace(-100, 100, ns)

    # Create sinogram with two horizontal sine pulses
    f_hat = np.zeros((ns, ntheta))

    # Define the widths and centers of the sine pulses
    pulse_width = 20  # total width of pulse in ks units
    pulse1_center = -50
    pulse2_center = 50

    # Create smooth sine pulses
    pulse1_mask = np.logical_and(ks >= pulse1_center - pulse_width/2, ks <= pulse1_center + pulse_width/2)
    pulse2_mask = np.logical_and(ks >= pulse2_center - pulse_width/2, ks <= pulse2_center + pulse_width/2)

    # Create sine shapes within those regions
    f_hat[pulse1_mask, :] = np.sin(
        np.pi * (ks[pulse1_mask] - (pulse1_center - pulse_width/2)) / pulse_width)[:, np.newaxis]

    f_hat[pulse2_mask, :] = np.sin(
        np.pi * (ks[pulse2_mask] - (pulse2_center - pulse_width/2)) / pulse_width)[:, np.newaxis]


    #################################################################################################################
    # Test case: plotting the sinogram and the result of our interpolation 
    #################################################################################################################

    # Define Cartesian grid
    nx, ny = 200, 200
    kx = np.linspace(-100, 100, nx)
    ky = np.linspace(-100, 100, ny)

    #Apply our interpolation method
    u_hat_cart=interpolate(f_hat,theta,ks,kx,ky)                            

    plt.figure(figsize=(11,5))

    #Plot original sinogram
    plt.subplot(1,2,1)                                                     
    plt.imshow(f_hat,aspect='auto',extent=[theta[0], theta[-1], ks[0], ks[-1]])
    plt.title('Original sinogram with two pulses')
    plt.xlabel(r'Angle $\theta$ in radians')
    plt.ylabel(r'Radius $s$ in pixels')
    plt.colorbar()

    #Plot result after interpolation
    plt.subplot(1, 2, 2)                                                    
    plt.imshow(np.abs(u_hat_cart.T), extent=[kx[0], kx[-1], ky[0], ky[-1]])                         #Use abs convert complex to real
    plt.title('Resulting image after interpolating')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

    '''