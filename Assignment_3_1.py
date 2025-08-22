import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon,resize

###############################################################

'''
Lecture 3
Assignment 1

A function RadonMatrix(n, s, theta) that discretizes the Radon transform of a given n × n image, defined on [-1, 1]^2, 
for given arrays s and theta and returns a matrix of size len(s)*len(theta) by n*n.

This code includes: 
- function to compute which pixels are intersected by a single ray defined by an angle (theta) and offset (s)
- function to create matrix that approximates the radon transform given angles and offsets
- the code to apply our implementation of the discrete radon transform on the SL phantom
- the code to compute the numerical radon transform of the SL phantom
- the code to graph both results 

'''

#################################################################################################################
# The function to compute which pixels are intersected by a single ray defined by an angle (theta) and offset (s)
#################################################################################################################

def Ray(n,theta,s):

    #Create grid from -1 to 1
    grid=np.linspace(-1,1,n+1)                                  #We have n+1 lines across n pixels

    sin_theta=np.sin(theta)                                     #Prevent repeated computation by storing these
    cos_theta=np.cos(theta)

    t_list=[]                                                   #Initialize list to store values t where the ray intersects pixel boundaries

    #Intersections with vertical lines
    if np.abs(sin_theta)>1e-10:                                 #Prevents division by zero                      
        for xi in grid:                                         #For every line between pixels
            t=(xi-s*cos_theta)/sin_theta                        #Find correct t
            y=t*cos_theta-s*sin_theta                           #Find y value
            if -1<=y<=1:                                        #If in image we add our t value to t_list
                t_list.append(t)
    
    #Intersections with horizontal lines
    if np.abs(cos_theta)>1e-10:                                 #Prevents division by zero      
        for yi in grid:                                         #For every line between pixels
            t=(yi+s*sin_theta)/cos_theta                        #Find correct t
            x=t*sin_theta+s*cos_theta                           #Find corresponding x value
            if -1<=x<=1:                                        #Check if intersection point is in image
                t_list.append(t)

    sorted_t_list=np.sort(t_list)                               #Sort intersections by t value                      
    if len(sorted_t_list)<2:                                    #Check if ray intersects image
        return np.array([],dtype=int),np.array([])              #If it does not, return empty arrays

    #Compute midpoints and segment lengths
    t_mid=[]                                                    #Initialize list to store middle between consecutive t's 
    t_length=[]                                                 #Initialize list to store segment lengths
    for i in range(len(sorted_t_list)-1):                       #Loop through every element of sorted_t_list except the last
        mid=(sorted_t_list[i]+sorted_t_list[i+1])/2             #Compute middle between this element and the next
        t_mid.append(mid)                                       #Store it in t_mid
        length=sorted_t_list[i+1]-sorted_t_list[i]              #Compute the lenght of the segment between this element and the next
        t_length.append(length)                                 #Store the length, (always nonnegative) in t_length 
    t_mid=np.array(t_mid)                                       #Ensure correct types
    t_length=np.array(t_length)

    #Compute x and y coordinates for t's in t_mid (arrays)
    x=t_mid*sin_theta+s*cos_theta
    y=t_mid*cos_theta-s*sin_theta

    #Find correspondig indices for the pixels where (x,y)'s are in
    i=np.floor((x+1)*n/2).astype(int)                           #Round down and cast to integer
    j=np.floor((y+1)*n/2).astype(int)
    valid=(i>=0)&(i<n)&(j>=0)&(j<n)                             #Check if indices correspond to an existing pixel                                              
    i=i[valid]                                                  #Only keep the valid indices
    j=j[valid]
    t_length=t_length[valid]

    #Convert indices in arrays i and j to a one dimensional array J
    J=j*n+i

    return J, t_length

#################################################################################################################
# The function to create matrix that approximates the radon transform given angles and offsets
#################################################################################################################

def RadonMatrix(n,theta,s):

    nt=len(theta)                                                           #Find length of theta (list)
    ns=len(s)                                                               #Find length of s (list)
    A=np.zeros((ns*nt,n**2))                                                #Initialize zero matrix with correct shape
    
    for i in range(nt):                                         
        for j in range(ns):
            J,w=Ray(n,theta[i],s[j])                                        #Use ray as defined above
            A[j*nt+i,J]=w                                                   #Fill the matrix A
    
    return A

############################################################################################################################################
# Applying our implementation of the discrete radon transform on the SL phantom
############################################################################################################################################
n=64
phantom=shepp_logan_phantom()                                                   #Load SL phantom and resize for correct shape           
phantom_resized = resize(phantom, (n, n), mode='reflect', anti_aliasing=True)

#Define theta and s
theta=np.linspace(0,np.pi,180,endpoint=False)                                   #180 angles from 0 to π
s=np.linspace(-1,1,n)                                                           #n offset positions

#Compute discrete radon transform
A=RadonMatrix(n,theta,s)                                        

#Make image 1D and create sinogram
sinogram=A@phantom_resized.ravel()
sinogram=sinogram.reshape(len(s),len(theta))

#Load SL phantom and resize for correct shape           
n2=256
phantom2=shepp_logan_phantom()                                   
phantom_resized2 = resize(phantom2, (n2, n2), mode='reflect', anti_aliasing=True)

#Define theta and s
theta2=np.linspace(0,np.pi,180,endpoint=False)                                  #180 angles from 0 to π
s2=np.linspace(-1,1,n2)                                                         #n offset positions

A2=RadonMatrix(n2,theta2,s2)                                                    #Compute discrete radon transform

#Make image 1D and create sinogram
sinogram2=A2@phantom_resized2.ravel()
sinogram2=sinogram2.reshape(len(s2),len(theta2))

#################################################################################################################
# Computing the numerical radon transform of the SL phantom
#################################################################################################################

#Computing the radon transform n=64
f_numerical = radon(phantom_resized, theta=np.rad2deg(theta))

#Computing and matching the pixel size to that of the exact transform   
dx=2/n                                                                            #Width of a pixel                                                               
f_numerical_scaled=f_numerical*dx

#Computing the radon transform n=256
f_numerical2 = radon(phantom_resized2, theta=np.rad2deg(theta2))

#Computing and matching the pixel size to that of the exact transform   
dx2=2/n2                                                                          #Width of a pixel                                                               
f_numerical_scaled2=f_numerical2*dx2

#################################################################################################################
# Plotting both results
#################################################################################################################

#Plot the sinograms
plt.figure(figsize=(10,10))

#Discrete radon transform n=64
plt.subplot(2,2,1)
plt.imshow(sinogram,extent=(0,180,-1,1), aspect='auto')
plt.xlabel(r'Angle $\theta$ (degrees)')
plt.ylabel(r'Offset $s$')
plt.title(r'Discrete Radon Transform of Shepp-Logan Phantom, $n=64$')
plt.colorbar()

#Discrete radon transform n=256
plt.subplot(2,2,2)
plt.imshow(sinogram2,extent=(0,180,-1,1), aspect='auto')
plt.xlabel(r'Angle $\theta$ (degrees)')
plt.ylabel(r'Offset $s$')
plt.title(r'Discrete Radon Transform of Shepp-Logan Phantom, $n=256$')
plt.colorbar()

#Numerical radon transform from python n=64
plt.subplot(2,2,3)
plt.imshow(f_numerical_scaled, extent=(0,180,-1,1), aspect='auto')
plt.xlabel(r'Angle $\theta$ (degrees)')
plt.ylabel(r'Offset $s$')
plt.title(r'Numerical Radon Transform, $n=64$ (skimage)')
plt.colorbar()

#Numerical radon transform from python n=256
plt.subplot(2,2,4)
plt.imshow(f_numerical_scaled2, extent=(0,180,-1,1), aspect='auto')
plt.xlabel(r'Angle $\theta$ (degrees)')
plt.ylabel(r'Offset $s$')
plt.title(r'Numerical Radon Transform, $n=256$ (skimage)')
plt.colorbar()

plt.show()