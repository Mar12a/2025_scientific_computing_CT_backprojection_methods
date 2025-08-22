import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.data import shepp_logan_phantom
from time import perf_counter
import tracemalloc

###############################################################

'''
Lecture 3
Assignment 3

Write a function Radon(u, n, s, theta) which performs the Radon transform of the given image u and the corresponding transpose 
RadonTranspose(f, n, s, theta), for given arrays s and theta and returns a matrix of size len(s)*len(theta) by n*n.

This code includes: 
- function to compute which pixels are intersected by a single ray defined by an angle (theta) and offset (s)
- function to create matrix that approximates the radon transform given angles and offsets
- function to create a sparse matrix that approximates the radon transform given angles and offsets
- function to computes the transpose of the radon transform
- function to measure the average time taken to perform f=A@u
- code to run and plot the results to compare the multiplication and transpose for the normal and matrix free implementation 
- code to compute the relative error between the matrix-free (Radon) implementation and the implementation using a matrix (RadonMatrix)
- code to compute the memory usage matrix-free and matrix-based implementation

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
    
    nt=len(theta)                                               #Find length of theta (list)
    ns=len(s)                                                   #Find length of s (list)
    A=np.zeros((ns*nt,n**2))                                    #Initialize zero matrix with correct shape
    for i in range(nt):                                         
        for j in range(ns):
            J,w=Ray(n,theta[i],s[j])                            #Use ray as defined above
            A[j*nt+i,J]=w                                       #Fill the matrix A
    return A

#################################################################################################################
# The function to perform the radon transform of a given image u
#################################################################################################################

def Radon(u,n,s,theta):
    
    nt=len(theta)                                               #Find length of theta (list)
    ns=len(s)                                                   #Find length of s (list)
    f=np.zeros((ns,nt))                                         #Initialize output sinogram with zeros

    for i in range(nt):
        for j in range(ns):
            #Apply Ray to obtain array with intersected pixels and use intersection lengths as weights
            J,w=Ray(n,theta[i],s[j])     
            #Use 2D J, take inner product of two numpy arrays to perform matrix-multiplication one row at a time                                           
            f[j,i]+=np.dot(np.array(w),u[np.unravel_index(J,u.shape, order='C')])  
    
    return f

#################################################################################################################
# The function to computes the transpose of the radon transform
#################################################################################################################

def RadonTranspose(f,n,s,theta):
    
    nt=len(theta)                                               #Find length of theta (list)
    ns=len(s)                                                   #Find length of s (list)
    u=np.zeros((n,n))                                           #Initialize output u with zeros
    u=u.ravel()                                                 #Make 1D to be easily compatible with J
    for i in range(nt):
        for j in range(ns):
            J,w=Ray(n,theta[i],s[j])                            #Use Ray
            u[J]+=f[j,i]*np.array(w)                            #Perform matrix-multiplication one column at a time 
    return u.reshape((n,n))

#################################################################################################################
# Function to measure the average time taken to perform f=A@u
#################################################################################################################

def TimeMatrixVector(A,N):
    n=A.shape[1]                                                #Take the number of columns of A
    u=np.random.randn(n)                                        #Create a random vector with the correct length
    T=np.zeros(N)                                               #Store the time for every computation
    for i in range(N):                                          #Do the computation N times
        t0=perf_counter()                                       #Start timer
        f=A@u                                                   #Perform computation
        t1=perf_counter()                                       #End timer
        T[i]=t1-t0
    return np.mean(T)                                           #Return the average time taken  

#################################################################################################################
# Code to run and plot the results to compare the multiplication and transpose for the normal and matrix free implementation 
#################################################################################################################

n=128
nt=180
ns=128
theta=np.linspace(0,np.pi,nt,endpoint=False)                    #180 angles between 0, pi radians
s=np.linspace(-1,1,ns)                                          #Create list with ns offsets

phantom=shepp_logan_phantom()                                   #Load SL phantom and resize for correct shape           
phantom_resized = resize(phantom,(n, n),mode='reflect',anti_aliasing=True)

f_direct=Radon(phantom_resized,n,s,theta)                       #Use Radon for matrix-free radon transform
A=RadonMatrix(n,theta,s)                                        #Use RadonMatrix 
f_matrix=A@phantom_resized.ravel()
f_matrix=f_matrix.reshape((ns,nt))

bp_direct=RadonTranspose(f_direct,n,s,theta)                    #Apply RadonTranspose to f_direct
bp_matrix=A.T@f_matrix.ravel()                                  #Multiply with transpose of A from RadonMatrix
bp_matrix=bp_matrix.reshape((n,n))

x=np.linspace(-1,1,n)
y=np.linspace(-1,1,n)


# Plotting 
plt.figure(figsize=(16, 12))
plt.subplot(2,2,1)
plt.imshow(f_direct, extent=(0,180,-1,1),aspect='auto')
plt.xlabel(r'Angle $\theta$ (degrees)')
plt.ylabel(r'Offset $s$')
plt.title("Matrix-free radon applied to SL-phantom")
plt.colorbar()

plt.subplot(2,2,2)
plt.imshow(f_matrix, extent=(0,180,-1,1),aspect='auto')
plt.xlabel(r'Angle $\theta$ (degrees)')
plt.ylabel(r'Offset $s$')
plt.title("Radon implementation with (dense) matrix")
plt.colorbar()

plt.subplot(2,2,3)
plt.imshow(bp_direct, extent=[x[0], x[-1], y[-1], y[0]])
plt.xlabel(r'Angle $\theta$ (degrees)')
plt.ylabel(r'Offset $s$')
plt.title("Matrix-free transpose of radon")
plt.colorbar()

plt.subplot(2,2,4)
plt.imshow(bp_matrix, extent=[x[0], x[-1], y[-1], y[0]])
plt.xlabel(r'Angle $\theta$ (degrees)')
plt.ylabel(r'Offset $s$')
plt.title("Transpose of radon using (dense) matrix")
plt.colorbar()

plt.show()


#################################################################################################################
# Compute the relative error between the matrix-free (Radon) implementation and the implementation using a matrix (RadonMatrix)
#################################################################################################################

error1=np.linalg.norm(f_direct-f_matrix)/np.linalg.norm(f_direct)
print(f"Relative error between implementations or radon with or without matrices: {error1}")
error2=np.linalg.norm(bp_direct-bp_matrix)/np.linalg.norm(bp_direct)
print(f"Relative error between implementations with or without matrices for transpose of radon: {error2}")

N=10                                                        #Amount of tests for comparing average time usage
T=np.zeros(N)                                               #Store the time for every computation
for i in range(N):                                          #Do the computation N times
    t0=perf_counter()                                       #Start timer
    f_direct = Radon(phantom_resized, n, s, theta)          #Perform Radon
    t1=perf_counter()                                       #End timer
    T[i]=t1-t0

print(f"Matrix-free radon time and standard deviation: {np.mean(T),np.std(T)}")

T=np.zeros(N)                                               #Store the time for every computation
for i in range(N):                                          #Do the computation N times
    t0=perf_counter()                                       #Start timer
    A=RadonMatrix(n,theta,s)                                #Use RadonMatrix 
    f_matrix=A@phantom_resized.ravel()
    t1=perf_counter()                                       #End timer
    T[i]=t1-t0

print(f"Matrix radon time and standard deviation: {np.mean(T),np.std(T)}")

T=np.zeros(N)                                               #Store the time for every computation
for i in range(N):                                          #Do the computation N times
    t0=perf_counter()                                       #Start timer
    bp_direct=RadonTranspose(f_direct,n,s,theta)            #Apply RadonTranspose to f_direct
    t1=perf_counter()                                       #End timer
    T[i]=t1-t0

print(f"Matrix-free transpose of radon time and standard deviation: {np.mean(T),np.std(T)}")

T=np.zeros(N)                                               #Store the time for every computation
for i in range(N):                                          #Do the computation N times
    t0=perf_counter()                                       #Start timer
    bp_matrix=A.T@f_matrix.ravel()                          #Multiply with transpose of A from RadonMatrix, since A is stored we do not need to compute A again
    t1=perf_counter()                                       #End timer
    T[i]=t1-t0

print(f"Matrix of transpose radon time and standard deviation: {np.mean(T),np.std(T)}")


#################################################################################################################
# Compute the memory usage matrix-free and matrix-based implementation
#################################################################################################################

#Memory usage matrix-free implementation
tracemalloc.start()
f_direct=Radon(phantom_resized,n,s,theta)
current,peak=tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"Memory used (matrix-free): Current = {current / 1e6} MB; Peak = {peak / 1e6} MB")

# Measure memory for matrix-based method
tracemalloc.start()
A=RadonMatrix(n,theta,s)
f_matrix=A@phantom_resized.ravel()
current,peak=tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"Memory used (matrix-based): Current = {current / 1e6} MB; Peak = {peak / 1e6} MB")
