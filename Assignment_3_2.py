import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from time import perf_counter

###############################################################

'''
Lecture 3
Assignment 2

A function RadonSparseMatrix which returns a sparse matrix representation of RadonMatrix,
for given arrays s and theta and returns a matrix of size len(s)*len(theta) by n*n.

This code includes: 
- function to compute which pixels are intersected by a single ray defined by an angle (theta) and offset (s)
- function to create matrix that approximates the radon transform given angles and offsets
- function to create a sparse matrix that approximates the radon transform given angles and offsets
- function to measure the average time taken to perform f=A@u
- the code to run and plot the results to compare the different matrices coo, csr, csc

'''

#################################################################################################################
# Function to compute which pixels are intersected by a single ray defined by angle (theta) and offset (s)
#################################################################################################################

def Ray(n,theta,s):
   
    grid=np.linspace(-1,1,n+1)                                  #We have n+1 lines across n pixels

    sin_theta=np.sin(theta)
    cos_theta=np.cos(theta)

    t_list=[]

    # Intersections with vertical lines
    if np.abs(sin_theta)>1e-10:
        for xi in grid:
            t=(xi-s*cos_theta)/sin_theta
            y=t*cos_theta-s*sin_theta
            if -1<=y<=1:
                t_list.append(t)

    # Intersections with horizontal lines
    if np.abs(cos_theta)>1e-10:
        for yi in grid:
            t=(yi+s*sin_theta)/cos_theta
            x=t*sin_theta+s*cos_theta
            if -1<=x<=1:
                t_list.append(t)

    sorted_t_list=np.sort(t_list)
    if len(sorted_t_list)<2:
        return np.array([],dtype=int),np.array([])

    # Compute midpoints and segment lengths
    t_mid=[]
    t_length=[]
    for i in range(len(sorted_t_list)-1):
        mid=(sorted_t_list[i]+sorted_t_list[i+1])/2
        t_mid.append(mid)
        length=sorted_t_list[i+1]-sorted_t_list[i]
        t_length.append(length)
    t_mid=np.array(t_mid)
    t_length=np.array(t_length)

    # Get (x,y) for ray segments
    x=t_mid*sin_theta+s*cos_theta
    y=t_mid*cos_theta-s*sin_theta

    # Convert coordinates to pixel indices
    i=np.floor((x+1)*n/2).astype(int)
    j=np.floor((y+1)*n/2).astype(int)
    valid=(i>=0)&(i<n)&(j>=0)&(j<n)
    i=i[valid]
    j=j[valid]
    t_length=t_length[valid]

    J=j*n+i

    return J, t_length

#################################################################################################################
# Function to create dense matrix that approximates the Radon transform
#################################################################################################################

def RadonMatrix(n,theta,s):
    
    nt=len(theta)
    ns=len(s)
    A=np.zeros((ns*nt,n**2))
    for i in range(nt):                                         
        for j in range(ns):
            J,w=Ray(n,theta[i],s[j])
            A[j*nt+i,J]=w
    return A

#################################################################################################################
# Function to create sparse matrix that approximates the Radon transform
#################################################################################################################

def RadonSparseMatrix(n,theta,s,format='coo'):
    
    nt=len(theta)
    ns=len(s)
    weights=[]
    Is=[]
    Js=[]
    for i in range(nt):
        for j in range(ns):
            J,w=Ray(n,theta[i],s[j])
            weights+=w.tolist()
            Is+=[j*nt+i]*len(J)
            Js+=J.tolist()
    A=coo_matrix((weights,(Is,Js)),shape=(nt*ns,n*n))
    if format=='csr':
        return A.tocsr()
    elif format=='csc':
        return A.tocsc()
    else:
        return A   

#################################################################################################################
# Function to measure the average time taken to perform f=A@u
#################################################################################################################

def TimeMatrixVector(A,N):
   
    n=A.shape[1]
    u=np.random.randn(n)
    T=np.zeros(N)
    for i in range(N):
        t0=perf_counter()
        f=A@u
        t1=perf_counter()
        T[i]=t1-t0
    return np.mean(T), np.std(T)

#################################################################################################################
# Code to test and compare matrix formats
#################################################################################################################

if __name__ == "__main__":

    # Define test parameters
    n=64
    theta=np.linspace(0,np.pi,180,endpoint=False)
    s=np.linspace(-1,1,n)
    N=10

    # Compute Radon matrices
    A_dense=RadonMatrix(n,theta,s)
    A_sparse_coo=RadonSparseMatrix(n,theta,s)
    A_sparse_csr=RadonSparseMatrix(n,theta,s,'csr')
    A_sparse_csc=RadonSparseMatrix(n,theta,s,'csc')

    # Validate matrix equivalence
    print(f"Difference between dense and coo is: {np.linalg.norm(A_dense-A_sparse_coo)/np.linalg.norm(A_dense)}")
    print(f"Difference between dense and csr is: {np.linalg.norm(A_dense-A_sparse_csr)/np.linalg.norm(A_dense)}")
    print(f"Difference between dense and csc is: {np.linalg.norm(A_dense-A_sparse_csc)/np.linalg.norm(A_dense)}")

    # Measure timing for f=A@u
    print(f"Average time taken to compute f=A@u for A dense, standard deviation: {TimeMatrixVector(A_dense,N)}")
    print(f"Average time taken to compute f=A@u for A sparse coo, standard deviation: {TimeMatrixVector(A_sparse_coo,N)}")
    print(f"Average time taken to compute f=A@u for A sparse csr, standard deviation: {TimeMatrixVector(A_sparse_csr,N)}")
    print(f"Average time taken to compute f=A@u for A sparse csc, standard deviation: {TimeMatrixVector(A_sparse_csc,N)}")

    # Measure timing for f=A^T@u
    print(f"Average time taken to compute f=A^T@u for A dense, standard deviation: {TimeMatrixVector(A_dense.T,N)}")
    print(f"Average time taken to compute f=A^T@u for A sparse coo, standard deviation: {TimeMatrixVector(A_sparse_coo.T,N)}")
    print(f"Average time taken to compute f=A^T@u for A sparse csr, standard deviation: {TimeMatrixVector(A_sparse_csr.T,N)}")
    print(f"Average time taken to compute f=A^T@u for A sparse csc, standard deviation: {TimeMatrixVector(A_sparse_csc.T,N)}")
