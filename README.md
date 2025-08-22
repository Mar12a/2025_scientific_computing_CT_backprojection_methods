# scientific_computing_CT_backprojection_methods
**Course: Scientific Computing 2025**, Mathematics BSc

**Grade: 10/10**

A project which codes the back projection of the Radon transform using various methods which is used in order to construct the images from a CT scanner. This projected was created in collaboration with a fellow student. 

These lectures cover the basics of Computed Tomography:

* [Lecture 1](https://tristanvanleeuwen.github.io/InleidingSC2-CT/lecture1.html#/) History of CT, the Lambert-Beer law and the Radon Transform
* [Lecture 2](https://tristanvanleeuwen.github.io/InleidingSC2-CT/lecture2.html#/) The Fourier-Slice Theorem and Fourier reconstruction
* [Lecture 3](https://tristanvanleeuwen.github.io/InleidingSC2-CT/lecture3.html#/) Discretisation of the Radon transform
* [Lecture 4](https://tristanvanleeuwen.github.io/InleidingSC2-CT/lecture4.html#/) Image reconstruction using Filtered Back Projection 
* [Lecture 5](https://tristanvanleeuwen.github.io/InleidingSC2-CT/lecture5.html#/) Algebraic reconstruction methods
* [Lecture 6](https://tristanvanleeuwen.github.io/InleidingSC2-CT/lecture6.html#/) Practical aspects


## The included code: 
### Assignment_1_1.py 
We compute the exact radon transform of a uniform square of [-1,1]^2 and compare it to the result of the radon transform of the square

This code includes: 
- function to compute the exact radon
- the code to plot the exact radon next to the numerical 
- computing the relative error and plotting it vs the angle

### Assignment_1_2.py 
For the SL phantom inspect the influence of noise and angular sampling 

This code includes: 
- function which, for given theta and sigma produces data for the SL phantom, and returns the reconstructed image.
- graph noise vs error and anular sampling vs error to determine the influence of angular sampling and noise on the reconstruction error.

### Assignment_2_1.py 

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

### Assignment_2_2.py 
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

### Assignment_3_1.py 

A function RadonMatrix(n, s, theta) that discretizes the Radon transform of a given n × n image, defined on [-1, 1]^2, 
for given arrays s and theta and returns a matrix of size len(s)*len(theta) by n*n.

This code includes: 
- function to compute which pixels are intersected by a single ray defined by an angle (theta) and offset (s)
- function to create matrix that approximates the radon transform given angles and offsets
- the code to apply our implementation of the discrete radon transform on the SL phantom
- the code to compute the numerical radon transform of the SL phantom
- the code to graph both results 

### Assignment_3_2.py 

A function RadonSparseMatrix which returns a sparse matrix representation of RadonMatrix,
for given arrays s and theta and returns a matrix of size len(s)*len(theta) by n*n.

This code includes: 
- function to compute which pixels are intersected by a single ray defined by an angle (theta) and offset (s)
- function to create matrix that approximates the radon transform given angles and offsets
- function to create a sparse matrix that approximates the radon transform given angles and offsets
- function to measure the average time taken to perform f=A@u
- the code to run and plot the results to compare the different matrices coo, csr, csc

### Assignment_3_3.py
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

### Assignment_4.py 
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

### Assignment_5.py
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

