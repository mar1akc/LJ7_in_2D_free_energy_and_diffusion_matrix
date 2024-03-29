# LJ7 in 2D: free energy and diffusion matrix
This package computes the free energy and the diffusion matrix for the LJ7 in 2D in the collective variables $(\mu_2,\mu_3)$, the second and the third central moments of the coordination numbers.
The provided directory Data contains all data files necessary for visualizing the free energy and the diffusion matrix at $\beta = 5$ and $\beta = 10$. Furthermore, the provided data allow you to evaluate the free energy and the diffusion matrix as well as their gradients at any query point in the collective variable space $(\mu_2,\mu_3)$.

Therefore, if you need the free energy and the diffusion matrix at $\beta = 5$ or $\beta = 10$, it suffices to download the directories Data and Figures and the ipynb file.

### The Lennard-Jones-7 (LJ7) system

Imagine seven particles of diameter in 2D interacting according to Lennard-Jones pair potential. Evaporation of the particles is enforced via the restraining potential that starts acting if an LJ particle deviates from the center of mass by a distance greater than 2.  The total potential energy of LJ7 system described by the vector of coordinates $x\in \mathbb{R}^{14}$ consists of the Lennard-Jones pair potential $LJ(x)$ and the restraining potential $R$:

$$U(x) = LJ(x) + R(x)$$

Further details on LJ7 setup and $(\mu_2,\mu_3)$ can be found in
[Evans, Cameron, Tiwary (ACHA 2023)](https://www.sciencedirect.com/science/article/pii/S1063520323000015) or [arXiv:2108.08979](https://arxiv.org/abs/2108.08979).

The overdamped Langevin dynamics 

$$dX_t =  - \nabla U(X_t)dt +\sqrt{2\beta^{-1}}dW_t$$

governs the system. The MALA time integrator [Roberts and Tweedy, 1996](https://en.wikipedia.org/wiki/Metropolis-adjusted_Langevin_algorithm) is used.

### Computation the free energy for LJ7 in 2D in the $(\mu_2,\mu_3)$ collective variables.

The procedure of computing the free energy consists of four steps. 

**Step 1.** First, we deposit Gaussian bumps with decaying heights by running well-tempered metadynamics as described in [Barducci, Bucci, and Parrinello (2008)] (https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.100.020603,
DOI: 10.1103/PhysRevLett.100.020603).
The sum of the deposited Gaussian bumps properly rescaled approximated the negative of the free energy. However, this approach for free energy estimation is unreliable as  suboptimal choices of the parameters gamma (the artificial temperature), the height and covariance function of the Gaussian bumps, and the total number of Gaussian bumps result in either wrong and/or noisy free energy estimate. Therefore, we use the biased potential for higher efficiency of the binning approach for free energy evaluation. 
 
**Step 2.** The evaluation of the sum of Gaussian bumps and its gradient is expensive. Therefore, we approximate biasing potential and its gradient via building a bicubic interplant as described in the Wiki article “Bicubic Interpolation”: https://en.wikipedia.org/wiki/Bicubic_interpolation.

**Step 3.** Then we run a long trajectory in the biased potential and bin each point in the process. The bin centers are located at the grid points of the bicubic interpolant.

**Step 4.** Finally, we estimate the free energy as follows.
The number of the binned points $N(i,j)$ in bin $(i,j)$ is proportional to $\exp(-\beta(F(i,j) + B(i,j))$ where $F$ is the desired free energy and $B$ is the biasing potential.
Hence, we evaluate the free energy as

$$F(i,j) = -{\log(N(i,j))\over \beta} - B(i,j).$$

We choose the free constant of the free energy so that its minimal value is zero.
We also build a bicubic interplant of the computed free energy. It allows us to evaluate it and its gradient on at any query point. Extrapolation is allowed.

### A guide for running codes
To run the C codes, open the Terminal and change the directory to the working directory where you copied the codes. Create GBumpsData and Data directories in your working directory or copy the provided directories and the data files in them into your working directory.
To run each C file, copy and paste the compile command from the description below or from the top section of the code file and press Return to compile. Then, if no errors occurred, type ./a.out to run.

**Step 1.** 
Run the code 
> LJ7in2D_WTMetad_mu2mu3.c.
Compile command: 

	gcc LJ7in2D_WTMetad_mu2mu3.c -lm -O3

The parameters are assigned as #define directives at the top of the code.
This run takes several hours.
This code will output the file with the heights and positions of the Gaussian bumps. For convenience, this file is moved to the directory GBumpsData and renamed so that the parameter values are specified in its name.

You do not need to do this step if a suitable file with Gaussian bumps already exists. If you need a different beta value, you do not need to rerun this code as the biasing potential does not need to be perfect.

The parameters for the well-tempered metadynamics are:

> BETA = 5.0 // the inverse temperature in the overdamped Langevin dynamics
>
> GAMMA = 1.0 // the decay parameter for the height of the bumps
>
> SIGMA = 0.02 // the standard deviation of the Gaussian bump function
>
> HEIGHT = 0.01 // the height of the Gaussian bump
>
> TAU = 5.0e-5 // the time step
>
> NSTEPS_BETWEEN_DEPOSITS 500 // the number of steps between depositions of Gaussian bumps
>
> NBUMPS_MAX 50000 // the total number of Gaussian bumps
>
> KAPPA = 100.0 // the spring constant for the restraining potential


**Remark.** Note that even if you are planning to compute the free energy and the diffusion matrix at a different value of $\beta$, it still makes sense to tun well-tempered metadynamics at $\beta = 5$ because the bumps are used only for biasing the trajectory to be binned and then unbiased.

> Output directory: GBumpsData
> 
> Output file: GbumpsData/GaussianBumps_beta5.txt


**Steps 2, 3, 4.** 
> Run the code LJ7in2D_bicubicFE_binning.c.

Compile command: 

	gcc LJ7in2D_bicubicFE_binning.c -lm -O3

> Input directory: GBumpsData

This code reads the input file 

> char fpot_name[] = "GBumpsData/GaussianBumps_beta5.txt";

The trajectory consists of 1e9 time steps. The run takes approximately 40 minutes.
This code outputs several files.

> Output directory: Data

The parameter file

> Data/bicubic_params.txt
 
the bins, 

> sprintf(fname,”Data/LJ7bins_beta%.0f.txt",BETA);

the computed free energy, 

> sprintf(fname,"Data/LJ7free_energy_beta%.0f.txt",BETA);

and set of bicubic matrices for bicubic interpolation of the free energy

> sprintf(fname,"Data/LJ7free_energy_bicubic_matrix_beta%.0f.txt",BETA).


### Computation of the diffusion matrix for LJ7 in the (mu2,mu3) collective variables

**Step 1.**
> Run the code LJ7in2D_prepare_configurations.c

Compile command:

	gcc LJ7in2D_prepare_configurations.c -lm -O3

> Input directory: GBumpsData

The input file is the data file with the WT Metadynamics deposited Gaussian bump data:

> GBumpsData/GaussianBumps_beta5.txt

Output file is an $N_1\cdot N_2$-by-14 array of initial configurations at the grid cells:

> LJ7bins_confs.txt 
	
The code computes the bicubic interplant, runs a trajectory of 1e7 steps in the biased potential, and saves one representative configuration at each cell.

A good value of BETA is 5.0 even if you need to compute the diffusion matrix at a different value of BETA.



**Step 2.** 
> Run the code LJ7in2D_diffusion_matrix.c
 
Compile command:  

	gcc LJ7in2D_diffusion_matrix.c -lm -O3

> Input directory: Data

 Input files.
A file generated by LJ7in2D_bicubicFE_binning.c that contains the computational domain and mesh parameters:
 
> bicubic_params.txt; 

A file generated by LJ7in2D_prepare_configurations.c.

> LJ7bins_confs.txt — initial configuration in the 14D space at each grid cell.

Output files:

> sprintf(fname,"Data/LJ7_M11_beta%.0f.txt",BETA);
> 
> sprintf(fname,"Data/LJ7_M12_beta%.0f.txt",BETA);
> 
> sprintf(fname,"Data/LJ7_M22_beta%.0f.txt",BETA);
 
These files are $N_2$-by-$N_1$ arrays of entries $M_{11}$, $M_{12}$, and $M_{22}$ of the diffusion matrix $M$, respectively.	

> sprintf(fname,"Data/LJ7_M11_bicubic_matrix_beta%.0f.txt",BETA);
> 
> sprintf(fname,"Data/LJ7_M12_bicubic_matrix_beta%.0f.txt",BETA);
> 
> sprintf(fname,"Data/LJ7_M22_bicubic_matrix_beta%.0f.txt",BETA);
 
These files are $N_1\cdot N_2$-by-16 arrays of the bicubic matrix coefficients that allow one to evaluate $M$ at any point in the CV space.

The code computes the entries $M_{11}$, $M_{12}$, and $M_{22}$ at all grid points where the initial configurations are available.
At each cell $(i,j)$, a trajectory of NSTEPS (line 25) steps in the potential biased with the spring force is run. The resulting potential is 

$$U(x) = LJ(x) + R(x) + 0.5*B_{\kappa}((CV_1(x)-CV_1[i])^2 + (CV_2(x)-CV_2[j])^2).$$

The constant $B_{\kappa}$ (BKAPPA) is set to 500.0, the time step is 1.0e-5.
Then the entries of M are obtained as

$$M_{11} = {1\over N_{steps}}\sum_{k=1}^{Nsteps} \sum_{j=1}^{dim}(dCV_1/dx_j)^2,$$

$$M_{12} = {1\over N_{steps}}\sum_{k=1}^{Nsteps} \sum_{j=1}^{dim}(dCV_1/dx_j)(dCV_2/dx_j),$$

$$M22 = {1\over N_{steps}}\sum_{k=1}^{Nsteps} \sum_{j=1}^{dim}(dCV_2/dx_j)^2.$$

The diffusion matrix is found at all other grid points via nearest neighbor interpolation.
The arrays of bicubic matrices are computed for $M_{11}$, $M_{12}$, and $M_{22}$ to enable their evaluation everywhere in the CV space.
If NSTEPS = 1e4, CPU time is ~152 seconds, if NSTEPS = 1e6, CPU time is ~4.5 hours (~15000 seconds).


### Visualization is done in Python
Run the Python code 

> LJ7free_energy_function.ipynb.

The free energy and its gradient as well as the diffusion matrix are visualized.
This file includes a function that computes the bicubic interplant from the bicubic matrix data produced by the C codes and shows how to use it in the example of recomputing the free energy, its gradient, and the diffusion matrix to a finer mesh.

