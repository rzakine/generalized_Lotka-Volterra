########################################3
"""
The following code computes the evolution of N species in a 1 dimensional domain with periodic boundary conditions.
We use a semi-implicit scheme to solve the N coupled PDEs.
We are interested in the final abundances and final distribution of species in the domain.
"""
#######################

from numpy import loadtxt
from matplotlib import cm
import math
import cmath
import time
import os
import numba 
from numba import njit, jit, prange
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc

from itertools import count
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


start_time = time.time()

PI = math.pi
seed = 15    # seed for random number generator
np.random.seed(seed)


L  = 32     # domain size
Q  = 200    # number of mesh steps
h  = L/Q    # mesh size
N = 200     # number of species

dt = 0.1    # time step for dynamic evolution

D     = 0.2    # diffusion coefficient
a     = 1      # carying capacity
mu    = 0.4    # mean cooperation/competition
gamma = 0     # correlation
sigma = 0.1  # disorder amplitude for interactions

# Abundancies diverge for sigma > sigma_c = sqrt(2) for uncorrelated matrix, i.e. gamma=0.
# Divergence for sigma_c = 1/sqrt(2)=0.707 for fully symmetric, i.e. gamma=1




rng = np.random.default_rng()

################ Build interaction Matrix A
mean = [0, 0]
cov  = [[1, gamma], [gamma, 1]]  # diagonal covariance
x, y = rng.multivariate_normal(mean, cov, N*(N-1)//2).T
A = np.zeros((N,N))
c = 0
for i in range(0,N):
	for j in range(i+1,N):
		A[i,j] = mu/N + sigma* x[c]/np.sqrt(N) 
		A[j,i] = mu/N + sigma* y[c]/np.sqrt(N) 
		c = c+1
		#A[i,j] = A[j,i]  # fully symmetric
		#A[i,j] = -A[j,i] # fully asymmetric



####### Define density field

rho = 1*np.ones(N)

rho_space = np.ones((N,Q))
# introduce an initial perturbation on the fields
for i in range(0,N):
	rho_space[i,:] = rho[i]*( np.ones(Q)+np.random.normal(0,0.03,size=Q) )
rho = np.copy(rho_space)





##### Define linear operator for diffusion in implicit time scheme
LinOp = np.zeros((N,Q))
for i in range(0,N):
	kkd = np.pi * np.arange(0, Q)/Q
	LinOp[i] = 1./( 1 + D*dt/h**2*(2*np.sin(kkd))**2 )  

delta = 4 # absolute size of the kernel. One must have delta<L.


start_compile = time.time()
########### Define PDE solution scheme:
@njit
def simu(N, Q, L, delta, LinOp, rho , iterations, plotStep, dt):
	h = L/Q
	sizeKernelForMesh = int(delta/h) # delta * Q/L 
	data_abundances   = np.empty((N,iterations//plotStep+1), dtype=numba.f8)
	data_rho          = np.empty((N,Q, iterations//plotStep+1), dtype=numba.f8)
		
	######## Define door kernel:
	doorKernel   = np.zeros(Q) 
	for i in range(0, sizeKernelForMesh+1):
		doorKernel[i]  = 1/(2*sizeKernelForMesh+1)
		doorKernel[-i] = 1/(2*sizeKernelForMesh+1)
	doorKernel = doorKernel/np.sum(doorKernel)
	doorKernelFourier = np.fft.fft(doorKernel) 

	####### Define arrays in numba
	rhs_Fourier = np.zeros((N,Q), dtype=numba.c8)
	rho_Fourier = np.zeros((N,Q), dtype=numba.c8)
	
	for c in range(0, iterations+1):	
	
		## Semi-implicit scheme. 
		## Compute linear terms in Fourier space. 
		## Compute non linear terms in real space.
		
		rho_Fourier = np.fft.fft(rho) # Fourier transform field (last axis, so space, is used)
		
		# Compute spatial convolution in Fourier space and compute it back in real space
		nonLinearFourier = doorKernelFourier*np.fft.fft(1 - rho/a + A@rho)
		nonLinear        = np.fft.ifft(nonLinearFourier).real
		
		# Fourier transform right-hand side (nonlinear part):
		rhs = rho*nonLinear
		rhs_Fourier[:] = np.fft.fft(rhs[:])  
		
		# Compute timestep in Fourier domain:
		rho_Fourier_bis = LinOp*(rho_Fourier + dt * rhs_Fourier) 
		
		# Inverse Fourier transform:
		rho[:] = np.fft.ifft( rho_Fourier_bis[:] ).real  

		if(0==(c%plotStep)): # save data
			data_abundances[:,c//plotStep] = np.sum(rho, axis=1)/Q
			data_rho[:,:,c//plotStep]      = np.copy(rho)
			# Print time step and abundances:
			print(f"Time step: {c}")
			print("Abundances:")
			print(np.sum(h*rho, axis=1))
			print()

	return data_abundances, data_rho
print("--- Time for compilation: %.3f s ---" % (time.time() - start_compile))



#######################################################################
""" Start simulations """

iterations = 2000    # number of time steps
plotStep   = 200     # step for data extraction

data_array_size      = iterations//plotStep  
abundances, data_rho = simu(N, Q, L, delta, LinOp, rho , iterations+1, plotStep, dt)


##################################################################
""" Extract data """

### Extract final densities
np.savetxt("rho_final_N%d_L%d_Q%d_D%.2f_sigma%.2f_mu%.2f_gamma%.2f_seed%d.txt"%(N, L, Q, D,sigma,mu,gamma,seed), data_rho[:,:, data_array_size-1], fmt="%.6e" )

### Extract final abundances of each species
np.savetxt("abundances_final_N%d_L%d_Q%d_D%.2f_sigma%.2f_mu%.2f_gamma%.2f_seed%d.txt"%(N, L, Q, D,sigma,mu,gamma,seed), abundances[:,-1] )

### Extract array of survived species, the criterion can be changed
theta = np.where( (abundances[:, -1]<1e-15) , 0, 1) 
phi   = np.sum(theta)/N # fraction of surviving species

# Extract surviving fraction for given parameters:
np.savetxt(f"phi_and_parameters_N{N}_sigma{sigma:.2f}_mu{mu:.2f}_gamma{gamma:.2f}_seed{seed:d}.txt", np.array([phi, N, sigma, mu, gamma]), fmt="%.5e" )  


now = time.time()
print("--- Time to run simulation: %.3f seconds ---" % (now-start_time))






######################################################################
""" Plot data """

#### Plot biomass vs Time

fig = plt.figure(figsize=(5, 4))
ax  = fig.add_subplot(111)
for i in np.arange(0,N):
	ax.plot(np.arange(0,iterations+1,plotStep)*dt, abundances[i], lw= 0.7)
ax.set_title(r'Surviving fraction $\phi=$'+str(phi))
ax.set_yscale('log')
ax.set_xlabel(r'Time')
ax.set_ylabel(r'Biomasses $M_i$')
plt.savefig("mass_log-scale_N%d_L%d_Q%d_mu%.2f_sigma%.2f_gamma%.2f_c%d.png"%(N, L, Q, mu, sigma, gamma, c), bbox_inches='tight', dpi=150)
plt.close()
plt.clf()




#### Plot successive profile vs time

for c in range(0,iterations+1, plotStep):
	fig = plt.figure(figsize=(5, 4))
	ax  = fig.add_subplot(111)
	for i in np.arange(0,N,5): # plot 1/5 of species
		ax.plot(np.arange(0,Q)*h, data_rho[i,:,c//plotStep], lw= 0.7)
	ax.plot(np.arange(0,Q)*h, 1/(1-mu)*np.ones(Q), lw= 0.7, ls='--', color='k')
	ax.set_xlim(0,Q*h)
	ax.set_ylim(0,20)
	ax.set_xlabel(r'$x$')
	ax.set_ylabel(r'$\rho_i$')
	plt.savefig("N%d_L%d_Q%d_mu%.2f_sigma%.2f_gamma%.2f_c%d.png"%(N, L, Q, mu, sigma, gamma, c), bbox_inches='tight', dpi=150)
	plt.close()
	plt.clf()


#Plot final result
print("--- Total time to run code: %.3f seconds ---" % (time.time() - start_time))


 


