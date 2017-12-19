# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 10:53:38 2017

@author: Jeremy
"""

###############################################################################
###########                     IMPORTS                   #####################
###############################################################################

import numpy as np
import scipy as sc
import math
import matplotlib.pyplot as plot


###############################################################################
###########            FUNCTIONS TO INIT ARRAYS           #####################
###############################################################################


############################
# NAME    : sd_init_sp
# DESC    : Init the position of solution points in [0; 1]
# INPUTS  : - n_solution_points - number of solution points
# OUTPUTS : Array of n_solution_points with the x coordinates of solution points
############################
def sd_init_sp(n_solution_points):
    c = np.polynomial.chebyshev.Chebyshev.basis(n_solution_points)
    roots = np.polynomial.chebyshev.Chebyshev.roots(c)
    return roots


############################
# NAME    : sd_init_fp
# DESC    : Init the position of flux points in [0; 1]
# INPUTS  : - n_flux_points - number of flux points
# OUTPUTS : Array of n_flux_points with the x coordinates of flux points
############################
def sd_init_fp(n_flux_points):
    n = n_flux_points - 2
    c = np.polynomial.legendre.Legendre.basis(n)
    roots = np.polynomial.legendre.Legendre.roots(c)
    fp = np.zeros(n_flux_points)
    
    for i in range(0, n):
        fp[i+1] = roots[i]
    
    fp[0] = -1.
    fp[len(fp) - 1] = 1.
    return fp
    

############################
# NAME    : sd_init_mesh
# DESC    : Return the mesh array
# INPUTS  : - x - x coordinate of the point to compute
# OUTPUTS : mesh array of size n_cells + 1
############################
def sd_init_mesh(x_min, x_max, n_cells):
    return np.linspace(x_min, x_max, n_cells+1)


############################
# NAME    : sd_init_function
# DESC    : Compute the initial solution at point x
# INPUTS  : - x - x coordinate of the point to compute
# OUTPUTS : f(x)
############################
def sd_init_function(x):
    #We can decide to init with a gaussian curve for now
    # f : [x_min; x_max] -> R
    #         x          ->    1/(sigma*sqrt(2*pi)) * exp( -(x - mean)^2 / ( 2 * sigma^2))
    
    sigma = 0.15
    mean = 0.25
#    return 1.
    return 1/(sigma * math.sqrt(2 * math.pi)) * math.exp(-math.pow((x - mean),2) / (2 * sigma * sigma))
    

############################
# NAME    : sd_iso2phi and sd_phi2iso
# DESC    : Compte x from phi cell to iso cell or reversed
# INPUTS  :- a - Lower bound of the phi intervalle
#          - b - upper bound of the phi intervalle
#          - x - x coordinate of the point to compute
# OUTPUTS : iso->phi or phi->iso
############################
def sd_iso2phi(a, b, x):
    return ((b-a)*x + (b+a))/2

def sd_phi2iso(a, b, x):
    return (2*x - (b + a)) / (b - a)

############################
# NAME    : sd_init_solution
# DESC    : Init the solution at time 0
# INPUTS  : - x_min           - lower bound for x for the physical domain
#           - x_max           - upper bound for x for the physical domain
#           - mesh            - mesh to use to sample the initial solution
#           - solution_points - array of solution points in [0; 1]
# OUTPUTS : Array of size mesh_size*n_solution_points with the s(t=0) solution
############################
def sd_init_solution(x_min, x_max, mesh, solution_points):
    
    solution = np.zeros(((len(mesh)-1)*len(solution_points), 1)) #+2 to add the boundaries
    index = 0
    x = 0.
    
    #Boundaries solutions
    #solution.itemset(0, sd_init_function(x_min))
    #solution.itemset(len(solution)-1, sd_init_function(x_max))
    
    for i in range(0,len(mesh)-1):
        for j in range(0, len(solution_points)):
            x = sd_iso2phi(mesh[i], mesh[i+1], solution_points[j])
            solution[index] = sd_init_function(x)
            index = index + 1
    
    return solution


############################
# NAME    : sd_init_mat_flux
# DESC    : Init the flux interpolation matrix that contains the Lagrange coeficient at flux points from the solution points
# INPUTS  : - solution_points - coordinates of solution points in [0; 1]
#           - flux_points     - coordinates of flux points in [0; 1]
# OUTPUTS : Matrix[n_fp x n_sp] with matrix[i,j] = Lj(FPi)
############################
def sd_init_mat_flux(solution_points, flux_points):
    mat = np.zeros((len(flux_points), len(solution_points)))
    Y = np.identity(len(solution_points))

    for i in range(0, len(solution_points)):
        L = sc.interpolate.lagrange(solution_points, Y[i]) #We want the Li that is equal to 1 at sps
        for j in range(0, len(flux_points)):
            mat[j, i] = L(flux_points[j])
            
    return mat


############################
# NAME    : sd_init_mat_solution
# DESC    : Init the flux interpolation matrix that contains the Lagrange coeficient at solution points from the flux points
# INPUTS  : - solution_points - coordinates of solution points in [0; 1]
#           - flux_points     - coordinates of flux points in [0; 1]
# OUTPUTS : Matrix[n_sp x n_fp] with matrix[i,j] = Lj(SPi)
############################
def sd_init_mat_solution(solution_points, flux_points):
    mat = np.zeros((len(solution_points), len(flux_points)))
    Y = np.identity(len(flux_points))
    
    for i in range(0, len(flux_points)):
        H = sc.interpolate.lagrange(flux_points, Y[i])
        H_d = np.polyder(H, 1)
        for j in range(0, len(solution_points)):
            mat[j, i] = H_d(solution_points[j])
            
    return mat


###############################################################################
###########                  LOOP FUNCTIONS               #####################
###############################################################################
  
############################
# NAME    : sd_loop_riemann
# DESC    : Applies the Riemann solver to insure continuity of fluxes
# INPUTS  : - advection_speed - advection speed parameter
#           - flux_array    - array that contain fluxes for the whole physical domain
#           - n_flux_points - number of flux points within a cell
#           - n_cells       - Number of cells in the mesh
# OUTPUTS : New flux array that contains a continuous flux between each cells
############################  
def sd_loop_riemann(advection_speed, flux_array, n_flux_points, n_cells):
    continuous_flux = np.array(flux_array)
    n = n_flux_points
    
    #Boundaries conditions
    if advection_speed < 0:
        continuous_flux[len(continuous_flux) - 1] = continuous_flux[0]
    else:
        continuous_flux[0] = continuous_flux[len(continuous_flux) - 1]
    
    for i in range(1, n_cells):
        if advection_speed > 0:
            continuous_flux.itemset(i*n, continuous_flux[i*n-1])
        else:
            continuous_flux.itemset(i*n-1, continuous_flux[i*n])
    
    return continuous_flux
  
############################
# NAME    : sd_loop_compute_fluxes
# DESC    : First loop that calculates the fluxes with the extrapolation matrices
# INPUTS  : - solution          - solution vector with the value of the solution at solution points
#           - mesh              - mesh used to sample the solution
#           - mat_extrapol_flux - extrapolation matrix for fluxes : mat[i, j] = Lj(FPi)
#           - advection_speed   - parameter for the advection
#           - n_flux_points     - number of flux points in [0; 1]
#           - n_solution_points - number of solution points in [0; 1]
# OUTPUTS : Flux array on the whole mesh calculated at flux_points
############################  
def sd_loop_compute_fluxes(solution, mesh, mat_extrapol_flux, advection_speed, n_flux_points, n_solution_points):
    
    flux = np.zeros(((len(mesh)-1) * n_flux_points, 1))
    cell_solution = np.zeros((n_solution_points, 1))
    cell_flux = np.zeros((n_flux_points, 1))
    iso = 0.
    
    for i in range(0, len(mesh)-1):
        iso = (mesh[i+1] - mesh[i])/2
        cell_solution[0:len(cell_solution)] = solution[i*n_solution_points:(i+1)*n_solution_points]/iso
        cell_flux[0:len(cell_flux)] = mat_extrapol_flux.dot(cell_solution)
        flux[i*n_flux_points:(i+1)*n_flux_points] = cell_flux*iso
    
    return advection_speed * flux
   
############################
# NAME    : sd_loop_compute_solution
# DESC    : First loop that calculates the fluxes with the extrapolation matrices
# INPUTS  : - flux                  - flux vector with the value of the flux at flux points
#           - n_cells               - Mesh used to sample the solution
#           - mat_extrapol_solution - extrapolation matrix for solutions : mat[i, j] = Li(SPj)
#           - n_flux_points         - number of flux points in [0; 1]
#           - n_solution_points     - number of solution points in [0; 1]
# OUTPUTS : Solution array on the whole mesh calculated at solution_points
############################
def sd_loop_compute_solution(flux, mesh, mat_extrapol_solution, n_flux_points, n_solution_points):
    
    solution = np.zeros(((len(mesh)-1) * n_solution_points, 1))
    cell_flux = np.zeros((n_flux_points, 1))
    cell_solution = np.zeros((n_solution_points, 1))
    iso = 0.
    
    for i in range(0, len(mesh)-1):
        iso = (mesh[i+1] - mesh[i])/2
        cell_flux[0:len(cell_flux)] = flux[i*n_flux_points:(i+1)*n_flux_points]/iso
        cell_solution[0:len(cell_solution)] = mat_extrapol_solution.dot(cell_flux)
        solution[i*n_solution_points:(i+1)*n_solution_points] = cell_solution*iso
    
    return solution
   
############################
# NAME    : sd_phi
# DESC    : Compute the solution at t+dt
# INPUTS  : - solution        - solution vector calculated at solution points
#           - mesh            - Mesh used to sample the solution
#           - mat_inter_flux  - extrapolation matrix for flux : mat[i, j] = Li(FPj)
#           - mat_inter_sol   - extrapolation matrix for flux : mat[i, j] = Li(SPj)
#           - advection_speed - advection speed
#           - n_flux_points   - number of flux points in [0; 1]
#           - n_sol_points    - number of solution points in [0; 1]
# OUTPUTS : Solution array on the whole mesh calculated at solution_points at t+dt
############################
def sd_phi(solution, mesh, mat_inter_flux, mat_inter_sol, advection_speed, n_flux_points, n_sol_points):
    flux = sd_loop_compute_fluxes(solution, mesh, mat_inter_flux, advection_speed, n_flux_points, n_sol_points)
    flux_c = sd_loop_riemann(advection_speed, flux, n_flux_points, len(mesh)-1)
    d_flux = sd_loop_compute_solution(flux_c, mesh, mat_inter_sol, n_flux_points, n_sol_points)
    
    return d_flux

###############################################################################
###########               INITIAL PARAMETERS              #####################
###############################################################################
p = 2                        #Degree of the polynomial
N = 1500                      #Number of cells in the mesh
n_sp = p+1                   #Number of solution points
n_fp = p+2                   #Number of flux points
x_min = -0.5                 #minimal x coordinate of the physical domain
x_max = 2.                   #maximum x coordinate of the physical domain
dt = 0.01                    #Time spte
T = 10.                      #Simulated time
A = 10.                      #Advection speed


###############################################################################
###########                 MAIN OF THE CODE              #####################
###############################################################################

#Initialize
sp = sd_init_sp(n_sp)
fp = sd_init_fp(n_fp)
mesh = sd_init_mesh(x_min, x_max, N)
solution = sd_init_solution(x_min, x_max, mesh, sp)
inter_flux_mat = sd_init_mat_flux(sp, fp)
inter_sol_mat = sd_init_mat_solution(sp, fp)
x_sol_points = np.zeros(n_sp * N)
x_flux_points = np.zeros(n_fp * N)

#Copy of the initial solution
u0 = np.zeros((len(solution), 1))
u0[0:len(u0)] = solution[0:len(solution)]

index_sol = 0
index_flux = 0
for i in range(0,len(mesh)-1):
    for j in range(0, len(sp)):
        x_sol_points[index_sol] = sd_iso2phi(mesh[i], mesh[i+1], sp[j])
        index_sol = index_sol + 1
        

#Solving the advection equation
for i in range(0, 100):
    
#    Runge-Kutte RK4
    k1 = sd_phi(solution, mesh, inter_flux_mat, inter_sol_mat, A, n_fp, n_sp)
    k2 = sd_phi(solution + dt/2 * k1, mesh, inter_flux_mat, inter_sol_mat, A, n_fp, n_sp)
    k3 = sd_phi(solution + dt/2 * k2, mesh, inter_flux_mat, inter_sol_mat, A, n_fp, n_sp)
    k4 = sd_phi(solution + dt*k3, mesh, inter_flux_mat, inter_sol_mat, A, n_fp, n_sp)
    solution = solution + dt/6 * (k1 + 2*(k2 + k3) + k4)
    
#    flux = sd_loop_compute_fluxes(solution, mesh, inter_flux_mat, A, n_fp, n_sp)
#    flux_c = sd_loop_riemann(A, flux, n_fp, N)
#    d_flux = sd_loop_compute_solution(flux_c, mesh, inter_sol_mat, n_fp, n_sp)
    
#    solution = solution - dt*d_flux

print("MAX_FLUX = ", np.amax(flux_c))
print("CELL_SIZE = ", mesh[1] - mesh[0])
plot.plot(x_sol_points, u0, 'x')
plot.plot(x_sol_points, solution, 'x')