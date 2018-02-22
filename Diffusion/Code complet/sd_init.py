# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 16:41:14 2018

@author: Jeremy
"""

#Functions used to initialize the different matrices for sd

###############################################################################
#                                   IMPORTS                                   #
###############################################################################

import math
import numpy as np
from scipy import interpolate


###############################################################################
#                               CONSTANTS                                     #
###############################################################################

GAMMA = 0.5                        #GAMMA for continuity u
ETA = 0.5                          #ETA et BETA for continuity sigma
BETA = -0.5                        #add doi of kirby and Arnold

###############################################################################
#                  FUNCTIONS : compute_error_matrix                           #
###############################################################################
def compute_error_matrix(n_sampling, mesh, solution_points):
    
    """
###############################################################################
# NAME : compute_error_matrix
# DESC : Compute the global error matrix in order to find the error between sd and theory
# INPUT : n_sampling      = number of sampling points
#         mesh            = array that contains the coordinates of the cells in the mesh
#         solution_points = array that contains the coordinates of the solution points in [-1; 1]
# OUTPUT : global_matrix  = global error matrix
###############################################################################"""

    #compute the local error matrix
    local_matrix = np.zeros((n_sampling, n_sampling))
    identity_matrix = np.identity(n_sampling)

    for i in range(0, n_sampling):
        polynom_line = interpolate.lagrange(solution_points, identity_matrix[i])

        for j in range(0, n_sampling):
            polynom_column = interpolate.lagrange(solution_points, identity_matrix[j])
            polynom_lagrange = np.polyint(np.polymul(polynom_line, polynom_column))
            local_matrix[i, j] = polynom_lagrange(1) - polynom_lagrange(-1)

    #compute the global error matrix
    n_cells = len(mesh) - 1
    global_matrix = np.zeros((n_cells*n_sampling, n_cells*n_sampling))

    for i in range(0, n_cells*n_sampling):
        for j in range(0, n_sampling):
            global_matrix[i, n_sampling*math.floor(i/n_sampling)+j] = \
            (1/(mesh[math.floor(i/n_sampling)+1] - mesh[math.floor(i/n_sampling)])) \
            * local_matrix[i%n_sampling, j]

    return global_matrix



###############################################################################
#                  FUNCTIONS : compute_jump_av_matrix                         #
###############################################################################
def compute_jump_av_matrix(n_flux_points, n_cells):
    
    """
###############################################################################
# NAME : compute_jump_av_matrix
# DESC : Compute the jump and average matrices
# INPUT : n_flux_points = number of flux points in [-1; 1]
#         n_cells       = number of cells in the mesh
# OUTPUT : jump_matrix                    = jump matrix (removed)
#          average_matrix                 = average matrix
#          average_gradient_border_matrix = average gradient border matrix
#          zero_interface_matrix          = matrix zero interface
#          jump_zeros_matrix              = jump zeros matrix
#          average_zeros_matrix           = average zeros matrix
###############################################################################"""
    
    #Init matrices
    average_matrix = np.identity(n_flux_points*n_cells)
#    jump_matrix = np.identity(n_flux_points*n_cells)
    average_gradient_border_matrix = np.zeros((n_flux_points*n_cells, 2*n_cells))
    zero_interface_matrix = np.identity(n_flux_points * n_cells)
    jump_zeros_matrix = np.zeros((n_flux_points*n_cells, n_flux_points*n_cells))
    average_zeros_matrix = np.zeros((n_flux_points*n_cells, n_flux_points*n_cells))

    for i in range(1, n_cells):
        #Average matrix
        average_matrix[n_flux_points*i, n_flux_points*i] = 0.5
        average_matrix[n_flux_points*i-1, n_flux_points*i] = 0.5
        average_matrix[n_flux_points*i, n_flux_points*i-1] = 0.5
        average_matrix[n_flux_points*i-1, n_flux_points*i-1] = 0.5
        
        #Jump matrix
#        jump_matrix[n_flux_points*i, n_flux_points*i] = -0.5
#        jump_matrix[n_flux_points*i-1, n_flux_points*i] = -0.5
#        jump_matrix[n_flux_points*i, n_flux_points*i-1] = 0.5
#        jump_matrix[n_flux_points*i-1, n_flux_points*i-1] = 0.5
        
        #Average gradient border matrix
        average_gradient_border_matrix[n_flux_points*i, 2*i-1] = 0.5
        average_gradient_border_matrix[n_flux_points*i-1, 2*i] = 0.5
        average_gradient_border_matrix[n_flux_points*i-1, 2*i-1] = 0.5
        average_gradient_border_matrix[n_flux_points*i, 2*i] = 0.5
        
        #Matrix zero interface
        zero_interface_matrix[n_flux_points*i, n_flux_points*i] = 0
        zero_interface_matrix[n_flux_points*i-1, n_flux_points*i-1] = 0
        
        #Average jump matrix
        jump_zeros_matrix[n_flux_points*i, n_flux_points*i] = -0.5
        jump_zeros_matrix[n_flux_points*i-1, n_flux_points*i] = -0.5
        jump_zeros_matrix[n_flux_points*i, n_flux_points*i-1] = 0.5
        jump_zeros_matrix[n_flux_points*i-1, n_flux_points*i-1] = 0.5
        
        #Average zeros matrix
        average_zeros_matrix[n_flux_points*i, n_flux_points*i] = 0.5
        average_zeros_matrix[n_flux_points*i-1, n_flux_points*i] = 0.5
        average_zeros_matrix[n_flux_points*i, n_flux_points*i-1] = 0.5
        average_zeros_matrix[n_flux_points*i-1, n_flux_points*i-1] = 0.5
        
    return(average_matrix, average_gradient_border_matrix, zero_interface_matrix,
           jump_zeros_matrix, average_zeros_matrix)
    
    
###############################################################################
#                  FUNCTIONS : compute_mat_jump_relvment                      #
###############################################################################
def compute_mat_jump_relvment(n_flux_points, n_cells):
    
    """
###############################################################################
# NAME : compute_mat_jump_relvment
# DESC : Compute the lifting matrix
# INPUT : n_flux_points = number of flux points in [-1; 1]
#         n_cells       = number of cells in the mesh
# OUTPUT : Lifting matrix
###############################################################################"""
    
    jump_lifting_matrix = np.zeros((n_cells*2, n_flux_points*n_cells))
    for i in range(1, n_cells):
        jump_lifting_matrix[2*i, n_flux_points*i] = -1
        jump_lifting_matrix[2*i, n_flux_points*i-1] = 1
        jump_lifting_matrix[2*i-1, n_flux_points*i] = -1
        jump_lifting_matrix[2*i-1, n_flux_points*i-1] = 1
    
    return jump_lifting_matrix


###############################################################################
#                  FUNCTIONS : compute_mat_evolution                          #
###############################################################################
def compute_mat_evolution(scheme, diffusion_coefficient, delta_t, order_of_accuracy, mesh):
    """
###############################################################################
# NAME : compute_mat_evolution
# DESC : Compute the global evolution matrix
# INPUT : scheme                = 1 sun, version 2 IP, version 3 LDG, version 4 BR2
#         diffusion_coefficient = diffusion coefficient
#         delta_t               = time step used when computing the runge kutta
#         order_of_accuracy     = order of accuracy of the scheme ( = number of solution points)
#         mesh                  = array containing the coordinates of the boundaries of each cells
# OUTPUT : Coordinates of solution points in [-1, 1]
#          Global evolution matrix
###############################################################################"""
    
    ###############################################################################
    #               COMPUTE THE EXTRAPOLATION AND EVOLUTION MATRICES              #
    ###############################################################################
    #sol_point : coordinates of the sololution points in [-1; 1]
    #flux_point : coordinates of the flux points in [-1; 1] (not used)
    #A1 : matrix of extrapolation from the sol point toward the flux point
    #A2 : matrix of extrapolation derivative
    #A3 : ???
    #A4 : ???
    #A5 : ???
    sol_point, flux_points, extrapol_matrix, deriv_matrix, A3, A4, A5 =\
    compute_point_and_mat_extrapo(order_of_accuracy-1, mesh)
    
    
    #compute jump and average matrix
    #B1 : jump matrix (not used)
    #B2 : average matrix
    #B3 : average gradient border matrix
    #B4 : matrix zero interface
    #B51 : average jump matrix
    #B52 : average zeros matrix
    #B6 : matrix to build the jump for the relevment
    B2, B3, B4, B51, B52 =\
                compute_jump_av_matrix(order_of_accuracy+1, len(mesh) - 1)
    B6 = compute_mat_jump_relvment(order_of_accuracy+1, len(mesh) - 1)
        
    if scheme == 1:
        global_evolution_matrix = B2.dot(extrapol_matrix)
        global_evolution_matrix = deriv_matrix.dot(global_evolution_matrix)
        global_evolution_matrix = extrapol_matrix.dot(global_evolution_matrix)
        global_evolution_matrix = B2.dot(global_evolution_matrix)
        global_evolution_matrix = deriv_matrix.dot(global_evolution_matrix)
        
    elif scheme == 2:
        M1 = B2.dot(extrapol_matrix)
        M1 = deriv_matrix.dot(M1)
        M1 = extrapol_matrix.dot(M1)
        M1 = B2.dot(M1)
        M1 = B4.dot(M1)
        
        M2 = B2.dot(extrapol_matrix)
        M2 = A3.dot(M2)
        M2 = B3.dot(M2)
        
        M3 = B51.dot(extrapol_matrix)
        
        global_evolution_matrix = (diffusion_coefficient*(delta_t*2))*\
                                  deriv_matrix.dot(M1 + M2 - ETA*M3)
        
    if scheme == 3:
        M12 = B2.dot(extrapol_matrix)
        M13 = GAMMA*B51.dot(extrapol_matrix)
        M11 = M12 + M13
        M11 = deriv_matrix.dot(M11)
        M11 = extrapol_matrix.dot(M11)
        M1 = B2.dot(M11)
        
        M2 = -ETA*B51.dot(extrapol_matrix)
        
        M3 = BETA*B51.dot(M11)
        
        global_evolution_matrix = (diffusion_coefficient*(delta_t*2))*\
                                   deriv_matrix.dot(M1 + M2 + M3)
        
    elif scheme == 4:
        
        #compute the inverse of the matrix of mass
        det = np.linalg.det(A4)
        if det == 0:
            print("relevment matrix is not inversible")
        else:
            A4 = np.linalg.inv(A4)
            
            M1 = B2.dot(extrapol_matrix)
            M1 = deriv_matrix.dot(M1)
            M1 = extrapol_matrix.dot(M1) 
            M1 = B4.dot(M1)
            
            M2 = B2.dot(extrapol_matrix)
            M2 = A3.dot(M2)
            M2 = B3.dot(M2)
            
            M3 = B6.dot(extrapol_matrix)
            M3 = A5.dot(M3)
            M3 = A4.dot(M3) 
            M3 = B52.dot(M3)
            
            global_evolution_matrix = (diffusion_coefficient*(delta_t*2))*\
                                       deriv_matrix.dot(M1 + M2 - ETA*M3)
    
    elif scheme == 5:
        
        #Compute the special evolution matrix for Gassner
        
        inner_sol_deriv_matrix, left_grad_deriv_matrix, right_grad_deriv_matrix, \
        left_sol_deriv_matrix, right_sol_deriv_matrix, negative_values_matrix, \
        positive_values_matrix, sum_jump_matrix, sum_average_matrix, arrangement_matrix = \
        compute_gassner_matrices(sol_point, flux_points, mesh, delta_t, diffusion_coefficient)
        
        
        matrix_a = positive_values_matrix.dot(left_sol_deriv_matrix)
        matrix_b = negative_values_matrix.dot(right_sol_deriv_matrix)
        matrix_c = negative_values_matrix.dot(right_grad_deriv_matrix)
        matrix_d = positive_values_matrix.dot(left_grad_deriv_matrix)
        interface = sum_jump_matrix.dot(matrix_a + matrix_b) + sum_average_matrix.dot(matrix_c - matrix_d)*0.5
                                                                 
        global_evolution_matrix = deriv_matrix.dot(inner_sol_deriv_matrix + arrangement_matrix.dot(interface))*diffusion_coefficient*delta_t
        
    return sol_point, global_evolution_matrix

###############################################################################
#                  FUNCTIONS : RKalpha6optim                                  #
###############################################################################
def RKalpha6optim(polynom_degree):
    """
###############################################################################
# NAME : RKalpha6optim
# DESC : Return the RK6 coefficient used to calculate new time steps
# INPUT : polynom_degree = degree of the polynomial interpolation
# OUTPUT : alpha = vector of 6 coefficients
###############################################################################"""
    
    alpha = np.zeros(6)
    alpha[2] = 0.24662360430959
    alpha[3] = 0.33183954253762
    alpha[4] = 0.5
    alpha[5] = 1.0
    if polynom_degree == 2:
        alpha[0] = 0.05114987425612
        alpha[1] = 0.13834878188543
    elif polynom_degree == 3:
        alpha[0] = 0.07868681448952
        alpha[1] = 0.12948018884941
    elif polynom_degree == 4:
        alpha[0] = 0.06377275785911
        alpha[1] = 0.15384606858263
    elif polynom_degree == 5:
        alpha[0] = 0.06964990321063
        alpha[1] = 0.13259436863348
    elif polynom_degree == 6:
        alpha[0] = 0.06809977676724
        alpha[1] = 0.15779153065865
    elif polynom_degree == 7:
        alpha[0] = 0.06961281995158
        alpha[1] = 0.14018408222804
    elif polynom_degree == 8:
        alpha[0] = 0.07150767268798
        alpha[1] = 0.16219675431011
    elif polynom_degree == 9:
        alpha[0] = 0.06599710352324
        alpha[1] = 0.13834850670675
    elif polynom_degree == 10:
        alpha[0] = 0.07268810031422
        alpha[1] = 0.16368178688643
            
    return alpha

###############################################################################
#                  FUNCTIONS : compute_point_and_mat_extrapo                  #
###############################################################################
def compute_point_and_mat_extrapo(degree, mesh):
    
    """
###############################################################################
# NAME : compute_point_and_mat_extrapo
# DESC : function to compute the positions of the flux points and the sol 
#        points in an parametric cell. It will also compute the matrices for 
#        extrapolation and derivation
# INPUT : degree = degree of the polynomial interpolation knowing that :
#                  number of solution points = degree + 1
#                  number of flux points = degree + 2
#         mesh   = coordinates of the cells in the mesh
# OUTPUT : sol_point
#          flux_point
#          mat_global_extrapolation
#          mat_global_d_flux_at_sol_point
#          mat_global_grad_border
#          global_R
#          global_sec_mem_rev_mat
###############################################################################"""
    
    n_cells = len(mesh) - 1
    n_solution_points = degree + 1
    n_flux_points = degree + 2
    
    #Compute the flux points and solution points
    flux_point = np.zeros(n_flux_points)
    solution_point = np.zeros(n_solution_points)
    
    #Flux points
    polynom_legendre = np.polynomial.legendre.Legendre.basis(degree)
    roots_legendre = np.polynomial.legendre.Legendre.roots(polynom_legendre)
    
    for i in range(0, degree):
        flux_point[i+1] = roots_legendre[i]
    
    flux_point[0] = -1.
    flux_point[n_flux_points - 1] = 1.
    
    #Solution points
    polynom_chebyshev = np.polynomial.chebyshev.Chebyshev.basis(n_solution_points)
    solution_point = np.polynomial.chebyshev.Chebyshev.roots(polynom_chebyshev)
    
    #computing the flux_point and the sol_point in the iso cell
    #computing the legendre polynomial of degree p
    polynom_legendre = np.polynomial.legendre.Legendre.basis(n_flux_points - 2)
    roots = np.polynomial.legendre.Legendre.roots(polynom_legendre)
    flux_point[1:len(flux_point)-1] = roots[:]
    flux_point[0] = -1.
    flux_point[len(flux_point) - 1] = 1.
            
    #building the derivative matrix to compute the derivative of the flux at
    #sol point
    
    #building the extrapolation matrix sol point toward flux point
    local_extrapolation_matrix = np.zeros((n_flux_points, n_solution_points))
    identity_matrix = np.identity(n_solution_points)
    for j in range(0, n_solution_points):
        
        polynom_lagrange = interpolate.lagrange(solution_point, identity_matrix[j])
        
        for i in range(0, n_flux_points):
            local_extrapolation_matrix[i, j] = polynom_lagrange(flux_point[i])
    
    local_derivative_matrix = np.zeros((n_solution_points, n_flux_points))
    identity_matrix = np.identity(n_flux_points)
    
    for j in range(0, n_flux_points):
        
        polynom_lagrange = interpolate.lagrange(flux_point, identity_matrix[j])
        polynom_derivative_lagrange = np.polyder(polynom_lagrange, 1)
        
        for i in range(0, n_solution_points):
            local_derivative_matrix[i, j] = polynom_derivative_lagrange(solution_point[i])
    
    #compute the global extrapolation matrix        
    global_extrapolation_matrix = np.zeros((n_cells*n_flux_points, n_cells*n_solution_points))
    
    for i in range(0, n_cells*n_flux_points):
        for j in range(0, n_solution_points):
            global_extrapolation_matrix[i, n_solution_points*math.floor(i/n_flux_points)+j] =\
            local_extrapolation_matrix[i%n_flux_points, j]
     
    #compute the global extrapolation matrix        
    global_derivative_matrix = np.zeros((n_cells*n_solution_points, n_cells*n_flux_points))
            
    for i in range(0, n_cells*n_solution_points):
        for j in range(0, n_flux_points):
            global_derivative_matrix[i, n_flux_points*math.floor(i/n_solution_points)+j] =\
            (1/(mesh[math.floor(i/n_solution_points)+1] - mesh[math.floor(i/n_solution_points)]))*\
            local_derivative_matrix[i%n_solution_points, j]
            
    #compute the vector to compute gradient of u at 1 and -1
    local_gradient_matrix = np.zeros((2, n_flux_points))
    identity_matrix = np.identity(degree+2)
    
    for j in range(0, n_flux_points):
        
        polynom_lagrange = interpolate.lagrange(flux_point, identity_matrix[j])
        polynom_derivative_lagrange = np.polyder(polynom_lagrange, 1)
        
        for i in range(0, 2):
            if i == 0:
                local_gradient_matrix[i, j] = polynom_derivative_lagrange(-1)
            if i == 1:
                local_gradient_matrix[i, j] = polynom_derivative_lagrange(1)
    
    #compute the global mat_grad_border matrix
    global_gradient_matrix = np.zeros((2*n_cells, n_cells*n_flux_points))
    for i in range(0, n_cells*2):
        for j in range(0, n_flux_points):
            global_gradient_matrix[i, n_flux_points*math.floor(i/2)+j] =\
            local_gradient_matrix[i%2, j]
            
    #compute the local relevment matrix
    local_lifting_matrix = np.zeros((n_flux_points, n_flux_points))
    identity_matrix = np.identity(n_flux_points)
    for i in range(0, n_flux_points):
        
        polynom_line = interpolate.lagrange(flux_point, identity_matrix[i])
        
        for j in range(0, n_flux_points):
            polynom_column = interpolate.lagrange(flux_point, identity_matrix[j])
            polynom_lagrange = np.polyint(np.polymul(polynom_line, polynom_column))
            local_lifting_matrix[i, j] = polynom_lagrange(1) - polynom_lagrange(-1)

    #compute the global relevment matrix
    global_lifting_matrix = np.zeros((n_cells*n_flux_points, n_cells*n_flux_points))
    for i in range(0, n_cells*n_flux_points):
        for j in range(0, n_flux_points):
            global_lifting_matrix[i, n_flux_points*math.floor(i/n_flux_points)+j] =\
            (1/(mesh[math.floor(i/n_flux_points)+1]-mesh[math.floor(i/n_flux_points)]))*\
            local_lifting_matrix[i%n_flux_points, j]
    
    #compute the matrix to build the second member of the relevment equation
    loc_sec_mem_rev_mat = np.zeros((n_flux_points, 2))
    identity_matrix = np.identity(n_flux_points)
    for i in range(0, n_flux_points):
        polynom_line = interpolate.lagrange(flux_point, identity_matrix[i])
        loc_sec_mem_rev_mat[i, 0] = polynom_line(-1)
        loc_sec_mem_rev_mat[i, 1] = -polynom_line(1)
        
    #compute the global matrix to build the second member of the relevment equation
    global_sec_mem_rev_mat = np.zeros((n_flux_points*n_cells, 2*n_cells))
    for i in range(0, n_flux_points*n_cells):
        for j in range(0, 2):
            global_sec_mem_rev_mat[i, 2*math.floor(i/n_flux_points)+j] =\
            loc_sec_mem_rev_mat[i%n_flux_points, j]
            
    return(solution_point, flux_point, global_extrapolation_matrix,
           global_derivative_matrix, global_gradient_matrix, global_lifting_matrix,
           global_sec_mem_rev_mat)
    
    
###############################################################################
#                  FUNCTIONS : brownien                                       #
###############################################################################
def gaussian(x_coordinate, x_origin, amplitude, var):
    """
###############################################################################
# NAME : gaussian
# DESC : function to compute the initial solution as a gaussian
# INPUT : x_coordinate = coordinate at which to compute the value
#         x_origin     = mean value of the gaussian
#         amplitude    = amplitude of the gaussian
#         var          = variance of the gaussian
# OUTPUT : value of the gaussian at x
# NOTE : The initial solution is defined as found in MasterThesis_Joncquieres
###############################################################################"""
    gauss = amplitude * np.exp(-((x_coordinate - x_origin)**2)/var)
    return gauss
    
    
###############################################################################
#                  FUNCTIONS : brownien                                       #
###############################################################################
def brownien(x_coordinate, time, diffusion_coefficient, x_origin, amplitude, var):  
    """
###############################################################################
# NAME : brownien
# DESC : function to compute the theoretical solution
# INPUT : x_coordinate          = coordinates of the cells in the mesh
#         time                  = time at which to compute the value
#         diffusion_coefficient = diffusion coefficient
#         x_origin              = middle point coordinate of the gaussian
#         amplitude             = amplitude of the initial gaussian curve
#         var                   = variance of the gaussian
# OUTPUT : value of the theoretical solution at time t
# NOTE : The theoretical solution is defined as found in MasterThesis_Joncquieres
###############################################################################"""
    brownian_amplitude = amplitude / (2 * math.sqrt(diffusion_coefficient*time/var) + 1)
    brownian_x = (x_coordinate - x_origin)
    brownian_var = 4*diffusion_coefficient*time + var
    brow = brownian_amplitude * math.exp(-brownian_x**2 / brownian_var)
    return brow


############################
# NAME    : sd_iso2phi and sd_phi2iso
# DESC    : Compte x from phi cell to iso cell or reversed
# INPUTS  :- a - Lower bound of the phi intervalle
#          - b - upper bound of the phi intervalle
#          - x - x coordinate of the point to compute
# OUTPUTS : iso->phi or phi->iso
############################
def iso2phi(a, b, x):
    return ((b-a)*x + (b+a))/2

def phi2iso(a, b, x):
    return (2*x - (b + a)) / (b - a)

###############################################################################
#                   FUNCTIONS : GASSNER                                       #
###############################################################################
def compute_gassner_matrices(solution_points, flux_points, mesh, delta_t, diffusion_coefficient):
    """
###############################################################################
# NAME : compute_gassner_matrices
# DESC : function to compute the matrices used in the Gassner method
# INPUT : solution_points       = coordinates of solution points in [-1; 1]
#         flux_points           = coordinates of flux points in [-1; 1]
#         mesh                  = coordinates of the interfaces of the cells
#         delta_t               = time step
#         diffusion_coefficient = diffusion coefficient
# OUTPUT : global_inner_sol_deriv_matrix  = global matrix of derivation of the solution at the interior flux point
#          global_left_grad_deriv_matrix  = global matrix of derivation of the gradient at the exterior flux point 1
#          global_right_grad_deriv_matrix = global matrix of derivation of the gradient at the exterior flux point -1
#          global_left_sol_deriv_matrix   = global matrix of derivation of the solution at the exterior flux point 1
#          global_right_sol_deriv_matrix  = global matrix of derivation of the solution at the exterior flux point -1
#          negative_values_matrix         = matrix to arrange the -1 values
#          positive_values_matrix         = matrix to arrange the +1 values
#          sum_jump_matrix                = matrix to sum the jump of derivatives at the interface
#          sum_average_matrix             = matrix to sum the average of derivatives at the interface
#          arrangement_matrix             = matrix to place ext value fp around inner value fp
###############################################################################"""
    
    #Useful variables
    n_solution_points = len(solution_points)                       #number of solution points in [-1; 1]
    n_flux_points = len(flux_points)                               #number of flux points in [-1; 1]
    n_cells = len(mesh) - 1                                        #number of cells in the mesh
    max_derive = math.ceil(n_solution_points/2) + 1                #compute the max degree of derivation
    eta_g = 1/math.sqrt(math.pi*delta_t*diffusion_coefficient)     #compute eta_g
    identity_matrix = np.identity(n_solution_points)               #identity matrix
    polynom_lagrange = {}
    
    #Init the matrices
    local_inner_sol_deriv_matrix = np.zeros((n_flux_points, n_solution_points))
    local_left_grad_deriv_matrix = np.zeros((max_derive, n_solution_points))
    local_right_grad_deriv_matrix = np.zeros((max_derive, n_solution_points))
    local_left_sol_deriv_matrix = np.zeros((max_derive, n_solution_points))
    local_right_sol_deriv_matrix = np.zeros((max_derive, n_solution_points))
    
    negative_values_matrix = np.zeros((max_derive*(n_cells-1), max_derive*n_cells))
    positive_values_matrix = np.zeros(((n_cells-1)*max_derive, n_cells*max_derive))
    sum_jump_matrix = np.zeros((n_cells-1, max_derive*(n_cells-1)))
    sum_average_matrix = np.zeros((n_cells-1, max_derive*(n_cells-1)))
    arrangement_matrix = np.zeros((n_cells*n_flux_points, n_cells-1))
    
    global_inner_sol_deriv_matrix = np.zeros((n_flux_points*n_cells, n_solution_points*n_cells))
    global_left_grad_deriv_matrix = np.zeros((max_derive*n_cells, n_solution_points*n_cells))
    global_right_grad_deriv_matrix = np.zeros((max_derive*n_cells, n_solution_points*n_cells))
    global_left_sol_deriv_matrix = np.zeros((max_derive*n_cells, n_solution_points*n_cells))
    global_right_sol_deriv_matrix = np.zeros((max_derive*n_cells, n_solution_points*n_cells))
    
    #compute all lagrange polynomial for interpolation
    for i in range(0, n_solution_points):
        polynom_lagrange[i] = interpolate.lagrange(solution_points, identity_matrix[i])
    
    #compute the local matrix of derivation of the solution at the interior flux point
    for j in range(0, n_solution_points):
#        polynom_column = interpolate.lagrange(solution_points, identity_matrix[j])
        polynom_column_derivative = np.polyder(polynom_lagrange[j], 1)
        for i in range(1, n_solution_points):
            local_inner_sol_deriv_matrix[i, j] = polynom_column_derivative(flux_points[i])
           
    #compute the global matrix of derivation of the solution at the interior flux point
    for i in range(0, n_flux_points*n_cells):
        for j in range(0, n_solution_points):
            global_inner_sol_deriv_matrix[i, n_solution_points*math.floor(i/n_flux_points)+j] =\
            local_inner_sol_deriv_matrix[i%n_flux_points, j]*2/(mesh[(i+1)%n_flux_points]-mesh[i%n_flux_points])
    

    #compute the local matrix of derivationS of the gradient at the exterior flux point 1
    for j in range(0, n_solution_points):
#        polynom_column = interpolate.lagrange(solution_points, identity_matrix[j])
        for i in range(0, max_derive):
            polynom_column_derivative = np.polyder(polynom_lagrange[j], 2*i+1)
            local_left_grad_deriv_matrix[i, j] = polynom_column_derivative(1)
            
    #compute the global matrix of derivationS of the gradient at the exterior flux point 1
    for i in range(0, max_derive*n_cells):
        for j in range(0, n_solution_points):
            global_left_grad_deriv_matrix[i, n_solution_points*math.floor(i/max_derive)+j] =\
            local_left_grad_deriv_matrix[i%max_derive, j]
            
    #compute the local matrix of derivation of the gradient at the exterior flux point -1
    for j in range(0, n_solution_points):
#        polynom_column = interpolate.lagrange(solution_points, identity_matrix[j])
        for i in range(0, max_derive):
            polynom_column_derivative = np.polyder(polynom_lagrange[j], 2*i+1)
            local_right_grad_deriv_matrix[i, j] = polynom_column_derivative(-1)
            
    #compute the global matrix of derivation of the gradient at the exterior flux point -1
    for i in range(0, max_derive*n_cells):
        for j in range(0, n_solution_points):
            global_right_grad_deriv_matrix[i, n_solution_points*math.floor(i/max_derive)+j] =\
            local_right_grad_deriv_matrix[i%max_derive, j]
        
    #compute the local matrix of derivationS of the solution at the exterior flux point 1
    for j in range(0, n_solution_points):
#        polynom_column = interpolate.lagrange(solution_points, identity_matrix[j])
        for i in range(0, max_derive):
            polynom_column_derivative = np.polyder(polynom_lagrange[j], 2*i)
            local_left_sol_deriv_matrix[i, j] = polynom_column_derivative(1)
            
    #compute the global matrix of derivation of the solution at the exterior flux point 1
    for i in range(0, max_derive*n_cells):
        for j in range(0, n_solution_points):
            global_left_sol_deriv_matrix[i, n_solution_points*math.floor(i/max_derive)+j] =\
            local_left_sol_deriv_matrix[i%max_derive, j]
            
    #compute the local matrix of derivationS of the solution at the exterior flux point -1
    for j in range(0, n_solution_points):
        polynom_column = interpolate.lagrange(solution_points, identity_matrix[j])
        for i in range(0, max_derive):
            polynom_column_derivative = np.polyder(polynom_column, 2*i)
            local_right_sol_deriv_matrix[i, j] = polynom_column_derivative(-1)
            
    #compute the global matrix of derivation of the solution at the exterior flux point -1
    for i in range(0, max_derive*n_cells):
        for j in range(0, n_solution_points):
            global_right_sol_deriv_matrix[i, n_solution_points*math.floor(i/max_derive)+j] =\
            local_right_sol_deriv_matrix[i%max_derive, j]
        
    #compute the matrix to arrange the -1 values
    #compute the matrix to arrange the 1 values
    for i in range(0, n_cells-1):
        for j in range(0, max_derive):
            negative_values_matrix[max_derive*i + j, max_derive*(i+1) + j] = 1
            positive_values_matrix[max_derive*i + j, max_derive*i + j] = -1
            
    #compute the matrix to sum the jump of derivatives at the interface
    for i in range(0, n_cells-1):
        for j in range(0, max_derive):
            sum_jump_matrix[i, j + i*max_derive] = \
            (2/(mesh[(i+1)]-mesh[i]))**(j)*\
            (eta_g*4**j/(math.factorial(2*j+1) \
                      /(math.factorial(j+1)*math.factorial(j))))*\
                      (diffusion_coefficient*delta_t)**(j+1)/math.factorial(j+1)/delta_t
            
    #compute the matrix to sum the average of derivatives at the interface
    for i in range(0, n_cells-1):
        for j in range(0, max_derive):
            sum_average_matrix[i, j + i*max_derive] = \
            (2/(mesh[(i+1)]-mesh[i]))**(j+1)*\
            (diffusion_coefficient*delta_t)**(j+1)/math.factorial(j+1)/delta_t
    
    #compute the matrix to place ext value fp around int value fp
    for j in range(0, n_cells-1):
        for i in range(0, 2):
            arrangement_matrix[(j+1)*n_flux_points + i - 1, j] = 1
            
    return(global_inner_sol_deriv_matrix, \
          global_left_grad_deriv_matrix, \
          global_right_grad_deriv_matrix, \
          global_left_sol_deriv_matrix, \
          global_right_sol_deriv_matrix, \
          negative_values_matrix, \
          positive_values_matrix, \
          sum_jump_matrix, \
          sum_average_matrix, \
          arrangement_matrix)
    