# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 19:22:01 2017

@author: Eric Heulhard
"""
#script to compute the diffusion equation with the spectral difference method



###############################################################################
#                                   IMPORTS                                   #
###############################################################################
import math
import numpy as np
import sd_init


###############################################################################
#                                CONSTANTS                                    #
###############################################################################
SWEEP_RK = 6                       #sweep of RK scheme
AMPLITUDE = 1.0                    #amplitude of the initial gaussian
VAR = 10                           #variance of the initial gaussian


def run_main(scheme, diffusion_coefficient, delta_t, n_cells, \
                           x_min, x_max, order_of_accuracy, n_time_step):
    """
###############################################################################
# NAME : run_main
# DESC : Compute the solution using the SD method.
# INPUT : scheme                = 1 sun, version 2 IP, version 3 LDG, version 4 BR2
#         diffusion_coefficient = diffusion coefficient
#         delta_t               = time step parameter
#         n_cells               = number of cells in the mesh
#         x_min                 = left coordinate of the physical domain
#         x_max                 = right coordinate of the physical domain
#         order_of_accuracy     = order of accuracy of the solution
#                                 (number of solution points = order_of_accuracy)
#         n_time_step           = number of step calculated
# OUTPUT : x_coordinates        = x coordinates of solution points in the whole mesh
#          simulated_time       = simulated time
#          initial_solution     = initial solution at the x coordinates(see above)
#          results              = calculated solution at the x coordinates
#          theoretical_values   = theoretical solution at the x coordinates
###############################################################################"""
    
    
    ###############################################################################
    #                              COMPUTE THE MESH                               #
    ###############################################################################
    coord = np.linspace(x_min, x_max, n_cells+1) #Coordinates of the interfaces of the cells
    
    #computing evolution matrix
    sol_points, evol_matrix = sd_init.compute_mat_evolution(\
                                 scheme, diffusion_coefficient, delta_t, order_of_accuracy, coord)
    
    #initializing the matrix where the result is going to be stocked
    results = np.zeros((order_of_accuracy*n_cells))
    alpha = sd_init.RKalpha6optim(order_of_accuracy-1)
    
    #initilization of the solution
    initial_solution = np.zeros(n_cells*order_of_accuracy)
    x_middle = coord[math.floor(len(coord)/2)]
    for i in range(0, n_cells-1):
        for k in range(0, order_of_accuracy):
            initial_solution[order_of_accuracy*i + k] = sd_init.gaussian(\
              sd_init.iso2phi(coord[i], coord[i+1], sol_points[k]), x_middle, AMPLITUDE, VAR)
           
#    initial_solution[:] = 0
#    initial_solution[0] = 30
    results = initial_solution
    
    #------------------------------------------------------------------------------
    #     REPLACE CFL WITH THE COMPUTATION OF THE HIGHEST EIGENVALUE
    #------------------------------------------------------------------------------
    
        
    ###############################################################################
    #                     COMPUTE THE SOLUTION OVER n_time_step                   #
    ###############################################################################
    
    #loop for every time step
    
    #store the runge kutta intermediate values
    temp_new_value_at_sol_point = np.zeros(order_of_accuracy*n_cells)
    
    for k in range(1, n_time_step):
        
        temp_new_value_at_sol_point = results
        
        #conditions dirichlet
        temp_new_value_at_sol_point[0] = 0
        temp_new_value_at_sol_point[order_of_accuracy*n_cells-1] = 0
            
        #-----------------------------------------
        # RUNGE KUTTA LOOP
        #-----------------------------------------
        for j in range(0, SWEEP_RK):
            temp_new_value_at_sol_point = temp_new_value_at_sol_point + \
                    alpha[j]*evol_matrix.dot(temp_new_value_at_sol_point)
        
        #computing the new value of the solutions
        results = temp_new_value_at_sol_point
    
    
    simulated_time = n_time_step * delta_t
    x_coordinates = np.zeros(n_cells * order_of_accuracy)
    theoretical_values = np.zeros(n_cells * order_of_accuracy)
    
    index_sol = 0
    for i in range(0, len(coord)-1):
        for j, solution_point in enumerate(sol_points):
            x_coordinates[index_sol] = sd_init.iso2phi(coord[i], coord[i+1], solution_point)
            index_sol = index_sol + 1
    
    for i, x_coord in enumerate(x_coordinates):
        theoretical_values[i] = \
           sd_init.brownien(\
               x_coord, simulated_time, diffusion_coefficient, x_middle, AMPLITUDE, VAR)
           
           
    return x_coordinates, simulated_time, initial_solution, results, theoretical_values
