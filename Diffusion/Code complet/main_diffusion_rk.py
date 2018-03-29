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
VAR = 500                           #variance of the initial gaussian


def run_main(scheme, diffusion_coefficient, delta_t, n_cells, \
                           x_min, x_max, order_of_accuracy, n_time_step,initial_choice,dirichlet,time_ech,\
                           order_of_analytique_solution,n_sampling_point):
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
    sol_points, evol_matrix_1,evol_matrix_2 = sd_init.compute_mat_evolution(\
                                 scheme, diffusion_coefficient, delta_t, order_of_accuracy, coord)
    
    #computing analytique solution point nd extrapolation point
    sol_points_analytique,mat_global_extrapo_sol_sampling_analytique = sd_init.compute_sol_point_analytique(\
                                                                        order_of_analytique_solution,\
                                                                        n_sampling_point,\
                                                                        n_cells+1)
    
    #initializing the matrix where the result is going to be stocked
    results = np.zeros((order_of_accuracy*n_cells))
    alpha = sd_init.RKalpha6optim(order_of_accuracy-1)
    x_middle = coord[math.floor(len(coord)/2)]
    
    #initilization of the solution
    if (initial_choice == 0):
        initial_solution = np.zeros(n_cells*order_of_accuracy)   
        for i in range(0, n_cells-1):
            for k in range(0, order_of_accuracy):
                initial_solution[order_of_accuracy*i + k] = sd_init.gaussian(\
                  sd_init.iso2phi(coord[i], coord[i+1], sol_points[k]), x_middle, AMPLITUDE, VAR)
            
    if (initial_choice == 1):
        initial_solution = np.zeros(n_cells*order_of_accuracy)
        
    if (initial_choice == 2):
        initial_solution = np.zeros(n_cells*order_of_accuracy)
        x_middle = coord[math.floor(len(coord)/2)]
        for i in range(0, n_cells):
            for k in range(0, order_of_accuracy):
                initial_solution[order_of_accuracy*i + k] = ((coord[i]+coord[i+1])*0.5 +\
                                (coord[i+1]-coord[i])*0.5*sol_points[k])/coord[len(coord)-1]
         
#    initial_solution[:] = 0
#    initial_solution[0] = 30
    results = initial_solution
    
    #------------------------------------------------------------------------------
    #     REPLACE CFL WITH THE COMPUTATION OF THE HIGHEST EIGENVALUE
    #------------------------------------------------------------------------------
    
        
    ###############################################################################
    #                     COMPUTE THE SOLUTION OVER n_time_step                   #
    ###############################################################################
    
    #store some value of the solution
    if time_ech > 0:
        n_t_ech = math.ceil(n_time_step/time_ech)
        result_stored = np.zeros((n_t_ech,n_cells*order_of_accuracy))
    simulated_time = n_time_step*delta_t
    
    #loop for every time step
    
    #store the runge kutta intermediate values
    temp_new_value_at_sol_point = np.zeros(order_of_accuracy*n_cells)
    n_eval = 0
    for k in range(1, n_time_step):
        
        temp_new_value_at_sol_point = results
         
            
        #-----------------------------------------
        # RUNGE KUTTA LOOP
        #-----------------------------------------
        for j in range(0, SWEEP_RK):
            
            
            temp_new_value_at_flux_point = evol_matrix_1.dot(temp_new_value_at_sol_point)
            #conditions dirichlet
            if (dirichlet == 1 and scheme < 5):
                temp_new_value_at_flux_point[0] = 0
                temp_new_value_at_flux_point[(order_of_accuracy+1)*n_cells-1] = 0
            if (dirichlet == 1 and scheme == 5):
                temp_new_value_at_flux_point = temp_new_value_at_sol_point
                temp_new_value_at_flux_point[0] = 0
                temp_new_value_at_flux_point[(order_of_accuracy)*n_cells-1] = 0
            if (dirichlet == 0 and scheme == 5):
                temp_new_value_at_flux_point = temp_new_value_at_sol_point
                
            temp_new_value_at_sol_point = temp_new_value_at_sol_point + \
                    alpha[j]*evol_matrix_2.dot(temp_new_value_at_flux_point)
        
        #computing the new value of the solutions
        results = temp_new_value_at_sol_point
        
        if time_ech > 0:
            if (k%time_ech == 0):
                result_stored[math.floor(k/time_ech)] = results
                n_eval = n_eval +1
            
            if (k==n_time_step-1):
                result_stored[n_t_ech-1] = results
                n_eval = n_eval +1
                
    
    #store the value of the analytic solution
    x_coordinates_analytique = np.zeros(n_cells * (order_of_analytique_solution))
    theoretical_values = np.zeros(n_cells * order_of_analytique_solution)
    
    index_sol = 0
    for i in range(0, len(coord)-1):
        for j, solution_point in enumerate(sol_points_analytique):
            x_coordinates_analytique[index_sol] = sd_init.iso2phi(coord[i], coord[i+1], sol_points_analytique[j])
            index_sol = index_sol + 1
    index_sol = 0
                
    if (initial_choice == 0):
        for i, x_coord in enumerate(x_coordinates_analytique):
            theoretical_values[i] = \
            sd_init.brownien(\
                       x_coord, simulated_time, diffusion_coefficient, x_middle, AMPLITUDE, VAR)
                       
    if (initial_choice == 1):
        for i in range(0, n_cells):
            for k in range(0, order_of_analytique_solution):
                theoretical_values[order_of_analytique_solution*i + k] = ((coord[i]+coord[i+1])*0.5 +\
                              (coord[i+1]-coord[i])*0.5*sol_points_analytique[k])/coord[len(coord)-1]
    
    print(n_eval)
    error_L2 = 0
    if time_ech > 0: 
        for k in range(1,n_t_ech):
            simulated_time = ((k+1)*time_ech ) * delta_t
            error_L2 = error_L2 + sd_init.compute_error(result_stored[k],simulated_time,n_sampling_point,sol_points,order_of_accuracy,coord,x_middle,\
                  AMPLITUDE,VAR,diffusion_coefficient)

     
    pos_all_sampling_points,mat_global_extrapo_sol_sampling = \
         sd_init.compute_sampling_point_and_mat_extrapo(order_of_accuracy-1 ,sol_points,coord,n_sampling_point) 
    sol_all_sampling_point = mat_global_extrapo_sol_sampling.dot(results)
    sol_theoretical_all_sampling_point = mat_global_extrapo_sol_sampling_analytique.dot(theoretical_values)
               
        
    initial_solution = mat_global_extrapo_sol_sampling.dot(initial_solution)
    
    return pos_all_sampling_points, simulated_time, initial_solution, sol_all_sampling_point, sol_theoretical_all_sampling_point,error_L2
