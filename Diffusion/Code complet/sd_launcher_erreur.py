# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 16:18:17 2018

@author: Jeremy
"""

import main_diffusion_rk as main
import matplotlib.pyplot as plt
import numpy as np

store_error = np.zeros((2,5))
for i in range(1,3):
    for k in range(1,6):
            ###############################################################################
            #                                   INPUTS                                    #
            ###############################################################################
            scheme = k                         #version 1 sun, version 2 IP, version 3 LDG, version 4 BR2, version 5 Gassner
            D = 1                              #diffusion coefficient
            delta_t =  0.05                        #time step parameter
            n_cells = 100                       #number of cells in the mesh
            x_min = 0                          #left coordinate of the physical domain
            x_max = 100                         #right coordinate of the physical domain
            order_of_accuracy = i+2              #order of accuracy of the solution
                                               # (number of solution points = order_of_accuracy)
            n_time_step = 400                   #number of steps calculated
            dirichlet = 1;                     #dirichlet = 0, no dirichlet condition, 1 dirichlet condition (1 at right border)
            order_of_analytique_solution = 5;  #precion of the analytique solution (number of sol_point in the cell
                                                # where the analytique solution is calculated)
            initial_choice = 0                 #initialization 0 gaussian, 1 dirichlet, 2 stationary
            time_ech = 40                      #periode where we compare the solution to the analytique solution
            n_sampling_point = 10              #number of sampling point to represent the solution
            x_coordinates, simulated_time, initial_solution, results, theoretical_values,error_L2 = \
                            main.run_main(scheme, D, delta_t, n_cells, x_min, \
                                          x_max, order_of_accuracy, n_time_step,initial_choice,dirichlet,time_ech,\
                                          order_of_analytique_solution,n_sampling_point)
            
            if (scheme == 1):
                schema = 'Sun'
            if (scheme == 2):
                schema = 'IP'
            if (scheme == 3):
                schema = 'LDG'
            if (scheme == 4):
                schema = 'BR2'
            if (scheme == 5):
                schema = 'Gassner'
            
            '''
            ###############################################################################
            #                     PLOT THE SOLUTION AT A GIVEN TIME STEP                  #
            ###############################################################################
            plt.plot(x_coordinates, initial_solution)
            plt.plot(x_coordinates, results)
            if (initial_choice == 0 ):
                plt.plot(x_coordinates, theoretical_values)
                plt.legend(['t = 0', \
                            't =' + str(round(simulated_time, 1)), \
                            'therorical solution at ' + str(round(simulated_time, 1))])
                plt.title(schema+', time step = '+str(delta_t)+', degree ='+str(order_of_accuracy)+', error L2 = '+str(error_L2))
            else :
                plt.legend(['t = 0', \
                            't =' + str(round(simulated_time, 1))])
                plt.title(schema+', time step = '+str(delta_t)+', degree ='+str(order_of_accuracy))
            '''  
            print(error_L2)
            
            store_error[i-1,k-1] = delta_t*error_L2

