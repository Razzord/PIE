# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 16:18:17 2018

@author: Jeremy
"""

import main_diffusion_rk as main
import matplotlib.pyplot as plt
import numpy as np

store_dt = np.zeros((4,5))
for i in range(1,5):
    for k in range(1,6):
        dt = 0.025
        delta_t = 0.05
        error_L2 = 0
        while (error_L2 < 200):
            ###############################################################################
            #                                   INPUTS                                    #
            ###############################################################################
            scheme = k                         #version 1 sun, version 2 IP, version 3 LDG, version 4 BR2, version 5 Gassner
            D = 1                              #diffusion coefficient
            delta_t = delta_t + dt                     #time step parameter
            n_cells = 49                       #number of cells in the mesh
            x_min = 0                          #left coordinate of the physical domain
            x_max = 50                         #right coordinate of the physical domain
            order_of_accuracy = i              #order of accuracy of the solution
                                               # (number of solution points = order_of_accuracy)
            n_time_step = 30                   #number of steps calculated
            dirichlet = 0;                     #dirichlet = 0, no dirichlet condition, 1 dirichlet condition (1 at right border)
            
            initial_choice = 0                 #initialization 0 gaussian, 1 dirichlet, 2 stationary
            time_ech = 10                      #periode where we compare the solution to the analytique solution
            x_coordinates, simulated_time, initial_solution, results, theoretical_values,error_L2 = \
                            main.run_main(scheme, D, delta_t, n_cells, x_min, \
                                          x_max, order_of_accuracy, n_time_step,initial_choice,dirichlet,time_ech)
            
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
            
            store_dt[i-4,k-1] = delta_t 

