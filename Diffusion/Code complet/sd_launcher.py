# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 16:18:17 2018

@author: Jeremy
"""

import main_diffusion_rk as main
import matplotlib.pyplot as plt

###############################################################################
#                                   INPUTS                                    #
###############################################################################
scheme = 1                         #version 1 sun, version 2 IP, version 3 LDG, version 4 BR2, version 5 Gassner
D = 1                              #diffusion coefficient
delta_t = 0.1                      #time step parameter
n_cells = 99                       #number of cells in the mesh
x_min = 0                          #left coordinate of the physical domain
x_max = 100                        #right coordinate of the physical domain
order_of_accuracy = 2              #order of accuracy of the solution
                                   # (number of solution points = order_of_accuracy)
n_time_step = 100                   #number of steps calculated

x_coordinates, simulated_time, initial_solution, results, theoretical_values = \
                main.run_main(scheme, D, delta_t, n_cells, x_min, x_max, order_of_accuracy, n_time_step)

###############################################################################
#                     PLOT THE SOLUTION AT A GIVEN TIME STEP                  #
###############################################################################
plt.plot(x_coordinates, initial_solution)
plt.plot(x_coordinates, results)
plt.plot(x_coordinates, theoretical_values)
plt.legend(['t = 0', \
            't =' + str(round(simulated_time, 1)), \
            'therorical solution at ' + str(round(simulated_time, 1))])
