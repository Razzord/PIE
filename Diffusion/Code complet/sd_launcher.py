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
scheme = 5                         #version 1 sun, version 2 IP, version 3 LDG, version 4 BR2, version 5 Gassner
D = 1                              #diffusion coefficient
delta_t = 0.1                      #time step parameter
n_cells = 198                       #number of cells in the mesh
x_min = 0                          #left coordinate of the physical domain
x_max = 200                         #right coordinate of the physical domain
order_of_accuracy = 3              #order of accuracy of the solution
                                   # (number of solution points = order_of_accuracy)
n_time_step = 100                   #number of steps calculated
dirichlet = 1;                     #dirichlet = 0, no dirichlet condition, 1 dirichlet condition (1 at right border)
order_of_analytique_solution = 5;  #precion of the analytique solution (number of sol_point in the cell
                                    # where the analytique solution is calculated)
initial_choice = 0                 #initialization 0 gaussian, 1 dirichlet, 2 stationary
time_ech = 10                      #periode where we compare the solution to the analytique solution,
                                  #2*time_ech < n_time_step pour que le calcul tourne
                                  #if time_ech < 0, no sampling and no error available (faster computation)
n_sampling_point = 20              #number of sampling point to represent the solution
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


error_L2 = error_L2*time_ech*delta_t

###############################################################################
#                     PLOT THE SOLUTION AT A GIVEN TIME STEP                  #
###############################################################################
plt.plot(x_coordinates, initial_solution,'.')
plt.plot(x_coordinates, results,'.')
if (initial_choice == 0 ):
    plt.plot(x_coordinates, theoretical_values,'.')
    plt.legend(['t = 0', \
                't =' + str(round(simulated_time, 1)), \
                'theoretical solution at ' + str(round(simulated_time, 1))])
    plt.title(schema+', time step = '+str(delta_t)+', degree ='+str(order_of_accuracy)+', error L2 = '+str(error_L2))
else :
    plt.legend(['t = 0', \
                't =' + str(round(simulated_time, 1))])
    plt.title(schema+', time step = '+str(delta_t)+', degree ='+str(order_of_accuracy))
  
print(error_L2)

