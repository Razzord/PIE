# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 15:05:09 2017

@author: Eric Heulhard
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 19:22:01 2017

@author: Eric Heulhard
"""
#script to compute the advection equation by the difference spectral methode


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#execute the function file
exec(open("./function_riemann_advection.py").read())
exec(open("./sol_flux_point.py").read())

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import math
import numpy as np
import scipy as sp
from scipy import interpolate
from scipy.special import legendre
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt
import operator


#----------------input----------------------------------
#------------------------------------------------------------------------------

#convection speed
c = 1;
#time step parameter
delta_t = 0.05;
#coordinate of the cells
size_of_domaine = 30;
coord = np.zeros(size_of_domaine);
for i in range(0,size_of_domaine) :
    coord.itemset(i,i); 
#number of interior flux point -> degree of the solution deg +1
deg = 2;
#initilization of the solution
u0 = np.zeros((len(coord)-1)*(deg+1));
for i in range(0,(size_of_domaine-1)*(deg+1)):
    u0.itemset(i,np.exp(-np.power(math.floor((deg+1)*(size_of_domaine-1)/2) -i, 2)/100)) ;
#number of step calculated
n_step_time = 200;
#initializing the matrix where the result is going to be stocked
results = np.zeros((n_step_time+1,(deg+1)*(len(coord)-1)));
results[0] = u0;

#-------------------------------------------------------------------
#------------------------------------------------------------------------------


#------------------compute the coordinates of the points-------------
#------------------------------------------------------------------------------


#compute the coordinate of the flux point
flux_point = compute_point_and_mat_extrapo(deg)[1];

#compute the coordinate of the sol point
sol_point = compute_point_and_mat_extrapo(deg)[0];

#compute the matrix of extrapolation from the sol point toward the flux point
mat_extra_sol_point_flux_point = compute_point_and_mat_extrapo(deg)[2];

#compute the matrix of extrapolation derivative
mat_d_flux_at_sol_point = compute_point_and_mat_extrapo(deg)[3];


#--------------------calculation of each iteration----------------
#------------------------------------------------------------------------------

#loop for every time step


for k in range(1,n_step_time):
    

    #calculating the value of the inner flux point in each cell ---------------

    #initializing the array to stock the value of the flux at the flux point

    all_flux_point_array = np.zeros((deg+2)*(size_of_domaine-1));
    
    #loop on each cell to compute the flux at the inner flux points
    for i in range(0,size_of_domaine-1) :
        #stock the value of the sol at the point sol at t
        all_flux_point_array[i*(deg+2):(i+1)*(deg+2)] \
        = c*mat_extra_sol_point_flux_point.dot(results[k-1][i*(deg+1):(i+1)*(deg+1)]);
        
    #Riemmann condition
    for i in range(0,size_of_domaine-1):
        flux_left = all_flux_point_array[i*(deg+2)-1] ; 
        flux_right = all_flux_point_array[i*(deg+2)] ;
        all_flux_point_array[i*(deg+2)-1] = our_riemann(flux_left,flux_right,c);
        all_flux_point_array[i*(deg+2)] = our_riemann(flux_left,flux_right,c);
        
    #--------------------------------------------------------------------------

    #-----------computing the new value of the solutions ----------------------

    #loop on each cell to compute the value of the new solution at the sol point
    #stock these value on a temporary line
    temp_new_value_at_sol_point = np.zeros((deg+1)*(len(coord)-1));
    
    for i in range(0,len(coord)-1):
        vec = mat_d_flux_at_sol_point.dot(all_flux_point_array[i*(deg+2):(i+1)*(deg+2)]);
        a = (delta_t*2/(coord[i+1]-coord[i]));
        l = a * np.array(vec);
        temp_new_value_at_sol_point[i*(deg+1):(i+1)*(deg+1)] \
        = operator.add(results[k-1][i*(deg+1):(i+1)*(deg+1)],-l)  ;
    
        #replace the kieme line in matrix result    
        results[k][i*(deg+1):(i+1)*(deg+1)] = temp_new_value_at_sol_point[i*(deg+1):(i+1)*(deg+1)];
    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#----------------plot the value of the solution at some given step time.----------

#which step time is going to be plot
n_time = [ 0,n_step_time-1];
#we increase the sampling to plot a result
n_sampling_point = 10;
#compute the position of the sampling point in the iso cell
pos_sampling_point = np.zeros(n_sampling_point)
for i in range(0,n_sampling_point):
    pos_sampling_point.itemset(i,(sol_point.item(0)+2*i/n_sampling_point));
#compute the position of the sampling points 
pos_all_sampling_points = np.zeros(n_sampling_point*(len(coord)-1))
#for each time asked we compute a higher sampling
#we store the results in a matrix
sol_all_sampling_point = np.zeros((len(n_time),n_sampling_point*(len(coord)-1)));
for k in range(0,len(n_time)):
    #for each cells get the value the solution at each sampling point
    for i in range(0,(len(coord)-1)):
        #creating the lagrange polynomial from the solution point
        #getting the value at each sol point
        sol_sol_point = np.zeros(deg+1);
        for j in range(0,deg+1):
            sol_sol_point.itemset(j,results[n_time[k],(deg+1)*i+j]);
            
        #LagrangeP will be the function sol based on the value of the solution at 
        #the solution point.
        LagrangeP = sp.interpolate.lagrange(sol_point,sol_sol_point);
        #storing the value of the solution at the sampling point
        for j in range(0,n_sampling_point):
            sol_all_sampling_point[k,i*n_sampling_point+j] = LagrangeP(pos_sampling_point[j]);
            pos_all_sampling_points[i*n_sampling_point+j] = coord[i]+(coord[i+1] - coord[i])*0.5*(1+pos_sampling_point[j]);
        
for k in range(0,len(n_time)):        
    plt.plot(pos_all_sampling_points,sol_all_sampling_point[k])
    plt.plot(pos_all_sampling_points,sol_all_sampling_point[k]) 
    
    
    
    
    
    