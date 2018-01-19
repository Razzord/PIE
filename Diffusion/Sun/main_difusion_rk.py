# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 19:22:01 2017

@author: Eric Heulhard
"""
#script to compute the advection equation by the difference spectral methode



import math 
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import operator
import sys


#----------------input----------------------------------
#------------------------------------------------------------------------------

#convection speed
c = 1;
#time step parameter
delta_t = 0.6;
#order of RK scheme
or_RK = 6;
#coordinate of the cells
size_of_domaine = 100;
coord = np.zeros(size_of_domaine);
for i in range(0,size_of_domaine) :
    coord.itemset(i,i); 
#number of interior flux point -> degree of the solution deg +1
deg = 1;
#number of step calculated
n_step_time = 1000;
#initializing the matrix where the result is going to be stocked
results = np.zeros((n_step_time+1,(deg+1)*(len(coord)-1)));
    

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#execute the function file
exec(open("./function_riemann_diffusion.py").read())
exec(open("./sol_flux_point_global.py").read())
exec(open("./runge_kunta.py").read())
exec(open("./error_L2_matrix.py").read())

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#-------------------------------------------------------------------
#------------------------------------------------------------------------------

alpha = RKalpha6optim(deg);

#------------------compute the coordinates of the points-------------
#------------------------------------------------------------------------------

#compte all the matrix of extrapolation and evolutions
list_mat = compute_point_and_mat_extrapo(deg,size_of_domaine,coord);

#compute the coordinate of the flux point
flux_point = list_mat[1];

#compute the coordinate of the sol point
sol_point = list_mat[0];

#compute the matrix of extrapolation from the sol point toward the flux point
mat_extra_sol_point_flux_point = list_mat[2];

#compute the matrix of extrapolation derivative
mat_d_flux_at_sol_point = list_mat[3];

#initilization of the solution
u0 = np.zeros((size_of_domaine-1)*(deg+1));
center_gaussian = coord[math.floor(len(coord)/2)];
largeur = 100;
for i in range(0,(size_of_domaine-1)):
    for k in range(0,deg+1):
        u0.itemset(((deg+1)*i+k),np.exp(-np.power(center_gaussian -coord[i]-sol_point[k]*0.5, 2)/largeur)) ;
results[0] = u0;

#calculate CFL
space_x = np.zeros(deg+2);
space_x[0] = 2*(1+sol_point[0])
if (deg > 1):
    for k in range(0,deg):
        space_x[k+1] = abs(sol_point[k+1]-sol_point[k]);
space_x[deg+1] = 2*(1-sol_point[deg])
if (deg == 1):
    space_x[0] = 1;
CFL = c*(delta_t/(deg+2)**2);
print(CFL)
if (CFL > 0.95):
    print("CFL to high");
    #♠sys.exit()
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


#--------------------calculation of each iteration----------------
#------------------------------------------------------------------------------

#loop for every time step


for k in range(1,n_step_time):
    
    temp_new_value_at_sol_point = results[k-1];
    
    #conditions dirichlet
    temp_new_value_at_sol_point[0] = 0;
    temp_new_value_at_sol_point[(deg+1)*(size_of_domaine-1)-1] = 1;
        
    #we apply a RK optimised of order or_RK ----------------------------------
    #-------------------------------------------------------------------------

    for j in range(1,or_RK):
        
        #first step--------------------------------------------------------
        #computing the value of the flux at the inner points
        #by extrapolation of the solution
        all_flux_point_array = (c*(delta_t*2))*np.array(\
        mat_extra_sol_point_flux_point.dot(temp_new_value_at_sol_point));
            
        #Riemmann condition
        for i in range(1,size_of_domaine-1):
            flux_left = all_flux_point_array[i*(deg+2)-1] ; 
            flux_right = all_flux_point_array[i*(deg+2)] ;
            all_flux_point_array[i*(deg+2)-1] = our_riemann(flux_left,flux_right);
            all_flux_point_array[i*(deg+2)] = our_riemann(flux_left,flux_right);
        
        #computing the intermediate flux at the sol point
        intermediate_value = mat_d_flux_at_sol_point.dot(all_flux_point_array) ;
        
        #second step--------------------------------------------------------
        #extrapolate the intermidiate flux value the flux point
        all_flux_point_array = mat_extra_sol_point_flux_point.dot(intermediate_value);
        
        #periodicity        
        #all_flux_point_array[0] = all_flux_point_array[(deg+2)*(size_of_domaine-1)-1] ;
        #no loss
        #all_flux_point_array[0] = all_flux_point_array[(deg+2)*(size_of_domaine-1)-1] = 0 ;
        
        #Riemmann condition
        for i in range(1,size_of_domaine-1):
            flux_left = all_flux_point_array[i*(deg+2)-1] ; 
            flux_right = all_flux_point_array[i*(deg+2)] ;
            all_flux_point_array[i*(deg+2)-1] = our_riemann(flux_left,flux_right);
            all_flux_point_array[i*(deg+2)] = our_riemann(flux_left,flux_right); 
            
        temp_new_value_at_sol_point = operator.add(temp_new_value_at_sol_point, \
        +alpha[j]*mat_d_flux_at_sol_point.dot(all_flux_point_array)) ;                                           
                                                   
    
    
    
    #-----------computing the new value of the solutions ----------------------

    results[k] = temp_new_value_at_sol_point;

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#----------------plot the value of the solution at some given step time.----------

#which step time is going to be plot
n_time = [ 0,1,n_step_time-1];
#we increase the sampling to plot a result
n_sampling_point = 5;
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
        
#----------plot the analytique solution----------------------------------
#----------we plot the solution only at the top of the gaussian---------
analytique_results = np.zeros((len(n_time),(n_sampling_point*(size_of_domaine-1))));
#-----definition of the function under the integral considering that the
#-----intial solution is gaussian--------------------

def brownien(y,x,t):     
    a = -(x- y)*(x- y);
    a = a/(4*c*t+largeur);
    a = (1/math.sqrt((4*c*t/largeur)+1))*math.exp(a);
    return(a)
    
for i in range(0,len(n_time)):
    for j in range(0,n_sampling_point*(size_of_domaine-1)):
        analytique_results[i,j] = brownien(center_gaussian+0.5,pos_all_sampling_points[j],n_time[i]*delta_t);


plt.plot(pos_all_sampling_points,sol_all_sampling_point[0])   
plt.plot(pos_all_sampling_points,sol_all_sampling_point[2])
plt.plot(pos_all_sampling_points,analytique_results[2])
plt.legend(['t = 0','t ='+str(round(n_time[2]*delta_t,1)),'therorical solution at '+str(round(n_time[2]*delta_t,1))])


#calculate the distance between results and theory
#computing the error matrix
error_matrix = compute_error_matrix(n_sampling_point,size_of_domaine,coord);
#projecting the analytique results in the Lagrange base for each cells
vec_dif_res_th = sol_all_sampling_point[2]-analytique_results[2];

error_L2 =  np.dot(vec_dif_res_th,error_matrix.dot(vec_dif_res_th));
print(error_L2);
    



    