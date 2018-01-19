# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 11:34:13 2017

@author: Eric Heulhard
"""

import math
import numpy as np
import scipy as sp
from scipy import interpolate
from scipy.special import legendre
import numpy.polynomial.polynomial as poly

#function to compute the positions of the flux points and the sol points in an 
#parametric cell
#entries : number of inner flux points
#outputs : position of the solution points and the flux points

def compute_point_and_mat_extrapo(deg,size_of_domaine,coord) :
    
    #computing the flux_point and the sol_point in the iso cell
    #computing the legendre polynomial of degree p
    P = legendre(deg);
    #computing the inverse of the roots of the legendre polynomial
    a = poly.polyroots(P);
    #storing the roots
    if deg%2 == 0:
        flux_point = np.zeros(deg)
        for i in range(0,deg) :
            flux_point.itemset(i,1/a[i])
    else :
        flux_point = np.zeros(deg)
        
        for i in range(0,math.floor(deg*0.5)) :
            flux_point.itemset(i,1/a[i])
        flux_point.itemset(math.floor(deg*0.5),0)
        for i in range(math.floor(deg*0.5)+1,deg) :
            flux_point.itemset(i,1/a[i-1])
        
    #we add the extrem flux points
    flux_point = np.append(-1,flux_point);
    flux_point = np.append(flux_point,1);
    
    #computing the solution points
    sol_point = np.zeros(deg+1);
    Tche = np.polynomial.chebyshev.Chebyshev.basis(deg+1);
    #computing the roots of the chebyshev polynomial
    sol_point = np.polynomial.chebyshev.Chebyshev.roots(Tche);
    
    #building the extrapolation matrix sol point toward flux point
    mat_sol_point = np.zeros((deg+2,deg+1));
    Id = np.identity(deg+1);
    for j in range(0,deg+1):
        Lag = sp.interpolate.lagrange(sol_point,Id[j]);
        for i in range(0,deg+2):
            mat_sol_point[i,j] = Lag(flux_point[i]);
            
    #building the derivative matrix to compute the derivative of the flux at
    #sol point
    
    mat_d_flux_at_sol_point =np.zeros((deg+1,deg+2));
    Id = np.identity(deg+2);
    for j in range(0,deg+2):
        Lag = sp.interpolate.lagrange(flux_point,Id[j]);
        Lag = np.polyder(Lag,1);
        for i in range(0,deg+1):
            mat_d_flux_at_sol_point[i,j] = Lag(sol_point[i]);
    
    #compute the global extrapolation matrix        
    mat_global_extrapolation = np.zeros(((size_of_domaine-1)*(deg+2),(size_of_domaine-1)*(deg+1)));
    
    for i in range(0,(size_of_domaine-1)*(deg+2)):
        for j in range(0,deg+1):
            mat_global_extrapolation[i,(deg+1)*math.floor(i/(deg+2))+j] =\
            mat_sol_point[i%(deg+2),j];
     
    #compute the global extrapolation matrix        
    mat_global_d_flux_at_sol_point = np.zeros(((size_of_domaine-1)*(deg+1),(size_of_domaine-1)*(deg+2)));
            
    for i in range(0,(size_of_domaine-1)*(deg+1)):
        for j in range(0,deg+2):
            mat_global_d_flux_at_sol_point[i,(deg+2)*math.floor(i/(deg+1))+j] =\
            (1/(coord[math.floor(i/(deg+1))+1]-coord[math.floor(i/(deg+1))]))*\
            mat_d_flux_at_sol_point[i%(deg+1),j];
            
    #compute the vector to compute gradient of u at 1 and -1
    mat_grad_border = np.zeros((2,deg+2));
    Id = np.identity(deg+2);
    for j in range(0,deg+2):
        Lag = sp.interpolate.lagrange(flux_point,Id[j]);
        d_Lag = np.polyder(Lag,1)
        for i in range(0,2):
            if (i==0):
                mat_grad_border[i,j] = d_Lag(-1)
            if (i == 1):
                mat_grad_border[i,j] = d_Lag(1)
    
    #compute the global mat_grad_border matrix
    mat_global_grad_border = np.zeros((2*(size_of_domaine-1),(size_of_domaine-1)*(deg+2)))
    for i in range(0,(size_of_domaine-1)*2):
        for j in range(0,deg+2):
            mat_global_grad_border[i,(deg+2)*math.floor(i/2)+j] =\
            mat_grad_border[i%2,j];
            
    #compute the local relevment matrix
    local_R = np.zeros((deg+2,deg+2));
    Id = np.identity(deg+2);
    for i in range(0,(deg+2)):
        Lag_ligne = sp.interpolate.lagrange(flux_point,Id[i]);
        for j in range(0,deg+2):
            Lag_colonne = sp.interpolate.lagrange(flux_point,Id[j]);
            Lag_P = np.polyint(np.polymul(Lag_ligne,Lag_colonne));
            local_R[i,j] = Lag_P(1) - Lag_P(-1);

    #compute the global relevment matrix
    global_R = np.zeros(((size_of_domaine-1)*(deg+2),(size_of_domaine-1)*(deg+2)));
    for i in range(0,(size_of_domaine-1)*(deg+2)):
        for j in range(0,deg+2):
            global_R[i,(deg+2)*math.floor(i/(deg+2))+j] =\
            (1/(coord[math.floor(i/(deg+2))+1]-coord[math.floor(i/(deg+2))]))*\
            local_R[i%(deg+2),j];
    
    #compute the matrix to build the second member of the relevment equation
    loc_sec_mem_rev_mat = np.zeros((deg+2,2))
    Id = np.identity(deg+2);
    for i in range(0,deg+2):
        Lag_Ligne = sp.interpolate.lagrange(flux_point,Id[i]);
        loc_sec_mem_rev_mat[i,0] = Lag_Ligne(-1)
        loc_sec_mem_rev_mat[i,1] = -Lag_Ligne(1)
        
    #compute the global matrix to build the second member of the relevment equation
    global_sec_mem_rev_mat = np.zeros(((deg+2)*(size_of_domaine-1),2*(size_of_domaine-1)))
    for i in range(0,(deg+2)*(size_of_domaine-1)):
        for j in range(0,2):
            global_sec_mem_rev_mat[i,2*math.floor(i/(deg+2))+j] =\
            loc_sec_mem_rev_mat[i%(deg+2),j];
            
    return(sol_point,flux_point,mat_global_extrapolation,\
           mat_global_d_flux_at_sol_point,mat_global_grad_border,global_R,\
           global_sec_mem_rev_mat);