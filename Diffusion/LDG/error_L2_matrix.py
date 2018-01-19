# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 22:38:26 2018

@author: Eric Heulhard
"""

#function to compute the matrix to measure the error

import numpy as np;
import math;

def compute_error_matrix(n_sampling,size_of_domaine,coord):
    
    #computing the solution points
    sol_point = np.zeros(n_sampling);
    Tche = np.polynomial.chebyshev.Chebyshev.basis(n_sampling);
    #computing the roots of the chebyshev polynomial
    sol_point = np.polynomial.chebyshev.Chebyshev.roots(Tche);
    
    #compute the local error matrix
    local_M = np.zeros((n_sampling,n_sampling));
    Id = np.identity(n_sampling);
    for i in range(0,(n_sampling)):
        Lag_ligne = sp.interpolate.lagrange(sol_point,Id[i]);
        for j in range(0,n_sampling):
            Lag_colonne = sp.interpolate.lagrange(sol_point,Id[j]);
            Lag_P = np.polyint(np.polymul(Lag_ligne,Lag_colonne));
            local_M[i,j] = Lag_P(1) - Lag_P(-1);

    #compute the global error matrix
    global_M = np.zeros(((size_of_domaine-1)*(n_sampling),(size_of_domaine-1)*(n_sampling)));
    for i in range(0,(size_of_domaine-1)*(n_sampling)):
        for j in range(0,n_sampling):
            global_M[i,(n_sampling)*math.floor(i/(n_sampling))+j] =\
            (1/(coord[math.floor(i/(n_sampling))+1]-coord[math.floor(i/(n_sampling))]))*\
            local_M[i%(n_sampling),j];
            
    return(global_M);