# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 16:02:22 2017

@author: Eric Heulhard
"""



#function to compute flux in the extrem flux point (riemann)
#entries : flux_left,flux_right,convection speed

def our_riemann(flux_left,flux_right,convection_speed) :

    if (convection_speed <= 0) :
        flux = flux_right ;
    if (convection_speed > 0) :
        flux = flux_left ;
    return (flux)

  