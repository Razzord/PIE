# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 16:02:22 2017

@author: Eric Heulhard
"""

import numpy as np



#function to compute flux in the extrem flux point (riemann)
#waiting for having the right function, this function only average two flux point
#entries : array_left,array_right

def BR2_continuity_u(flux_left_u,flux_right_u) :

    flux = (flux_left_u+flux_right_u)/2;
    return (flux)

def BR2_continuity_s(grad_left_u,grad_right_u,re_left,re_right,eta) :

    flux = (grad_left_u+grad_right_u)/2 - eta*(re_left+re_right)/2;
    return (flux)  