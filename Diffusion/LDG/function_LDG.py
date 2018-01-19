# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 16:02:22 2017

@author: Eric Heulhard
"""

import numpy as np



#function to compute flux in the extrem flux point (riemann)
#waiting for having the right function, this function only average two flux point
#entries : array_left,array_right

def LDG_continuity_u(flux_left,flux_right,gamma) :

    flux = (flux_left+flux_right)/2 + gamma*(flux_left-flux_right);
    return (flux)

def LDG_continuity_s(flux_left_s,flux_right_s,flux_left_u,flux_right_u,eta,beta) :

    flux = (flux_left_s+flux_right_s)/2 -eta*(flux_left_u-flux_left_u)+beta*(flux_left_s-flux_right_s);
    return (flux)  