# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 16:02:22 2017

@author: Eric Heulhard
"""

import numpy as np



#function to compute flux in the extrem flux point (riemann)
#waiting for having the right function, this function only average two flux point
#entries : array_left,array_right

def our_riemann(flux_left,flux_right) :

    flux = (flux_left+flux_right)/2;
    return (flux)

  