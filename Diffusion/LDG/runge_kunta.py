# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 14:20:15 2017

@author: Eric Heulhard
"""

def RKalpha6optim(p):
    ''' Runge-Kutta coefficients for time integration optimized for order 6'''
    alpha = np.zeros(6)
    alpha[2]=0.24662360430959
    alpha[3]=0.33183954253762
    alpha[4]=0.5
    alpha[5]=1.0
    if (p==2):
            alpha[0]=0.05114987425612
            alpha[1]=0.13834878188543
    if (p==3):
            alpha[0]=0.07868681448952
            alpha[1]=0.12948018884941
    if (p==4):
            alpha[0]=0.06377275785911
            alpha[1]=0.15384606858263
    if (p==5):
            alpha[0]=0.06964990321063
            alpha[1]=0.13259436863348
    if (p==6):
            alpha[0]=0.06809977676724
            alpha[1]=0.15779153065865
    if (p==7):
            alpha[0]=0.06961281995158
            alpha[1]=0.14018408222804
    if (p==8):
            alpha[0]=0.07150767268798
            alpha[1]=0.16219675431011
    if (p==9):
            alpha[0]= 0.06599710352324
            alpha[1]=0.13834850670675
    if (p==10):
            alpha[0]=0.07268810031422
            alpha[1]=0.16368178688643
    return alpha