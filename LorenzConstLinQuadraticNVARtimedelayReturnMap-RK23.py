# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 13:17:10 2021

NVAR with time delays.  Don't be efficient for now.

@author: Dan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.signal
import scipy.interpolate
import matplotlib.patches

##
## Parameters
##

# how far in to Lorenz solution to start
start=5.
# units of time to train for
traintime=10.
# ridge parameter for regression
ridge_param = 2.5e-6

# run a trial with the given warmup time, and write a return map plot
# to the file in plotname
def find_err(warmup, plotname=None, lettera='a', letterb='b'):
    ##
    ## More Parameters
    ##

    # time step
    dt=0.025
    # units of time to test for
    testtime=1000.
    # total time to run for
    maxtime = warmup+traintime+testtime
    # Lyapunov time of the Lorenz system
    lyaptime=1.104

    # discrete-time versions of the times defined above
    warmup_pts=round(warmup/dt)
    traintime_pts=round(traintime/dt)
    warmtrain_pts=warmup_pts+traintime_pts
    testtime_pts=round(testtime/dt)
    maxtime_pts=round(maxtime/dt)
    lyaptime_pts=round(lyaptime/dt)


    # input dimension
    d = 3
    # number of time delay taps
    k = 2
    # size of the linear part of the feature vector
    dlin = k*d
    # size of the nonlinear part of the feature vector
    dnonlin = int(dlin*(dlin+1)/2)
    # total size of the feature vector: constant + linear + nonlinear
    dtot = 1 + dlin + dnonlin

    # t values for whole evaluation time
    # (need maxtime_pts + 1 to ensure a step of dt)
    t_eval=np.linspace(0,maxtime,maxtime_pts+1) # need the +1 here to have a step of dt

    ##
    ## Lorenz '63
    ##
    
    sigma = 10
    beta = 8 / 3
    rho = 28

    def lorenz(t, y):
      
      dy0 = sigma * (y[1] - y[0])
      dy1 = y[0] * (rho - y[2]) - y[1]
      dy2 = y[0] * y[1] - beta * y[2]
      
      # since lorenz is 3-dimensional, dy/dt should be an array of 3 values
      return [dy0, dy1, dy2]
    
    # I integrated out to t=50 to find points on the attractor, then use these as the initial conditions
    
    lorenz_soln = solve_ivp(lorenz, (0, maxtime), [17.67715816276679, 12.931379185960404, 43.91404334248268] , t_eval=t_eval, method='RK23')

    # total variance of the Lorenz solution
    total_var=np.var(lorenz_soln.y[0:d,:])

    # calculate mean, min, and max for all three components of Lorenz solution
    lorenz_stats=np.zeros((3,3))
    for i in range(3):
        lorenz_stats[0,i]=np.mean(lorenz_soln.y[i,warmtrain_pts:maxtime_pts])
        lorenz_stats[1,i]=np.min(lorenz_soln.y[i,warmtrain_pts:maxtime_pts])
        lorenz_stats[2,i]=np.max(lorenz_soln.y[i,warmtrain_pts:maxtime_pts])

    ##
    ## NVAR
    ##

    # create an array to hold the linear part of the feature vector
    x = np.zeros((dlin,maxtime_pts))

    # fill in the linear part of the feature vector for all times
    for delay in range(k):
        for j in range(delay,maxtime_pts):
            x[d*delay:d*(delay+1),j]=lorenz_soln.y[:,j-delay]

    # create an array to hold the full feature vector for training time
    # (use ones so the constant term is already 1)
    out_train = np.ones((dtot,traintime_pts))
    
    # copy over the linear part (shift over by one to account for constant)
    out_train[1:dlin+1,:]=x[:,warmup_pts-1:warmtrain_pts-1]

    # fill in the non-linear part
    cnt=0
    for row in range(dlin):
        for column in range(row,dlin):
            out_train[dlin+1+cnt]=x[row,warmup_pts-1:warmtrain_pts-1]*x[column,warmup_pts-1:warmtrain_pts-1]
            cnt += 1
    
    # ridge regression: train W_out to map out_train to Lorenz[t] - Lorenz[t - 1]
    W_out = (x[0:d,warmup_pts:warmtrain_pts]-x[0:d,warmup_pts-1:warmtrain_pts-1]) @ out_train[:,:].T @ np.linalg.pinv(out_train[:,:] @ out_train[:,:].T + ridge_param*np.identity(dtot))
    
    # apply W_out to the training feature vector to get the training output
    x_predict = x[0:d,warmup_pts-1:warmtrain_pts-1] + W_out @ out_train[:,0:traintime_pts]
    
    # calculate NRMSE between true Lorenz and training output
    train_nrmse = np.sqrt(np.mean((x[0:d,warmup_pts:warmtrain_pts]-x_predict[:,:])**2)/total_var)

    # create a place to store feature vectors for prediction
    out_test = np.ones(dtot)               # full feature vector
    x_test = np.zeros((dlin,testtime_pts)) # linear part

    # copy over initial linear feature vector
    x_test[:,0] = x[:,warmtrain_pts-1]

    # do prediction
    for j in range(testtime_pts-1):
        # copy linear part into whole feature vector
        out_test[1:dlin+1]=x_test[:,j]
        # fill in the non-linear part
        cnt=0
        for row in range(dlin):
            for column in range(row,dlin):
                # shift by one for constant
                out_test[dlin+1+cnt]=x_test[row,j]*x_test[column,j]
                cnt += 1
        # fill in the delay taps of the next state
        x_test[d:dlin,j+1] = x_test[0:(dlin-d),j]
        # do a prediction
        x_test[0:d,j+1] = x_test[0:d,j] + W_out @ out_test[:]
    

    # calculate NRMSE between true Lorenz and prediction for one Lyapunov time
    test_nrmse = np.sqrt(np.mean((x[0:d,warmtrain_pts-1:warmtrain_pts+lyaptime_pts-1]-x_test[0:d,0:lyaptime_pts])**2)/total_var)

    # if requested, make a return plot
    if plotname:
        # get predicted return map
        rm = return_map_spline(x_test[2, :])
        # get true return map
        rm_cmp = return_map_spline(lorenz_soln.y[2,:testtime_pts])

        # plot
        fig, (ax1, ax2) = plt.subplots(1, 2, dpi=200, figsize=(6, 3))

        # whole return map
        ax1.scatter(rm_cmp[:, 0], rm_cmp[:, 1], marker='P', s=2, label='Lorenz63', color='blue', linewidths=0)
        ax1.scatter(rm[:, 0], rm[:, 1], marker='X', s=2, label='NG-RC', color='red', linewidths=0)
        ax1.set_xlim(30, 48)
        ax1.set_ylim(30, 48)
        ax1.set_xlabel('$M_i$')
        ax1.set_ylabel('$M_{i+1}$')

        # zoomed return map
        ax2.scatter(rm_cmp[:, 0], rm_cmp[:, 1], marker='P', s=5, label='Lorenz63', color='blue', linewidths=0)
        ax2.scatter(rm[:, 0], rm[:, 1], marker='X', s=5, label='NG-RC', color='red', linewidths=0)
        xlim2 = (34.6, 35.5)
        ylim2 = (35.7, 36.6)
        ax2.set_xlim(*xlim2)
        ax2.set_ylim(*ylim2)
        ax2.set_xlabel('$M_i$')
        ax2.set_ylabel('$M_{i+1}$')

        # draw the zoomed rectangle on the whole
        rect = matplotlib.patches.Rectangle((xlim2[0], ylim2[0]), xlim2[1] - xlim2[0], ylim2[1] - ylim2[0], linewidth=1, edgecolor='k', facecolor='none')
        ax1.add_patch(rect)

        # subplot labels
        ax1.text(-0.1, 1.05, lettera + ')', transform=ax1.transAxes, fontsize=10, va='top', ha='right')
        ax2.text(-0.25, 1.05, letterb + ')', transform=ax2.transAxes, fontsize=10, va='top', ha='right')

        # write out
        plt.tight_layout()
        plt.savefig(plotname, dpi=600)    

# use interpolating splines to find maxima of input signal, and return an array
# of (M_i, M_i+1) pairs
def return_map_spline(v):
    spline = scipy.interpolate.InterpolatedUnivariateSpline(np.arange(len(v)), v, k=4)
    spline_d = spline.derivative()
    spline_dd = spline_d.derivative()

    # when is the derivative of v zero?
    extimes = spline_d.roots()

    # discard times out of bound
    extimes = extimes[extimes > 0]
    extimes = extimes[extimes < len(v) - 1]

    # select only local maxima
    extimes = extimes[spline_dd(extimes) < 0]

    # find values
    ex = spline(extimes)

    # construct return map
    return np.stack([ex[:-1], ex[1:]], axis=-1)

find_err(start, plotname='lorenz-rmap.png')
find_err(start, plotname='lorenz-rmap.svg')
find_err(start, plotname='lorenz-rmap.eps')
find_err(start, plotname='lorenz-rmap.pdf')
