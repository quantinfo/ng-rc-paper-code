# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 13:17:10 2021

NVAR with time delays for Lorenz prediction, NRMSE.
Don't be efficient for now.

Measure x,y, predict z

@author: Dan
"""

import numpy as np
from scipy.integrate import solve_ivp
import timeit

##
## Parameters
##

# number of trials to run for NRMSE calculation
npts=10
# how far in to Lorenz solution to start the NVAR trials
start=5.
# how far apart the warmup intervals should be for each trial
interval=20.

# calculate warmup times for each trial
warmup_v=np.arange(start,interval*npts+start,interval)

# storage for trial results
train_nrmse_v=np.zeros(npts)
test_nrmse_v=np.zeros(npts)
run_time_v=np.zeros(npts)

# ridge parameter for regression
ridge_param = .05

# run an NVAR trial and return results, for the given warmup time
def find_err(warmup):
    ##
    ## Parameters
    ##

    # time step
    dt=0.05
    # units of time to train for
    traintime = 20.
    # units of time to test for
    testtime=45.
    # total time to run for
    maxtime = warmup+traintime+testtime

    # discrete-time versions of the times defined above
    warmup_pts=round(warmup/dt)
    traintime_pts=round(traintime/dt)
    warmtrain_pts=warmup_pts+traintime_pts
    maxtime_pts=round(maxtime/dt)

    # input dimension
    d = 3
    # number of time delay taps
    k = 4
    # number of time steps between taps, skip = 1 means take consecutive points
    skip = 5
    # size of linear part of feature vector (leave out z)
    dlin = k*(d-1)
    # size of nonlinear part of feature vector
    dnonlin = int(dlin*(dlin+1)/2)
    # total size of feature vector: constant + linear + nonlinear
    dtot = 1+dlin + dnonlin

    # t values for whole evaluation time
    # (need maxtime_pts + 1 to ensure a step of dt)
    t_eval=np.linspace(0,maxtime,maxtime_pts+1)

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

    # calculate standard deviation of z component
    zstd = np.std(lorenz_soln.y[2,:])

    ##
    ## NVAR
    ##

    # create an array to hold the linear part of the feature vector
    x = np.zeros((dlin,maxtime_pts))

    # create an array to hold the full feature vector for all time after warmup
    # (use ones so the constant term is already 1)
    out = np.ones((dtot,maxtime_pts-warmup_pts))

    # record start time
    stime = timeit.default_timer()

    # fill in the linear part of the feature vector for all times
    for delay in range(k):
        for j in range(delay,maxtime_pts):
            # only include x and y
            x[(d-1)*delay:(d-1)*(delay+1),j]=lorenz_soln.y[0:2,j-delay*skip]

    # copy over the linear part (shift over by one to account for constant)
    # unlike forecasting, we can do this all in one shot, and we don't need to
    # shift times for one-step-ahead prediction
    out[1:dlin+1,:]=x[:,warmup_pts:maxtime_pts]

    # fill in the non-linear part
    cnt=0
    for row in range(dlin):
        for column in range(row,dlin):
            # shift by one for constant
            out[dlin+1+cnt,:]=x[row,warmup_pts:maxtime_pts]*x[column,warmup_pts:maxtime_pts]
            cnt += 1

    # ridge regression: train W_out to map out to Lorenz z
    W_out = lorenz_soln.y[2,warmup_pts:warmtrain_pts] @ out[:,0:traintime_pts].T @ np.linalg.pinv(out[:,0:traintime_pts] @ out[:,0:traintime_pts].T + ridge_param*np.identity(dtot))

    # record end time, and total time
    etime = timeit.default_timer()
    run_time=etime-stime

    # once we have W_out, we can predict the entire shot
    # apply W_out to the feature vector to get the output
    # this includes both training and testing phases
    z_predict = W_out @ out[:,:]
    
    # calculate NRMSE between true Lorenz z and training output
    train_nrmse = np.sqrt(np.mean((lorenz_soln.y[2,warmup_pts:warmtrain_pts]-z_predict[0:traintime_pts])**2))/zstd

    # calculate NRMSE between true Lorenz z and prediction
    test_nrmse = np.sqrt(np.mean((lorenz_soln.y[2,warmtrain_pts:maxtime_pts]-z_predict[traintime_pts:maxtime_pts-warmup_pts])**2))/zstd

    return train_nrmse,test_nrmse,run_time

# run the trials and store the results
for i in range(npts):
    train_nrmse_v[i],test_nrmse_v[i],run_time_v[i]=find_err(warmup_v[i])

# print a summary
print('\n ridge regression parameter: '+str(ridge_param)+'\n')
print('mean, meanerr, train nrmse: '+str(np.mean(train_nrmse_v))+' '+str(np.std(train_nrmse_v)/np.sqrt(npts)))
print('mean, meanerr, test nrmse: '+str(np.mean(test_nrmse_v))+' '+str(np.std(test_nrmse_v)/np.sqrt(npts)))
print('mean, meanerr, run time: '+str(np.mean(run_time_v))+' '+str(np.std(run_time_v)/np.sqrt(npts)))
