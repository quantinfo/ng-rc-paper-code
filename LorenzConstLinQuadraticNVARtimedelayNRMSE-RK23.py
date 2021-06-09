# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 13:17:10 2021

NVAR with time delays for Lorenz forecasting, NRMSE and fixed points.
Don't be efficient for now.

May 18: fixed nrmse calculation, error of fixed points

@author: Dan
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

##
## Parameters
##

# number of NRMSE trials
npts=10
# how far in to Lorenz solution to start
start=5.
# units of time to train for
traintime=10.
# ridge parameter for regression
ridge_param = 2.5e-6

# create a vector of warmup times to use, dividing space into
# npts segments of length traintime
warmup_v=np.arange(start,traintime*npts+start,traintime)

# storage for results
train_nrmse_v=np.zeros(npts)
test_nrmse_v=np.zeros(npts)
n_fp1_diff_v=np.zeros(npts)
n_fp2_diff_v=np.zeros(npts)
n_fp0_diff_v=np.zeros(npts)
p_fp1_norm_v=np.zeros((npts, 3))
p_fp2_norm_v=np.zeros((npts, 3))
p_fp0_norm_v=np.zeros((npts, 3))

# run a trial with the given warmup time
def find_err(warmup):
    ##
    ## More Parameters
    ##

    # time step
    dt=0.025
    # Lyapunov time of the Lorenz system
    lyaptime=1.104
    # units of time to test for
    testtime=lyaptime
    # total time to run for
    maxtime = warmup+traintime+testtime

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
    # size of nonlinear part of feature vector
    dnonlin = int(dlin*(dlin+1)/2)
    # total size of feature vector: constant + linear + nonlinear
    dtot = 1 + dlin + dnonlin

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

    # calculate mean, min, and max for all three components of Lorenz solution
    lorenz_stats=np.zeros((3,3))
    for i in range(3):
        lorenz_stats[0,i]=np.mean(lorenz_soln.y[i,warmtrain_pts:maxtime_pts])
        lorenz_stats[1,i]=np.min(lorenz_soln.y[i,warmtrain_pts:maxtime_pts])
        lorenz_stats[2,i]=np.max(lorenz_soln.y[i,warmtrain_pts:maxtime_pts])

    # total variance of the Lorenz solution
    total_var=np.var(lorenz_soln.y[0:d,:])

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
            # shift by one for constant
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
        out_test[1:dlin+1]=x_test[:,j] # shift by one for constant
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
    
    # setup variables for predicted and true fixed points
    t_fp0=np.zeros(d)
    t_fp1=np.zeros(d)
    t_fp2=np.zeros(d)
    # true fixed point 0 is 0
    # true fixed point 1 is at...
    t_fp1[0]=np.sqrt(beta*(rho-1))
    t_fp1[1]=np.sqrt(beta*(rho-1))
    t_fp1[2]=rho-1
    # true fixed point 2 is at...
    t_fp2[0]=-t_fp1[0]
    t_fp2[1]=-t_fp1[1]
    t_fp2[2]=t_fp1[2]

    # this function does a single step NVAR prediction for a trial fixed point
    # and returns the difference between the input and prediction
    # we can then solve func(p_fp) == 0 to find a fixed point p_fp
    def func(p_fp):
        # create a trial input feature vector
        out_vec=np.ones(dtot)
        # fill in the linear part
        for ii in range(k):
            # all past input is p_fp
            out_vec[1+ii*d:1+(ii+1)*d]=p_fp[0:d]
        # fill in the nonlinear part of the feature vector
        cnt=0
        for row in range(dlin):
            for column in range(row,dlin):
                out_vec[dlin+1+cnt]=out_vec[1+row]*out_vec[1+column]
                cnt += 1
        return W_out @ out_vec

    # solve for the first fixed point and calculate distances
    p_fp1 = fsolve(func, t_fp1)
    n_fp1_diff=np.sqrt(np.sum((t_fp1-p_fp1)**2)/total_var)
    p_fp1_norm = (t_fp1 - p_fp1) / np.sqrt(total_var)

    # solve for second fixed point
    p_fp2 = fsolve(func, t_fp2)
    n_fp2_diff=np.sqrt(np.sum((t_fp2-p_fp2)**2)/total_var)
    p_fp2_norm = (t_fp2 - p_fp2) / np.sqrt(total_var)

    # solve for 0 fixed point
    p_fp0=fsolve(func, t_fp0)
    n_fp0_diff=np.sqrt(np.sum((t_fp0-p_fp0)**2)/total_var)
    p_fp0_norm = (t_fp0 - p_fp0) / np.sqrt(total_var)

    # return our findings
    return train_nrmse,test_nrmse,n_fp1_diff,n_fp2_diff,n_fp0_diff,p_fp1_norm,p_fp2_norm,p_fp0_norm

# run many trials and collect the results
for i in range(npts):
    train_nrmse_v[i],test_nrmse_v[i],n_fp1_diff_v[i],n_fp2_diff_v[i],n_fp0_diff_v[i],p_fp1_norm_v[i],p_fp2_norm_v[i],p_fp0_norm_v[i]=find_err(warmup_v[i])

# output summaries
print('\n ridge regression parameter: '+str(ridge_param)+'\n')
print('mean, meanerr, train nrmse: '+str(np.mean(train_nrmse_v))+' '+str(np.std(train_nrmse_v)/np.sqrt(npts)))
print('mean, meanerr, test nrmse: '+str(np.mean(test_nrmse_v))+' '+str(np.std(test_nrmse_v)/np.sqrt(npts)))

# mean / err of (normalized L2 distance from true to predicted fixed point)
print()
print('mean, meanerr, fp1 nL2 distance: '+str(np.mean(n_fp1_diff_v))+' '+str(np.std(n_fp1_diff_v)/np.sqrt(npts)))
print('mean, meanerr, fp2 nL2 distance: '+str(np.mean(n_fp2_diff_v))+' '+str(np.std(n_fp2_diff_v)/np.sqrt(npts)))
print('mean, meanerr, fp0 nL2 distance: '+str(np.mean(n_fp0_diff_v))+' '+str(np.std(n_fp0_diff_v)/np.sqrt(npts)))

# mean / err of (normalized difference between true and predicted fixed point)
print()
print('mean, meanerr, fp1', np.mean(p_fp1_norm_v, axis=0), np.std(p_fp1_norm_v, axis=0) / np.sqrt(npts))
print('mean, meanerr, fp2', np.mean(p_fp2_norm_v, axis=0), np.std(p_fp2_norm_v, axis=0) / np.sqrt(npts))
print('mean, meanerr, fp0', np.mean(p_fp0_norm_v, axis=0), np.std(p_fp0_norm_v, axis=0) / np.sqrt(npts))

# normalized L2 distance between true and (mean of predicted fixed point)
print()
print('nL2 distance to mean, meanerr, fp1', np.sqrt(np.sum(np.mean(p_fp1_norm_v, axis=0) ** 2)), np.sqrt(np.sum(np.var(p_fp1_norm_v, axis=0)) / npts))
print('nL2 distance to mean, meanerr, fp2', np.sqrt(np.sum(np.mean(p_fp2_norm_v, axis=0) ** 2)), np.sqrt(np.sum(np.var(p_fp2_norm_v, axis=0)) / npts))
print('nL2 distance to mean, meanerr, fp0', np.sqrt(np.sum(np.mean(p_fp0_norm_v, axis=0) ** 2)), np.sqrt(np.sum(np.var(p_fp0_norm_v, axis=0)) / npts))
