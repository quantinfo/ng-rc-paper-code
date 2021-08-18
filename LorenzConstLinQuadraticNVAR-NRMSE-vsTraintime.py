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
import matplotlib.pyplot as plt

##
## Parameters
##

# number of NRMSE trials
npts=20
npts_train=21
# how far in to Lorenz solution to start
start=5.
start_train=4.
end_train=24.
step_train=(end_train-start_train)/(npts_train-1)

# time step
dt=0.025
    
# ridge parameter for regression
ridge_param = 2.5e-6

# create a vector of warmup and train times to use, dividing space into
# npts segments of length traintime
traintime_v=np.arange(start_train,end_train+step_train,step_train)

warmup_v=np.empty((npts_train,npts))
for i in range(npts_train):
    warmup_v[i,:]=np.arange(start,traintime_v[i]*npts+start,traintime_v[i])
    
testNRMSE=np.empty(npts_train)
testNRMSEerr=np.empty(npts_train)

# storage for results
test_nrmse_v=np.zeros(npts)

# run a trial with the given warmup time
def find_err(warmup,traintime):
    ##
    ## More Parameters
    ##

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

    # total variance of the Lorenz solution, corrected July 15, 2021, DJG
    total_var=np.var(lorenz_soln.y[0,:])+np.var(lorenz_soln.y[1,:])+np.var(lorenz_soln.y[2,:])

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
    
    # return our findings
    return test_nrmse 

print('ridge regression parameter: '+str(ridge_param)+'\n')
# run many trials and collect the results
for j in range(npts_train):
    for i in range(npts):
        test_nrmse_v[i]=find_err(warmup_v[j,i],traintime_v[j])
        
    testNRMSE[j]=np.mean(test_nrmse_v)
    testNRMSEerr[j]=np.std(test_nrmse_v)/np.sqrt(npts)
    # output summaries
   
    print('test nrmse for traintime = '+str(traintime_v[j])+' mean, meanerr: '+str(testNRMSE[j])+' '+str(testNRMSEerr[j]))

plt.figure(figsize=(3.5,2.2))
plt.errorbar(traintime_v/dt,testNRMSE,yerr=testNRMSEerr)
plt.xlabel('training data set size')
plt.xlim(100.,1000.)
plt.xticks([200,400,600,800,1000])
plt.ylabel('NRMSE')
plt.ylim(0.,0.032)
plt.savefig('NRMSEvsTrainingPoints.png',bbox_inches="tight")
plt.savefig('NRMSEvsTrainingPoints.svg',bbox_inches="tight")
plt.savefig('NRMSEvsTrainingPoints.eps',bbox_inches="tight")
plt.savefig('NRMSEvsTrainingPoints.pdf',bbox_inches="tight")
plt.show()
