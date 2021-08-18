# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 13:17:10 2021

NVAR with time delays for Lorenz prediction.  Don't be efficient for now.

Measure x,y, predict z

@author: Dan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

##
## Parameters
##

# time step
dt=0.05
# units of time to warm up NVAR, need to have warmup_pts >= 1
warmup = 5.
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
testtime_pts=round(testtime/dt)
maxtime_pts=round(maxtime/dt)

# input dimension
d = 3
# number of time delay taps
k = 4
# number of time steps between taps. skip = 1 means take consecutive points
skip = 5
# size of linear part of feature vector (leave out z)
dlin = k*(d-1)
# size of nonlinear part of feature vector
dnonlin = int(dlin*(dlin+1)/2)
# total size of feature vector: linear + nonlinear
dtot = dlin + dnonlin

# ridge parameter for regression
ridge_param = .05

# make sure we have enough warmup time
assert k*skip <= warmup_pts

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

lorenz_soln = solve_ivp(lorenz, (0, maxtime), [17.67715816276679, 12.931379185960404, 43.91404334248268] , t_eval=t_eval, method='RK45')

# calculate standard deviation of z component
zstd = np.std(lorenz_soln.y[2,:])

##
## NVAR
##

# create an array to hold the linear part of the feature vector
x = np.zeros((dlin,maxtime_pts))

# create an array to hold the full feature vector for all time after warmup
# (use ones so the constant term is already 1)
out = np.ones((dtot+1,maxtime_pts-warmup_pts))

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
W_out = lorenz_soln.y[2,warmup_pts:warmtrain_pts] @ out[:,0:traintime_pts].T @ np.linalg.pinv(out[:,0:traintime_pts] @ out[:,0:traintime_pts].T + ridge_param*np.identity(dtot+1))

# once we have W_out, we can predict the entire shot
# apply W_out to the feature vector to get the output
# this includes both training and testing phases
z_predict = W_out @ out[:,:]


# calculate NRMSE between true Lorenz z and training output
rms = np.sqrt(np.mean((lorenz_soln.y[2,warmup_pts:warmtrain_pts]-z_predict[0:traintime_pts])**2))
print('training rms: '+str(rms))
print('training nrms: '+str(rms/zstd))

# calculate NRMSE between true Lorenz z and prediction
rms = np.sqrt(np.mean((lorenz_soln.y[2,warmtrain_pts:maxtime_pts]-z_predict[traintime_pts:maxtime_pts-warmup_pts])**2))
print('testing rms: '+str(rms))
print('testing nrms: '+str(rms/zstd))

##
## Plot
##
    
t_linewidth=.8

fig1 = plt.figure()
fig1.set_figheight(8)
fig1.set_figwidth(12)

h=240
w=2

# top left of grid is 0,0
axs1 = plt.subplot2grid(shape=(h,w), loc=(0, 0), colspan=1, rowspan=30) 
axs2 = plt.subplot2grid(shape=(h,w), loc=(36, 0), colspan=1, rowspan=30)
axs3 = plt.subplot2grid(shape=(h,w), loc=(72, 0), colspan=1, rowspan=30)
axs4 = plt.subplot2grid(shape=(h,w), loc=(132, 0), colspan=1, rowspan=30)
axs5 = plt.subplot2grid(shape=(h,w), loc=(168, 0), colspan=1, rowspan=30)
axs6 = plt.subplot2grid(shape=(h,w), loc=(204, 0),colspan=1, rowspan=30)


# training phase x
axs1.set_title('training phase')
axs1.plot(t_eval[warmup_pts:warmtrain_pts]-warmup,x[0,warmup_pts:warmtrain_pts],color='b',linewidth=t_linewidth)
axs1.set_ylabel('x')
axs1.axes.xaxis.set_ticklabels([])
axs1.axes.set_xbound(-.08,traintime+.05)
axs1.axes.set_ybound(-21.,21.)
axs1.text(-.14,.9,'a)', ha='left', va='bottom',transform=axs1.transAxes)

# training phase y
axs2.plot(t_eval[warmup_pts:warmtrain_pts]-warmup,x[1,warmup_pts:warmtrain_pts],color='b',linewidth=t_linewidth)
axs2.set_ylabel('y')
axs2.axes.xaxis.set_ticklabels([])
axs2.axes.set_xbound(-.08,traintime+.05)
axs2.axes.set_ybound(-26.,26.)
axs2.text(-.14,.9,'b)', ha='left', va='bottom',transform=axs2.transAxes)

# training phase z
axs3.plot(t_eval[warmup_pts:warmtrain_pts]-warmup,lorenz_soln.y[2,warmup_pts:warmtrain_pts],color='b',linewidth=t_linewidth)
axs3.plot(t_eval[warmup_pts:warmtrain_pts]-warmup,z_predict[0:traintime_pts],color='r',linewidth=t_linewidth)
axs3.set_ylabel('z')
axs3.set_xlabel('time')
axs3.axes.set_xbound(-.08,traintime+.05)
axs3.axes.set_ybound(3.,48.)
axs3.text(-.14,.9,'c)', ha='left', va='bottom',transform=axs3.transAxes)

# testing phase x
axs4.set_title('testing phase')
axs4.plot(t_eval[warmtrain_pts:maxtime_pts]-warmup,x[0,warmtrain_pts:maxtime_pts],color='b',linewidth=t_linewidth)
axs4.set_ylabel('x')
axs4.axes.xaxis.set_ticklabels([])
axs4.axes.set_ybound(-21.,21.)
axs4.axes.set_xbound(traintime-.5,maxtime-warmup+.5)
axs4.text(-.14,.9,'d)', ha='left', va='bottom',transform=axs4.transAxes)

# testing phase y
axs5.plot(t_eval[warmtrain_pts:maxtime_pts]-warmup,x[1,warmtrain_pts:maxtime_pts],color='b',linewidth=t_linewidth)
axs5.set_ylabel('y')
axs5.axes.xaxis.set_ticklabels([])
axs5.axes.set_ybound(-26.,26.)
axs5.axes.set_xbound(traintime-.5,maxtime-warmup+.5)
axs5.text(-.14,.9,'e)', ha='left', va='bottom',transform=axs5.transAxes)

# testing phose z
axs6.plot(t_eval[warmtrain_pts:maxtime_pts]-warmup,lorenz_soln.y[2,warmtrain_pts:maxtime_pts],color='b',linewidth=t_linewidth)
axs6.plot(t_eval[warmtrain_pts:maxtime_pts]-warmup,z_predict[traintime_pts:maxtime_pts-warmup_pts],color='r',linewidth=t_linewidth)
axs6.set_ylabel('z')
axs6.set_xlabel('time')
axs6.axes.set_ybound(3.,48.)
axs6.axes.set_xbound(traintime-.5,maxtime-warmup+.5)
axs6.text(-.14,.9,'f)', ha='left', va='bottom',transform=axs6.transAxes)

plt.savefig('infer-lorenz.png')
plt.savefig('infer-lorenz.svg')
plt.savefig('infer-lorenz.eps')
plt.savefig('infer-lorenz.pdf')
