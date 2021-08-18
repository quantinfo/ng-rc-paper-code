# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 13:17:10 2021

NVAR with time delays for Lorenz prediction, W_out plots.
Don't be efficient for now.

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

# choose cte = 1 to include constant term, cte = 0 to exclude it
# (this only affects the plots, not the NVAR code!)
cte = 1

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

##
## Plot
##

y_pos=np.arange(dtot+1)
if cte == 1:
    labels = ['c','x(t)','y(t)','x(t-5dt)','y(t-5dt)','x(t-10dt)','y(t-10dt)','x(t-15dt)','y(t-15dt)']
else:
    labels = ['x(t)','y(t)','x(t-5dt)','y(t-5dt)','x(t-10dt)','y(t-10dt)','x(t-15dt)','y(t-15dt)']
labels += ['x(t)x(t)','x(t)y(t)','x(t)x(t-5dt)','x(t)y(t-5dt)','x(t)x(t-10dt)','x(t)y(t-10dt)','x(t)x(t-15dt)','x(t)y(t-15dt)']
labels += ['y(t)y(t)','y(t)x(t-5dt)','y(t)y(t-5dt)','y(t)x(t-10dt)','y(t)y(t-10dt)','y(t)x(t-15dt)','y(t)y(t-15dt)']
labels += ['x(t-5dt)x(t-5dt)','x(t-5dt)y(t-5dt)','x(t-5dt)x(t-10dt)','x(t-5dt)y(t-10dt)','x(t-5dt)x(t-15dt)','x(t-5dt)y(t-15dt)']
labels += ['y(t-5dt)y(t-5dt)','y(t-5dt)x(t-10dt)','y(t-5dt)y(t-10dt)','y(t-5dt)x(t-15dt)','y(t-5dt)y(t-15dt)']
labels += ['x(t-10dt)x(t-10dt)','x(t-10dt)y(t-10dt)','x(t-10dt)x(t-15dt)','x(t-10dt)y(t-15dt)']
labels += ['y(t-10dt)y(t-10dt)','y(t-10dt)x(t-15dt)','y(t-10dt)y(t-15dt)']
labels += ['x(t-15dt)x(t-15dt)','x(t-15dt)y(t-15dt)']
labels += ['y(t-15dt)y(t-15dt)']

    
# the default figure size is 6.4" high and 4.8" wide
# https://matplotlib.org/2.0.2/users/mathtext.html

fig1a, axs1a = plt.subplots(1,2)
fig1a.set_figheight(8.5)
fig1a.set_figwidth(6.)


if cte == 1:
    axs1a[0].barh(y_pos,W_out[:],color='b')
else:
    axs1a[0].barh(y_pos,W_out[1:],color='b')
axs1a[0].set_yticks(y_pos)
axs1a[0].set_yticklabels(labels)
axs1a[0].set_ylim(44.5,-.5)
#axs1a.set_xlim(-.11,.22)
axs1a[0].set_xlabel('$[W_{out}]_z$')
axs1a[0].grid()

##### zoom in ####

if cte == 1:
    axs1a[1].barh(y_pos,W_out[:],color='b')
else:
    axs1a[1].barh(y_pos,W_out[1:],color='b')
axs1a[1].set_yticks(y_pos)
axs1a[1].set_yticklabels([])
axs1a[1].set_ylim(44.5,-.5)
axs1a[1].set_xlim(-.25,.25)
axs1a[1].set_xlabel('$[W_{out}]_z$')
axs1a[1].grid()

plt.savefig('infer-lorenz-wout.png', bbox_inches='tight')
plt.savefig('infer-lorenz-wout.svg', bbox_inches='tight')
plt.savefig('infer-lorenz-wout.eps', bbox_inches='tight')
plt.savefig('infer-lorenz-wout.pdf', bbox_inches='tight')
