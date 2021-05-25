# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 13:17:10 2021

NVAR with time delays.  Don't be efficient for now.

Measure x,y, predict z

we don't need to do one-step ahead prediction, so we can calculate the output vector
in a single shot for both training and testing because we feed it x and y ground
truth always.  We just use the data during the
testing phase to determine W_out

for timing
https://pythonhow.com/measure-execution-time-python-code/

@author: Dan

## I just modified a few lines to include the constant term in the feature vector and in the Wout plot - Wendson.
## Also, now we are using RK23 for the Lorenz system integration

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
#from numba import jit
import timeit

dt=0.05
warmup = 5.  # need to have warmup_pts >=1
traintime = 20.
testtime=45.
maxtime = warmup+traintime+testtime

warmup_pts=round(warmup/dt)
traintime_pts=round(traintime/dt)
warmtrain_pts=warmup_pts+traintime_pts
testtime_pts=round(testtime/dt)
maxtime_pts=round(maxtime/dt)

# I added "cte" to include the constant term in the features vector - Wendson
# choose cte = 1 to include the constant term in the features vector or cte = 0 otherwise. 
cte = 1

d = 3 # input_dimension = 3
k = 4 # number of time delay daps
skip = 5 # skip = 1 means take consecutive points
#dlin = k*d  # size of linear part of vector X
dlin = k*(d-1)  # leave out z
dnonlin = int(dlin*(dlin+1)/2)  # size of nonlinear part of outvector
dtot = dlin + dnonlin # size of total outvector

ridge_param = .05

if (k*skip > warmup_pts):
    wait = print('\n need longer training time - kill program and fix')

t_eval=np.linspace(0,maxtime,maxtime_pts+1) # need the +1 here to have a step of dt

# Lorenz '63

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

zstd = np.std(lorenz_soln.y[2,:])
#print('z standard deviation '+str(zstd))

# define arrays outside of function
x = np.zeros((dlin,maxtime_pts)) 
out = np.ones((dtot+1,maxtime_pts-warmup_pts))  # intiialize to ones, which will take care of the constant vector, add +1 for constant
W_out = np.zeros(dtot)
 
#@jit


def train():
    
    for delay in range(k):
        for j in range(delay,maxtime_pts):
            x[(d-1)*delay:(d-1)*(delay+1),j]=lorenz_soln.y[0:2,j-delay*skip]   # make this specific to this problem - only incudes 0 and 1
                                                                    # skip allows skipping steps in the embedding

# can generate the entire out state vector based on x and y at one shot over the entire train and testing time
# because this only involves x and y - z never appears

    out[1:dlin+1,:]=x[:,warmup_pts:maxtime_pts]  # no -1 here because not doing one-step-ahead prediction, the 1, +1 is to account
                                                     # for constant term, do for the entire time after warmup
    cnt=0
    for row in range(dlin):
        for column in range(row,dlin):
        # important - dlin here, not d (I was making this mistake previously)
            out[dlin+1+cnt,:]=x[row,warmup_pts:maxtime_pts]*x[column,warmup_pts:maxtime_pts]  # also no 1's here, +1 is for constant
            cnt += 1

# drop the first few points when training
# x has the time delays too, so you need to take the first d components

# ONLY train on z variable (which is #2), only use data from training phase
    W_out = lorenz_soln.y[2,warmup_pts:warmtrain_pts] @ out[:,0:traintime_pts].T @ np.linalg.pinv(out[:,0:traintime_pts] @ out[:,0:traintime_pts].T + ridge_param*np.identity(dtot+1))
    
    return x, out, W_out

stime = timeit.default_timer()
x, out, W_out =train()
etime = timeit.default_timer()
print("Program execution time in seconds "+str(etime-stime))

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