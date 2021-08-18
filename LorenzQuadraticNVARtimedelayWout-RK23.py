# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 13:17:10 2021

NVAR with time delays for Lorenz forecasting, W_out plots.
Don't be efficient for now.

@author: Dan
"""
 
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

##
## Parameters
##

# time step
dt=0.025
# units of time to warm up NVAR. need to have warmup_pts >= 1
warmup = 5.
# units of time to train for
traintime = 10.
# units of time to test for
testtime=120.
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

# choose cte = 1 to include constant term, cte = 0 to exclude it
cte = 1

# input dimension
d = 3
# number of time delay taps
k = 2
# size of linear part of feature vector
dlin = k*d
# size of nonlinear part of feature vector
dnonlin = int(dlin*(dlin+1)/2)
# total size of feature vector: linear + nonlinear
dtot = dlin + dnonlin

# ridge parameter for regression
ridge_param = 2.5e-6

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
out_train = np.ones((dtot+ cte,traintime_pts))  

# copy over the linear part (shift over by one to account for constant if needed)
out_train[cte:dlin + cte,:]=x[:,warmup_pts-1:warmtrain_pts-1]

# fill in the non-linear part
cnt=0
for row in range(dlin):
    for column in range(row,dlin):
        # shift by one for constant if needed
        out_train[dlin+cnt + cte]=x[row,warmup_pts-1:warmtrain_pts-1]*x[column,warmup_pts-1:warmtrain_pts-1]
        cnt += 1
        

# ridge regression: train W_out to map out_train to Lorenz[t] - Lorenz[t - 1]
W_out = (x[0:d,warmup_pts:warmtrain_pts]-x[0:d,warmup_pts-1:warmtrain_pts-1]) @ out_train[:,:].T @ np.linalg.pinv(out_train[:,:] @ out_train[:,:].T + ridge_param*np.identity(dtot+ cte))

##
## Plot
##

y_pos=np.arange(dtot + cte)
if cte == 0:
    labels = ['x(t)','y(t)','z(t)','x(t-dt)','y(t-dt)','z(t-dt)']
else:
    labels = ['c','x(t)','y(t)','z(t)','x(t-dt)','y(t-dt)','z(t-dt)']
labels += ['x(t)x(t)','x(t)y(t)','x(t)z(t)','x(t)x(t-dt)','x(t)y(t-dt)','x(t)z(t-dt)']
labels += ['y(t)y(t)','y(t)z(t)','y(t)x(t-dt)','y(t)y(t-dt)','y(t)z(t-dt)']
labels += ['z(t)z(t)','z(t)x(t-dt)','z(t)y(t-dt)','z(t)z(t-dt)']
labels += ['x(t-dt)x(t-dt)','x(t-dt)y(t-dt)','x(t-dt)z(t-dt)']
labels += ['y(t-dt)y(t-dt)','y(t-dt)z(t-dt)']
labels += ['z(t-dt)z(t-dt)']


if cte == 0:
    colorx = ['r','r']  # color red if directly in the vector field
    for ix in range(25):
        colorx += ['b']
        
    colory = ['r','r','b','b','b','b','b','b','r']
    for iy in range(18):
        colory += ['b']
        
    colorz = ['b','b','r','b','b','b','b','r']
    for iz in range(19):
        colorz += ['b']
else:
    colorx = ['b','r','r']  # color red if directly in the vector field
    for ix in range(25):
        colorx += ['b']
        
    colory = ['b','r','r','b','b','b','b','b','b','r']
    for iy in range(18):
        colory += ['b']
        
    colorz = ['b','b','b','r','b','b','b','b','r']
    for iz in range(19):
        colorz += ['b']
    
# the default figure size is 6.4" high and 4.8" wide
# https://matplotlib.org/2.0.2/users/mathtext.html

fig1a, axs1a = plt.subplots(1,3)
fig1a.set_figheight(7.)
fig1a.set_figwidth(6.)

axs1a[0].barh(y_pos,W_out[0,:],color=colorx)
axs1a[0].set_yticks(y_pos)
axs1a[0].set_yticklabels(labels)
axs1a[0].set_ylim(26.5+cte,-.5)
axs1a[0].set_xlabel('$[W_{out}]_x$')
axs1a[0].grid()

axs1a[1].barh(y_pos,W_out[1,:],color=colory)
axs1a[1].set_yticks(y_pos)
axs1a[1].axes.set_yticklabels([])
axs1a[1].set_ylim(26.5+cte,-.5)
axs1a[1].set_xlabel('$[W_{out}]_y$')
axs1a[1].grid()

axs1a[2].barh(y_pos,W_out[2,:],color=colorz)
axs1a[2].set_yticks(y_pos)
axs1a[2].axes.set_yticklabels([]) #,rotation='vertical')
axs1a[2].set_ylim(26.5+cte,-.5)
#axs1a[2].set_xlim(-.09,.1)
#axs1a[2].set_xticks([-.08,0.,.08])
axs1a[2].set_xlabel('$[W_{out}]_z$')
axs1a[2].grid()

plt.savefig('predict-lorenz-wout.png', bbox_inches='tight')
plt.savefig('predict-lorenz-wout.svg', bbox_inches='tight')
plt.savefig('predict-lorenz-wout.eps', bbox_inches='tight')
plt.savefig('predict-lorenz-wout.pdf', bbox_inches='tight')

##### zoom in ####

fig1b, axs1b = plt.subplots(1,3)
fig1b.set_figheight(7.)
fig1b.set_figwidth(6.)

axs1b[0].barh(y_pos,W_out[0,:],color=colorx)
axs1b[0].set_yticks(y_pos)
axs1b[0].set_yticklabels(labels)
axs1b[0].set_ylim(26.5+cte,-.5)
axs1b[0].set_xlim(-.2,.2)
axs1b[0].set_xticks([-0.1,0.,.1])
axs1b[0].set_xlabel('$[W_{out}]_x$')
axs1b[0].grid()

axs1b[1].barh(y_pos,W_out[1,:],color=colory)
axs1b[1].set_yticks(y_pos)
axs1b[1].axes.set_yticklabels([])
axs1b[1].set_ylim(26.5+cte,-.5)
axs1b[1].set_xlim(-.3,.3)
axs1b[1].set_xticks([-0.2,0.,.2])
axs1b[1].set_xlabel('$[W_{out}]_y$')
axs1b[1].grid()

axs1b[2].barh(y_pos,W_out[2,:],color=colorz)
axs1b[2].set_yticks(y_pos)
axs1b[2].axes.set_yticklabels([]) #,rotation='vertical')
axs1b[2].set_ylim(26.5+cte,-.5)
axs1b[2].set_xlim(-.07,.07)
axs1b[2].set_xticks([-0.05,0.,.05])
axs1b[2].set_xlabel('$[W_{out}]_z$')
axs1b[2].grid()

plt.savefig('predict-lorenz-wout-zoom.png', bbox_inches='tight')
plt.savefig('predict-lorenz-wout-zoom.svg', bbox_inches='tight')
plt.savefig('predict-lorenz-wout-zoom.eps', bbox_inches='tight')
plt.savefig('predict-lorenz-wout-zoom.pdf', bbox_inches='tight')
