# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 13:17:10 2021

NVAR with time delays.  Don't be efficient for now.

https://www.geeksforgeeks.org/how-to-create-different-subplot-sizes-in-matplotlib/
https://matplotlib.org/stable/gallery/userdemo/demo_gridspec01.html#sphx-glr-gallery-userdemo-demo-gridspec01-py
https://www.delftstack.com/howto/matplotlib/how-to-hide-axis-text-ticks-and-or-tick-labels-in-matplotlib/

@author: Dan

## I just modified a few lines to include the constant term in the feature vector and in the Wout plot - Wendson.
## Also, now we are using RK23 for the Lorenz system integration
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

dt=0.025
warmup = 10.  # need to have warmup_pts >=1
traintime = 10.
testtime=120.
maxtime = warmup+traintime+testtime
lyaptime=1.104

warmup_pts=round(warmup/dt)
traintime_pts=round(traintime/dt)
warmtrain_pts=warmup_pts+traintime_pts
testtime_pts=round(testtime/dt)
maxtime_pts=round(maxtime/dt)
lyaptime_pts=round(lyaptime/dt)

# I added "cte" to include the constant term in the features vector - Wendson
# choose cte = 1 to include the constant term in the features vector or cte = 0 otherwise. 
cte = 1 # 

d = 3 # input_dimension = 3
k = 2 # number of time delay daps
dlin = k*d  # size of linear part of outvector
dnonlin = int(dlin*(dlin+1)/2)  # size of nonlinear part of outvector
dtot = dlin + dnonlin # size of total outvector

ridge_param = 2.5e-6 #0.12#0.08#10.

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

lorenz_stats=np.zeros((3,3))
for i in range(3):
    lorenz_stats[0,i]=np.mean(lorenz_soln.y[i,warmtrain_pts:maxtime_pts])
    lorenz_stats[1,i]=np.min(lorenz_soln.y[i,warmtrain_pts:maxtime_pts])
    lorenz_stats[2,i]=np.max(lorenz_soln.y[i,warmtrain_pts:maxtime_pts])

x = np.zeros((dlin,maxtime_pts))

for delay in range(k):
    for j in range(delay,maxtime_pts):
        x[d*delay:d*(delay+1),j]=lorenz_soln.y[:,j-delay]   # don't subtract mean or normalize - goes to negative numbers for first k points

out_train = np.ones((dtot+ cte,traintime_pts))  

out_train[cte:dlin + cte,:]=x[:,warmup_pts-1:warmtrain_pts-1]

cnt=0
for row in range(dlin):
    for column in range(row,dlin):
        # important - dlin here
        out_train[dlin+cnt + cte]=x[row,warmup_pts-1:warmtrain_pts-1]*x[column,warmup_pts-1:warmtrain_pts-1]
        cnt += 1
        

W_out = np.zeros((d,dtot))

# drop the first few points when training
# x has the time delays too, so you need to take the first d components

# use when subtracting linear part of propagator
W_out = (x[0:d,warmup_pts:warmtrain_pts]-x[0:d,warmup_pts-1:warmtrain_pts-1]) @ out_train[:,:].T @ np.linalg.pinv(out_train[:,:] @ out_train[:,:].T + ridge_param*np.identity(dtot+ cte))


#W_out = (x[0:d,warmup_pts:warmtrain_pts]) @ out_train[:,:].T @ np.linalg.pinv(out_train[:,:] @ out_train[:,:].T + ridge_param*np.identity(dtot+ cte))


# plot W_out for each component
#plt.bar(range(dtot), W_out[0,:])
#plt.show()
#
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

##### zoom in ####

fig1a, axs1a = plt.subplots(1,3)
fig1a.set_figheight(7.)
fig1a.set_figwidth(6.)

axs1a[0].barh(y_pos,W_out[0,:],color=colorx)
axs1a[0].set_yticks(y_pos)
axs1a[0].set_yticklabels(labels)
axs1a[0].set_ylim(26.5+cte,-.5)
axs1a[0].set_xlim(-.07,.07)
axs1a[0].set_xticks([-0.05,0.,.05])
axs1a[0].set_xlabel('$[W_{out}]_x$')
axs1a[0].grid()

axs1a[1].barh(y_pos,W_out[1,:],color=colory)
axs1a[1].set_yticks(y_pos)
axs1a[1].axes.set_yticklabels([])
axs1a[1].set_ylim(26.5+cte,-.5)
axs1a[1].set_xlim(-.1,.1)
axs1a[1].set_xticks([-0.07,0.,.07])
axs1a[1].set_xlabel('$[W_{out}]_y$')
axs1a[1].grid()

axs1a[2].barh(y_pos,W_out[2,:],color=colorz)
axs1a[2].set_yticks(y_pos)
axs1a[2].axes.set_yticklabels([]) #,rotation='vertical')
axs1a[2].set_ylim(26.5+cte,-.5)
axs1a[2].set_xlim(-.05,.05)
axs1a[2].set_xticks([-0.04,0.,.04])
axs1a[2].set_xlabel('$[W_{out}]_z$')
axs1a[2].grid()
