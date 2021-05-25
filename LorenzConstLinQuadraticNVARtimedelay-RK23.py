# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 13:17:10 2021

NVAR with time delays.  Don't be efficient for now.

https://www.geeksforgeeks.org/how-to-create-different-subplot-sizes-in-matplotlib/
https://matplotlib.org/stable/gallery/userdemo/demo_gridspec01.html#sphx-glr-gallery-userdemo-demo-gridspec01-py
https://www.delftstack.com/howto/matplotlib/how-to-hide-axis-text-ticks-and-or-tick-labels-in-matplotlib/

@author: Dan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

dt=0.025
warmup = 5.  # need to have warmup_pts >=1
traintime = 10.
testtime=120.
maxtime = warmup+traintime+testtime
plottime=20.
lyaptime=1.104

warmup_pts=round(warmup/dt)
traintime_pts=round(traintime/dt)
warmtrain_pts=warmup_pts+traintime_pts
testtime_pts=round(testtime/dt)
maxtime_pts=round(maxtime/dt)
plottime_pts=round(plottime/dt)
lyaptime_pts=round(lyaptime/dt)

d = 3 # input_dimension = 3
k = 2 # number of time delay daps
dlin = k*d  # size of linear part of outvector
dnonlin = int(dlin*(dlin+1)/2)  # size of nonlinear part of outvector
dtot = 1 + dlin + dnonlin # size of total outvector -- add one for constant

ridge_param = 2.5e-6 # 0.25

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

total_var=np.var(lorenz_soln.y[0:d,:])

x = np.zeros((dlin,maxtime_pts))

for delay in range(k):
    for j in range(delay,maxtime_pts):
        x[d*delay:d*(delay+1),j]=lorenz_soln.y[:,j-delay]   # don't subtract mean or normalize - goes to negative numbers for first k points

out_train = np.ones((dtot,traintime_pts))  # set to ones for constant vector

out_train[1:dlin+1,:]=x[:,warmup_pts-1:warmtrain_pts-1] # shift by one for constant

cnt=0
for row in range(dlin):
    for column in range(row,dlin):
        # important - dlin here, not d (I was making this mistake previously)
        # shift by one for constant
        out_train[dlin+1+cnt]=x[row,warmup_pts-1:warmtrain_pts-1]*x[column,warmup_pts-1:warmtrain_pts-1]
        cnt += 1

W_out = np.zeros((d,dtot))

# drop the first few points when training
# x has the time delays too, so you need to take the first d components

# use when subtracting linear part of propagator
W_out = (x[0:d,warmup_pts:warmtrain_pts]-x[0:d,warmup_pts-1:warmtrain_pts-1]) @ out_train[:,:].T @ np.linalg.pinv(out_train[:,:] @ out_train[:,:].T + ridge_param*np.identity(dtot))

# use when not subtracting linear part of propagator
#W_out = x[0:d,warmup_pts:warmtrain_pts] @ out_train[:,:].T @ np.linalg.pinv(out_train[:,:] @ out_train[:,:].T + ridge_param*np.identity(dtot))

x_predict = np.zeros((d,traintime_pts))

# use when subtracting linear part of propagator
x_predict = x[0:d,warmup_pts-1:warmtrain_pts-1] + W_out @ out_train[:,0:traintime_pts]

# use when non subtracting linear part of propagator
#x_predict = W_out @ out_train[:,0:traintime_pts]



#fig1a, axs1a = plt.subplots(3)
#plt.suptitle('training phase, k='+str(k)+' alpha='+str(ridge_param))
#axs1a[0].plot(t_eval[warmup_pts:warmtrain_pts],x[0,warmup_pts:warmtrain_pts]-x_predict[0,:],label='predict')
#axs1a[0].set_ylabel('x')
#axs1a[1].plot(t_eval[warmup_pts:warmtrain_pts],x[1,warmup_pts:warmtrain_pts]-x_predict[1,:],label='predict')
#axs1a[1].set_ylabel('y')
#axs1a[2].plot(t_eval[warmup_pts:warmtrain_pts],x[2,warmup_pts:warmtrain_pts]-x_predict[2,:],label='predict')
#axs1a[2].set_ylabel('z')
#axs1a[2].set_xlabel('time')

# has dlin components, need to seledt the first d
rms = np.sqrt(np.mean((x[0:d,warmup_pts:warmtrain_pts]-x_predict[:,:])**2)/total_var)
print('training nrmse: '+str(rms))

# have not edited from here down

out_test = np.zeros(dtot)

#out_test = out_train[:,traintime_pts-1]

# I have an issue in that I need data from the past, but using x_test as I have
# in other routines assumes I just have data from the current time
# I need x_test to have the same dimensions as x, which is dlin

x_test = np.zeros((dlin,testtime_pts))

x_test[:,0] = x[:,warmtrain_pts-1]  # don't take from x_predict because it only has d components
    
for j in range(testtime_pts-1):
    out_test[1:dlin+1]=x_test[:,j] # shift by one for constant
    # I am not being efficient here - just calculating the all over again - need to fix
    cnt=0
    for row in range(dlin):
        for column in range(row,dlin):
            # +1 for constant
            out_test[dlin+1+cnt]=x_test[row,j]*x_test[column,j]
            cnt += 1
    # need to shift down values, then determine latest prediction
    x_test[d:dlin,j+1]=x_test[0:(dlin-d),j]        
    x_test[0:d,j+1] = x_test[0:d,j]+W_out @ out_test[:]

test_nrmse = np.sqrt(np.mean((x[0:d,warmtrain_pts-1:warmtrain_pts+lyaptime_pts-1]-x_test[0:d,0:lyaptime_pts])**2)/total_var)
print('test nrmse: '+str(test_nrmse))

t_linewidth=1.1
a_linewidth=0.3
#fig1, axs1 = plt.subplots(4,2)
plt.rcParams.update({'font.size': 12})
fig1 = plt.figure()
fig1.set_figheight(8)
fig1.set_figwidth(12)

xlabel=[10,15,20,25,30]
h=120
w=100
# top left of grid is 0,0
axs1 = plt.subplot2grid(shape=(h,w), loc=(0, 9), colspan=22, rowspan=38) 
axs2 = plt.subplot2grid(shape=(h,w), loc=(52, 0), colspan=42, rowspan=20)
axs3 = plt.subplot2grid(shape=(h,w), loc=(75, 0), colspan=42, rowspan=20)
axs4 = plt.subplot2grid(shape=(h,w), loc=(98, 0), colspan=42, rowspan=20)
axs5 = plt.subplot2grid(shape=(h,w), loc=(0, 61), colspan=22, rowspan=38)
axs6 = plt.subplot2grid(shape=(h,w), loc=(52, 50),colspan=42, rowspan=20)
axs7 = plt.subplot2grid(shape=(h,w), loc=(75, 50), colspan=42, rowspan=20)
axs8 = plt.subplot2grid(shape=(h,w), loc=(98, 50), colspan=42, rowspan=20)
#plt.subplots_adjust(hspace=0.5)
#plt.suptitle('training phase, k='+str(k)+' alpha='+str(ridge_param))
#axs1 = fig1.add_suplot()
axs1.plot(x[0,warmtrain_pts:maxtime_pts],x[2,warmtrain_pts:maxtime_pts],linewidth=a_linewidth)
axs1.set_xlabel('x')
axs1.set_ylabel('z')
axs1.set_title('ground truth')
axs1.text(-.25,.92,'a)', ha='left', va='bottom',transform=axs1.transAxes)
axs1.axes.set_xbound(-21,21)
axs1.axes.set_ybound(2,48)

axs2.set_title('training phase') #, k='+str(k)+' alpha='+str(ridge_param))
axs2.plot(t_eval[warmup_pts:warmtrain_pts]-warmup,x[0,warmup_pts:warmtrain_pts],linewidth=t_linewidth)
axs2.plot(t_eval[warmup_pts:warmtrain_pts]-warmup,x_predict[0,:],linewidth=t_linewidth, color='r')
axs2.set_ylabel('x')
axs2.text(-.155,0.87,'b)', ha='left', va='bottom',transform=axs2.transAxes)
axs2.axes.xaxis.set_ticklabels([])
axs2.axes.set_ybound(-21.,21.)
axs2.axes.set_xbound(-.15,10.15)

axs3.plot(t_eval[warmup_pts:warmtrain_pts]-warmup,x[1,warmup_pts:warmtrain_pts],linewidth=t_linewidth)
axs3.plot(t_eval[warmup_pts:warmtrain_pts]-warmup,x_predict[1,:],linewidth=t_linewidth,color='r')
axs3.set_ylabel('y')
axs3.text(-.155,0.87,'c)', ha='left', va='bottom',transform=axs3.transAxes)
axs3.axes.xaxis.set_ticklabels([])
axs3.axes.set_xbound(-.15,10.15)

axs4.plot(t_eval[warmup_pts:warmtrain_pts]-warmup,x[2,warmup_pts:warmtrain_pts],linewidth=t_linewidth)
axs4.plot(t_eval[warmup_pts:warmtrain_pts]-warmup,x_predict[2,:],linewidth=t_linewidth,color='r')
axs4.set_ylabel('z')
axs4.text(-.155,0.87,'d)', ha='left', va='bottom',transform=axs4.transAxes)
axs4.set_xlabel('time')
axs4.axes.set_xbound(-.15,10.15)

#fig2, axs2 = plt.subplots(3)
#plt.subplots_adjust(hspace=0.43)
axs5.plot(x_test[0,:],x_test[2,:],linewidth=a_linewidth,color='r')
axs5.set_xlabel('x')
axs5.set_ylabel('z')
axs5.set_title('NG-RC prediction')
axs5.text(-.25,0.92,'e)', ha='left', va='bottom',transform=axs5.transAxes)
axs5.axes.set_xbound(-21,21)
axs5.axes.set_ybound(2,48)

axs6.set_title('testing phase') #, k='+str(k)+' alpha='+str(ridge_param))
axs6.set_xticks(xlabel)
axs6.plot(t_eval[warmtrain_pts-1:warmtrain_pts+plottime_pts-1]-warmup,x[0,warmtrain_pts-1:warmtrain_pts+plottime_pts-1],linewidth=t_linewidth)
axs6.plot(t_eval[warmtrain_pts-1:warmtrain_pts+plottime_pts-1]-warmup,x_test[0,0:plottime_pts],linewidth=t_linewidth,color='r')
axs6.set_ylabel('x')
axs6.text(-.155,0.87,'f)', ha='left', va='bottom',transform=axs6.transAxes)
axs6.axes.xaxis.set_ticklabels([])
axs6.axes.set_xbound(9.7,30.3)

axs7.set_xticks(xlabel)
axs7.plot(t_eval[warmtrain_pts-1:warmtrain_pts+plottime_pts-1]-warmup,x[1,warmtrain_pts-1:warmtrain_pts+plottime_pts-1],linewidth=t_linewidth)
axs7.plot(t_eval[warmtrain_pts-1:warmtrain_pts+plottime_pts-1]-warmup,x_test[1,0:plottime_pts],linewidth=t_linewidth,color='r')
axs7.set_ylabel('y')
axs7.text(-.155,0.87,'g)', ha='left', va='bottom',transform=axs7.transAxes)
axs7.axes.xaxis.set_ticklabels([])
axs7.axes.set_xbound(9.7,30.3)

axs8.set_xticks(xlabel)
axs8.plot(t_eval[warmtrain_pts-1:warmtrain_pts+plottime_pts-1]-warmup,x[2,warmtrain_pts-1:warmtrain_pts+plottime_pts-1],linewidth=t_linewidth)
axs8.plot(t_eval[warmtrain_pts-1:warmtrain_pts+plottime_pts-1]-warmup,x_test[2,0:plottime_pts],linewidth=t_linewidth,color='r')
axs8.set_ylabel('z')
axs8.text(-.155,0.87,'h)', ha='left', va='bottom',transform=axs8.transAxes)
axs8.set_xlabel('time')
axs8.axes.set_xbound(9.7,30.3)

plt.savefig('predict-lorenz.png')
