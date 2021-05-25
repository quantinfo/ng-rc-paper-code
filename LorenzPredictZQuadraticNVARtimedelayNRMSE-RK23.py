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
"""

import numpy as np
#import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
#from numba import jit
import timeit

npts=10
start=5.
interval=20.
warmup_v=np.arange(start,interval*npts+start,interval)
train_nrmse_v=np.zeros(npts)
test_nrmse_v=np.zeros(npts)
run_time_v=np.zeros(npts)

ridge_param = .05

def find_err(warmup):
    dt=0.05
    #warmup = 5.  # need to have warmup_pts >=1
    traintime = 20.
    testtime=45.
    maxtime = warmup+traintime+testtime
    
    warmup_pts=round(warmup/dt)
    traintime_pts=round(traintime/dt)
    warmtrain_pts=warmup_pts+traintime_pts
    #testtime_pts=round(testtime/dt)
    maxtime_pts=round(maxtime/dt)
    
    d = 3 # input_dimension = 3
    k = 4 # number of time delay daps
    skip = 5 # skip = 1 means take consecutive points
    #dlin = k*d  # size of linear part of vector X
    dlin = k*(d-1)  # leave out z
    dnonlin = int(dlin*(dlin+1)/2)  # size of nonlinear part of outvector
    dtot = 1+dlin + dnonlin # size of total outvector - add one for the constant vector
    #print('dtot: '+str(dtot))
    
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
    out = np.ones((dtot,maxtime_pts-warmup_pts))  # intiialize to ones, which will take care of the constant vector, add +1 for constant
    W_out = np.zeros(dtot)
    
    #print('Wout shape '+str(W_out.shape)+' out shape '+str(out.shape))
    
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
        W_out = lorenz_soln.y[2,warmup_pts:warmtrain_pts] @ out[:,0:traintime_pts].T @ np.linalg.pinv(out[:,0:traintime_pts] @ out[:,0:traintime_pts].T + ridge_param*np.identity(dtot))
        
        return x, out, W_out
    
    stime = timeit.default_timer()
    x, out, W_out = train()
    etime = timeit.default_timer()
    run_time=etime-stime
    #print("Program execution time in seconds "+str(etime-stime))
    
    # once we have W_out, we can predict the entire shot
    z_predict = np.zeros(maxtime_pts-warmup_pts)  # z variable
    
    z_predict = W_out @ out[:,:]
    #"""
    #elapsed_time = timeit.timeit(code_to_test,number=1)
    #print(elapsed_time)
    
    #print('Wout shape '+str(W_out.shape)+' out shape '+str(out.shape))
    # has dlin components, need to seledt the first d
    train_nrmse = np.sqrt(np.mean((lorenz_soln.y[2,warmup_pts:warmtrain_pts]-z_predict[0:traintime_pts])**2))/zstd
    
    #print('training rms: '+str(rms))
    #print('training nrmse: '+str(train_nrmse))
    
    test_nrmse = np.sqrt(np.mean((lorenz_soln.y[2,warmtrain_pts:maxtime_pts]-z_predict[traintime_pts:maxtime_pts-warmup_pts])**2))/zstd
    #print('testing rms: '+str(rms))
    #print('testing nrmse: '+str(test_nrmse))
    return train_nrmse,test_nrmse,run_time

for i in range(npts):
    train_nrmse_v[i],test_nrmse_v[i],run_time_v[i]=find_err(warmup_v[i])

print('\n ridge regression parameter: '+str(ridge_param)+'\n')
print('mean, meanerr, train nrmse: '+str(np.mean(train_nrmse_v))+' '+str(np.std(train_nrmse_v)/np.sqrt(npts)))
print('mean, meanerr, test nrmse: '+str(np.mean(test_nrmse_v))+' '+str(np.std(test_nrmse_v)/np.sqrt(npts)))
print('mean, meanerr, run time: '+str(np.mean(run_time_v))+' '+str(np.std(run_time_v)/np.sqrt(npts)))

"""
# https://riptutorial.com/matplotlib/example/16030/coordinate-systems-and-text
    
# plot over the training time
t_linewidth=.8
#fig1, axs1 = plt.subplots(4,2)
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


#plt.suptitle('training phase, k='+str(k)+' alpha='+str(ridge_param)+' skip='+str(skip))
axs1.set_title('training phase')
axs1.plot(t_eval[warmup_pts:warmtrain_pts]-warmup,x[0,warmup_pts:warmtrain_pts],color='b',linewidth=t_linewidth)
axs1.set_ylabel('x')
axs1.axes.xaxis.set_ticklabels([])
axs1.axes.set_xbound(-.08,7.05)
axs1.axes.set_ybound(-21.,21.)
axs1.text(-.14,.9,'a)', ha='left', va='bottom',transform=axs1.transAxes)
#axs1.yaxis.label.set_color('deepskyblue')
axs2.plot(t_eval[warmup_pts:warmtrain_pts]-warmup,x[1,warmup_pts:warmtrain_pts],color='b',linewidth=t_linewidth)
axs2.set_ylabel('y')
axs2.axes.xaxis.set_ticklabels([])
axs2.axes.set_xbound(-.08,7.05)
axs2.axes.set_ybound(-26.,26.)
axs2.text(-.14,.9,'b)', ha='left', va='bottom',transform=axs2.transAxes)
#axs2.yaxis.label.set_color('r')
axs3.plot(t_eval[warmup_pts:warmtrain_pts]-warmup,lorenz_soln.y[2,warmup_pts:warmtrain_pts],color='b',linewidth=t_linewidth)
axs3.plot(t_eval[warmup_pts:warmtrain_pts]-warmup,z_predict[0:traintime_pts],color='r',linewidth=t_linewidth)
axs3.set_ylabel('z')
#axs3.yaxis.label.set_color('lime')
axs3.set_xlabel('time')
axs3.axes.set_xbound(-.08,7.05)
axs3.axes.set_ybound(3.,48.)
axs3.text(-.14,.9,'c)', ha='left', va='bottom',transform=axs3.transAxes)

axs4.set_title('testing phase')
axs4.plot(t_eval[warmtrain_pts:maxtime_pts]-warmup,x[0,warmtrain_pts:maxtime_pts],color='b',linewidth=t_linewidth)
axs4.set_ylabel('x')
axs4.axes.xaxis.set_ticklabels([])
axs4.axes.set_ybound(-21.,21.)
axs4.axes.set_xbound(6.5,50.5)
axs4.text(-.14,.9,'d)', ha='left', va='bottom',transform=axs4.transAxes)
#axs4.yaxis.label.set_color('deepskyblue')
axs5.plot(t_eval[warmtrain_pts:maxtime_pts]-warmup,x[1,warmtrain_pts:maxtime_pts],color='b',linewidth=t_linewidth)
axs5.set_ylabel('y')
axs5.axes.xaxis.set_ticklabels([])
axs5.axes.set_ybound(-26.,26.)
axs5.axes.set_xbound(6.5,50.5)
axs5.text(-.14,.9,'e)', ha='left', va='bottom',transform=axs5.transAxes)
#axs5.yaxis.label.set_color('r')
axs6.plot(t_eval[warmtrain_pts:maxtime_pts]-warmup,lorenz_soln.y[2,warmtrain_pts:maxtime_pts],color='b',linewidth=t_linewidth)
axs6.plot(t_eval[warmtrain_pts:maxtime_pts]-warmup,z_predict[traintime_pts:maxtime_pts-warmup_pts],color='r',linewidth=t_linewidth)
axs6.set_ylabel('z')
#axs1[5].yaxis.label.set_color('lime')
axs6.set_xlabel('time')
axs6.axes.set_ybound(3.,48.)
axs6.axes.set_xbound(6.5,50.5)
axs6.text(-.14,.9,'f)', ha='left', va='bottom',transform=axs6.transAxes)
"""
