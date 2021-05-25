# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 13:17:10 2021

NVAR with time delays.  Don't be efficient for now.

https://www.geeksforgeeks.org/how-to-create-different-subplot-sizes-in-matplotlib/
https://matplotlib.org/stable/gallery/userdemo/demo_gridspec01.html#sphx-glr-gallery-userdemo-demo-gridspec01-py
https://www.delftstack.com/howto/matplotlib/how-to-hide-axis-text-ticks-and-or-tick-labels-in-matplotlib/

May 18: fixed nrmse calculation, error of fixed points

@author: Dan
"""

import numpy as np
#import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

npts=10
start=5.
traintime=10.
ridge_param = 2.5e-6 #.25

warmup_v=np.arange(start,traintime*npts+start,traintime)
train_nrmse_v=np.zeros(npts)
test_nrmse_v=np.zeros(npts)
n_fp1_diff_v=np.zeros(npts)
n_fp2_diff_v=np.zeros(npts)
n_fp0_diff_v=np.zeros(npts)
p_fp1_norm_v=np.zeros((npts, 3))
p_fp2_norm_v=np.zeros((npts, 3))
p_fp0_norm_v=np.zeros((npts, 3))

def find_err(warmup):
    dt=0.025
    #warmup = 60.  # need to have warmup_pts >=1
    #traintime = 10.
    
    lyaptime=1.104
    testtime=lyaptime
    maxtime = warmup+traintime+testtime
    
    warmup_pts=round(warmup/dt)
    traintime_pts=round(traintime/dt)
    warmtrain_pts=warmup_pts+traintime_pts
    testtime_pts=round(testtime/dt)
    maxtime_pts=round(maxtime/dt)
    lyaptime_pts=round(lyaptime/dt)


    d = 3 # input_dimension = 3
    k = 2 # number of time delay daps
    dlin = k*d  # size of linear part of outvector
    dnonlin = int(dlin*(dlin+1)/2)  # size of nonlinear part of outvector
    dtot = 1 + dlin + dnonlin # size of total outvector - add one for the constant vector
    
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
    
    out_train = np.ones((dtot,traintime_pts))  # initialize to one and add a dimension for the constant part
    
    # shift by one to not overwrite the constant part
    out_train[1:dlin+1,:]=x[:,warmup_pts-1:warmtrain_pts-1]
    
    cnt=0
    for row in range(dlin):
        for column in range(row,dlin):
            # important - dlin here, not d (I was making this mistake previously)
            # add one for the constant vector
            out_train[dlin+1+cnt]=x[row,warmup_pts-1:warmtrain_pts-1]*x[column,warmup_pts-1:warmtrain_pts-1]
            cnt += 1
    
    W_out = np.zeros((d,dtot))
    
    # drop the first few points when training
    # x has the time delays too, so you need to take the first d components
    
    # use when subtracting linear part of propagator
    W_out = (x[0:d,warmup_pts:warmtrain_pts]-x[0:d,warmup_pts-1:warmtrain_pts-1]) @ out_train[:,:].T @ np.linalg.pinv(out_train[:,:] @ out_train[:,:].T + ridge_param*np.identity(dtot))
    
    #print('Wout constant weights: '+str(W_out[:,0]))
    # use when not subtracting linear part of propagator
    #W_out = x[0:d,warmup_pts:warmtrain_pts] @ out_train[:,:].T @ np.linalg.pinv(out_train[:,:] @ out_train[:,:].T + ridge_param*np.identity(dtot))
    
    x_predict = np.zeros((d,traintime_pts))
    
    # use when subtracting linear part of propagator
    x_predict = x[0:d,warmup_pts-1:warmtrain_pts-1] + W_out @ out_train[:,0:traintime_pts]
    
    # use when non subtracting linear part of propagator
    #x_predict = W_out @ out_train[:,0:traintime_pts]
    
    total_var=np.var(lorenz_soln.y[0:d,:])
    
    # has dlin components, need to select the first d
    # multiply by 3 so that we are taking the sum of the variances
    train_nrmse = np.sqrt(np.mean((x[0:d,warmup_pts:warmtrain_pts]-x_predict[:,:])**2)/total_var)
    #print('training nrmse: '+str(rms)+'\n')
    
    out_test = np.ones(dtot)  # make ones for the constant vector
    
    # I have an issue in that I need data from the past, but using x_test as I have
    # in other routines assumes I just have data from the current time
    # I need x_test to have the same dimensions as x, which is dlin
    
    x_test = np.zeros((dlin,testtime_pts))
    
    x_test[:,0] = x[:,warmtrain_pts-1]  # don't take from x_predict because it only has d components
        
    for j in range(testtime_pts-1):
        out_test[1:dlin+1]=x_test[:,j]
        # I am not being efficient here - just calculating the all over again - need to fix
        cnt=0
        for row in range(dlin):
            for column in range(row,dlin):
                out_test[dlin+1+cnt]=x_test[row,j]*x_test[column,j]
                cnt += 1
        # need to shift down values, then determine latest prediction
        x_test[d:dlin,j+1] = x_test[0:(dlin-d),j]        
        x_test[0:d,j+1] = x_test[0:d,j] + W_out @ out_test[:]
    
    test_nrmse = np.sqrt(np.mean((x[0:d,warmtrain_pts-1:warmtrain_pts+lyaptime_pts-1]-x_test[0:d,0:lyaptime_pts])**2)/total_var)
    
    #print('k,alpha: '+str(k)+' '+str(ridge_param)+'\n')
    #print('x,y,z nrmse: '+str(xe_sd)+' '+str(ye_sd)+' '+str(ze_sd))
    #print('testing nrmse over 1 Lyapunov time: '+str(nrmse)+'\n')
    
    xtest_stats=np.zeros((3,3))
    for i in range(3):
        xtest_stats[0,i]=np.mean(x_test[i,:])
        xtest_stats[1,i]=np.min(x_test[i,:])
        xtest_stats[2,i]=np.max(x_test[i,:])
        #print('****************')
    #print('un-normalized difference in means for component '+str(2)+' :'+str(np.abs(lorenz_stats[0,2]-xtest_stats[0,2])))
    #print('difference in min for component '+str(2)+' :'+str(np.abs(lorenz_stats[1,2]-xtest_stats[1,2])))
    #print('difference in max for component '+str(2)+' :'+str(np.abs(lorenz_stats[2,2]-xtest_stats[2,2])))
    
    # setup variabled for predicted and true fixed points
    p_fp=np.zeros(d)
    t_fp0=np.zeros(d)
    t_fp1=np.zeros(d)
    t_fp2=np.zeros(d)
    t_fp1[0]=np.sqrt(beta*(rho-1))
    t_fp1[1]=np.sqrt(beta*(rho-1))
    t_fp1[2]=rho-1
    t_fp2[0]=-t_fp1[0]
    t_fp2[1]=-t_fp1[1]
    t_fp2[2]=t_fp1[2]
    
    def func(p_fp):
        func=np.zeros(d)
        out_vec=np.ones(dtot)
        for ii in range(k):
            out_vec[1+ii*d:1+(ii+1)*d]=p_fp[0:d]
        cnt=0
        for row in range(dlin):
            for column in range(row,dlin):
            # important - dlin here, not d (I was making this mistake previously)
                out_vec[dlin+1+cnt]=out_vec[1+row]*out_vec[1+column]
                cnt += 1
        func = W_out @ out_vec
        return func
    
    p_fp1 = fsolve(func, t_fp1)
    #print(' true, predicted, difference first fixed point: \n'+str(t_fp1)+str(p_fp)+str(np.abs(t_fp1-p_fp)))
    #print(' normalized L2 distance to first fixed point: '+str(np.sqrt(np.sum((t_fp1-p_fp)**2)/total_var))+'\n')
    
    n_fp1_diff=np.sqrt(np.sum((t_fp1-p_fp1)**2)/total_var)
    p_fp1_norm = (t_fp1 - p_fp1) / np.sqrt(total_var)
    
    p_fp2 = fsolve(func, t_fp2)
    #print(' true, predicted, difference second fixed point: \n'+str(t_fp2)+str(p_fp)+str(np.abs(t_fp2-p_fp)))
    #print(' normalized L2 distance to second fixed point: '+str(np.sqrt(np.sum((t_fp2-p_fp)**2)/total_var)))
    n_fp2_diff=np.sqrt(np.sum((t_fp2-p_fp2)**2)/total_var)
    p_fp2_norm = (t_fp2 - p_fp2) / np.sqrt(total_var)

    p_fp0=fsolve(func, t_fp0)
    n_fp0_diff=np.sqrt(np.sum((t_fp0-p_fp0)**2)/total_var)
    p_fp0_norm = (t_fp0 - p_fp0) / np.sqrt(total_var)
    #print(p_fp)
    #print(t_fp0)
    #print(n_fp0_diff)
    
    return train_nrmse,test_nrmse,n_fp1_diff,n_fp2_diff,n_fp0_diff,p_fp1_norm,p_fp2_norm,p_fp0_norm

for i in range(npts):
    train_nrmse_v[i],test_nrmse_v[i],n_fp1_diff_v[i],n_fp2_diff_v[i],n_fp0_diff_v[i],p_fp1_norm_v[i],p_fp2_norm_v[i],p_fp0_norm_v[i]=find_err(warmup_v[i])

print('\n ridge regression parameter: '+str(ridge_param)+'\n')
print('mean, meanerr, train nrmse: '+str(np.mean(train_nrmse_v))+' '+str(np.std(train_nrmse_v)/np.sqrt(npts)))
print('mean, meanerr, test nrmse: '+str(np.mean(test_nrmse_v))+' '+str(np.std(test_nrmse_v)/np.sqrt(npts)))
print('mean, meanerr, fp1 nL2 distance: '+str(np.mean(n_fp1_diff_v))+' '+str(np.std(n_fp1_diff_v)/np.sqrt(npts)))
print('mean, meanerr, fp2 nL2 distance: '+str(np.mean(n_fp2_diff_v))+' '+str(np.std(n_fp2_diff_v)/np.sqrt(npts)))
print('mean, meanerr, fp0 nL2 distance: '+str(np.mean(n_fp0_diff_v))+' '+str(np.std(n_fp0_diff_v)/np.sqrt(npts)))
print()
print('mean, meanerr, fp1', np.mean(p_fp1_norm_v, axis=0), np.std(p_fp1_norm_v, axis=0) / np.sqrt(npts))
print('mean, meanerr, fp2', np.mean(p_fp2_norm_v, axis=0), np.std(p_fp2_norm_v, axis=0) / np.sqrt(npts))
print('mean, meanerr, fp0', np.mean(p_fp0_norm_v, axis=0), np.std(p_fp0_norm_v, axis=0) / np.sqrt(npts))
print()
print('nL2 distance to mean, meanerr, fp1', np.sqrt(np.sum(np.mean(p_fp1_norm_v, axis=0) ** 2)), np.sqrt(np.sum(np.var(p_fp1_norm_v, axis=0)) / npts))
print('nL2 distance to mean, meanerr, fp2', np.sqrt(np.sum(np.mean(p_fp2_norm_v, axis=0) ** 2)), np.sqrt(np.sum(np.var(p_fp2_norm_v, axis=0)) / npts))
print('nL2 distance to mean, meanerr, fp0', np.sqrt(np.sum(np.mean(p_fp0_norm_v, axis=0) ** 2)), np.sqrt(np.sum(np.var(p_fp0_norm_v, axis=0)) / npts))

"""             
# won't work because parameters local to fine_err
t_linewidth=1.1
a_linewidth=.3
#fig1, axs1 = plt.subplots(4,2)
fig1 = plt.figure()
fig1.set_figheight(8)
fig1.set_figwidth(12)

#xlabel=[10,15,20,25,30]
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
#axs1.text(-30,46,'a)')
axs2.set_title('training phase') #, k='+str(k)+' alpha='+str(ridge_param))
axs2.plot(t_eval[warmup_pts:warmtrain_pts]-warmup,x[0,warmup_pts:warmtrain_pts],linewidth=t_linewidth)
axs2.plot(t_eval[warmup_pts:warmtrain_pts]-warmup,x_predict[0,:],linewidth=t_linewidth, color='r')
axs2.set_ylabel('x')
#axs2.text(-2.1,15,'b)')
axs2.axes.xaxis.set_ticklabels([])
axs3.plot(t_eval[warmup_pts:warmtrain_pts]-warmup,x[1,warmup_pts:warmtrain_pts],linewidth=t_linewidth)
axs3.plot(t_eval[warmup_pts:warmtrain_pts]-warmup,x_predict[1,:],linewidth=t_linewidth,color='r')
axs3.set_ylabel('y')
#axs3.text(-2.1,21,'c)')
axs3.axes.xaxis.set_ticklabels([])
axs4.plot(t_eval[warmup_pts:warmtrain_pts]-warmup,x[2,warmup_pts:warmtrain_pts],linewidth=t_linewidth)
axs4.plot(t_eval[warmup_pts:warmtrain_pts]-warmup,x_predict[2,:],linewidth=t_linewidth,color='r')
axs4.set_ylabel('z')
#axs4.text(-2.1,41,'d)')
axs4.set_xlabel('time')

#fig2, axs2 = plt.subplots(3)
#plt.subplots_adjust(hspace=0.43)
axs5.plot(x_test[0,:],x_test[2,:],linewidth=a_linewidth,color='r')
axs5.set_xlabel('x')
axs5.set_ylabel('z')
axs5.set_title('NG-RC prediction')
#axs1.text(60,46,'e)')
axs6.set_title('testing phase') #, k='+str(k)+' alpha='+str(ridge_param))
#axs6.set_xticks(xlabel)
axs6.plot(t_eval[warmtrain_pts-1:maxtime_pts-1]-warmup,x[0,warmtrain_pts-1:maxtime_pts-1],linewidth=t_linewidth)
axs6.plot(t_eval[warmtrain_pts-1:maxtime_pts-1]-warmup,x_test[0,:],linewidth=t_linewidth,color='r')
axs6.set_ylabel('x')
#axs6.text(5.75,15,'f)')
axs6.axes.xaxis.set_ticklabels([])
#axs7.set_xticks(xlabel)
axs7.plot(t_eval[warmtrain_pts-1:maxtime_pts-1]-warmup,x[1,warmtrain_pts-1:maxtime_pts-1],linewidth=t_linewidth)
axs7.plot(t_eval[warmtrain_pts-1:maxtime_pts-1]-warmup,x_test[1,:],linewidth=t_linewidth,color='r')
axs7.set_ylabel('y')
#axs7.text(5.75,21,'g)')
axs7.axes.xaxis.set_ticklabels([])
#axs8.set_xticks(xlabel)
axs8.plot(t_eval[warmtrain_pts-1:maxtime_pts-1]-warmup,x[2,warmtrain_pts-1:maxtime_pts-1],linewidth=t_linewidth)
axs8.plot(t_eval[warmtrain_pts-1:maxtime_pts-1]-warmup,x_test[2,:],linewidth=t_linewidth,color='r')
axs8.set_ylabel('z')
#axs8.text(5.75,41,'h)')
axs8.set_xlabel('time')
"""
