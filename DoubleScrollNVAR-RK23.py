# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 13:17:10 2021

NVAR with time delays.  Don't be efficient for now.

Modify for double scroll oscillator

Notes:  for a polynomial of size d raised to power n, there are
(d+n-1)!/(d-1)!n! terms.   For n=2, we have d(d+1)/2 terms. For n=3,
we have d(d+1)(d+2)/6 terms.

@author: Dan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import csv
import timeit

def do_nvar(dt=0.25, warmup=1., traintime=100., testtime=800., lyaptime=7.8125):
  #warmup = 1.  # need to have warmup_pts >=1
  #lyaptime=7.8125  # Henry Abarbanel finds Lyapunov exponent is 0.128
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
  dnonlin = int(dlin*(dlin+1)*(dlin+2)/6)  # size of nonlinear part of outvector for only a cubic
  dtot = dlin + dnonlin # size of total outvector

  ridge_param = 1.e-3

  t_eval=np.linspace(0,maxtime,maxtime_pts+1) # need the +1 here to have a step of dt

  # Double scroll

  r1 = 1.2
  r2 = 3.44
  r4 = 0.193
  alpha = 11.6
  ir = 2*2.25e-5 # the 2 is for the hyperbolic sine

  def doublescroll(t, y):
    # y[0] = V1, y[1] = V2, y[2] = I
    dV = y[0]-y[1] # V1-V2
    g = (dV/r2)+ir*np.sinh(alpha*dV)
    dy0 = (y[0]/r1)-g 
    dy1 = g-y[2]
    dy2 = y[1]-r4*y[2]

    return [dy0, dy1, dy2]

  # I integrated out to t=50 to find points on the attractor, then use these as the initial conditions

  doublescroll_soln = solve_ivp(doublescroll, (0, maxtime), [0.37926545,0.058339,-0.08167691] , t_eval=t_eval, method='RK23')
  
  total_var=np.var(doublescroll_soln.y[0:d,:])
  #print(' total var: '+str(total_var)+' sqrt '+str(np.sqrt(total_var)))
   
  stime = timeit.default_timer()

  x = np.zeros((dlin,maxtime_pts))

  stime = timeit.default_timer()
  for delay in range(k):
      for j in range(delay,maxtime_pts):
          x[d*delay:d*(delay+1),j]=doublescroll_soln.y[:,j-delay]   # don't subtract mean or normalize

  #print(doublescroll_soln.y[:,-1])
  out_train = np.zeros((dtot,traintime_pts))  

  out_train[0:dlin,:]=x[:,warmup_pts-1:warmtrain_pts-1]

  # do a cubic layer for double scroll

  cnt=0
  for row in range(dlin):
      for column in range(row,dlin):
          for span in range(column,dlin):
          # important - dlin here, not d (I was making this mistake previously)
              out_train[dlin+cnt]=x[row,warmup_pts-1:warmtrain_pts-1]*x[column,warmup_pts-1:warmtrain_pts-1]*x[span,warmup_pts-1:warmtrain_pts-1]
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

  etime = timeit.default_timer()
  #print("Program execution time in seconds for prediction "+str(etime-stime))

  # use when non subtracting linear part of propagator
  #x_predict = W_out @ out_train[:,0:traintime_pts]


  # has dlin components, need to seledt the first d
  train_nrmse = np.sqrt(np.mean((x[0:d,warmup_pts:warmtrain_pts]-x_predict[:,:])**2)/total_var)
  #print('training nrmse - normalized: '+str(train_nrmse))

  out_test = np.zeros(dtot)

  #out_test = out_train[:,traintime_pts-1]

  # I have an issue in that I need data from the past, but using x_test as I have
  # in other routines assumes I just have data from the current time
  # I need x_test to have the same dimensions as x, which is dlin

  x_test = np.zeros((dlin,testtime_pts))

  x_test[:,0] = x[:,warmtrain_pts-1]  # don't take from x_predict because it only has d components

  stime = timeit.default_timer()

  for j in range(testtime_pts-1):
      out_test[0:dlin]=x_test[:,j]
      # I am not being efficient here - just calculating the all over again - need to fix
      cnt=0
      for row in range(dlin):
          for column in range(row,dlin):
              for span in range(column,dlin):
                  out_test[dlin+cnt]=x_test[row,j]*x_test[column,j]*x_test[span,j]
                  cnt += 1
      # need to shift down values, then determine latest prediction
      x_test[d:dlin,j+1]=x_test[0:(dlin-d),j]        
      x_test[0:d,j+1] = x_test[0:d,j]+W_out @ out_test[:]

  etime = timeit.default_timer()
  #print("Program execution time in seconds for testing "+str(etime-stime))

  ds_stats=np.zeros((3,3))
  for i in range(3):
      ds_stats[0,i]=np.mean(x[i,warmtrain_pts:maxtime_pts])
      ds_stats[1,i]=np.min(x[i,warmtrain_pts:maxtime_pts])
      ds_stats[2,i]=np.max(x[i,warmtrain_pts:maxtime_pts])
      
  nrmse = np.sqrt(np.mean((x[0:d,warmtrain_pts-1:warmtrain_pts+lyaptime_pts-1]-x_test[0:d,0:lyaptime_pts])**2)/total_var)
  #print('k,alpha: '+str(k)+' '+str(ridge_param))
  #print('x,y,z nrmse: '+str(xe_sd)+' '+str(ye_sd)+' '+str(ze_sd))
  #print('nrmse '+str(nrmse))

  xtest_stats=np.zeros((3,3))
  for i in range(3):
      xtest_stats[0,i]=np.mean(x_test[i,:])
      xtest_stats[1,i]=np.min(x_test[i,:])
      xtest_stats[2,i]=np.max(x_test[i,:])
      #print('****************')
      #print('difference in means for component '+str(i)+' :'+str(np.abs(ds_stats[0,i]-xtest_stats[0,i])))
      #print('difference in min for component '+str(i)+' :'+str(np.abs(ds_stats[1,i]-xtest_stats[1,i])))
      #print('difference in max for component '+str(i)+' :'+str(np.abs(ds_stats[2,i]-xtest_stats[2,i])))


  # setup variabled for predicted and true fixed points
  fp_slope = (r1 / r2) - (r4 / r2) - 1
  fp_sinhx = alpha * (1 - r4 / r1)
  fp_sinhy = r1 * ir
  fp_xs = np.linspace(-1.4, 1.4, 50)
  #plt.plot(fp_xs, fp_slope * fp_xs)
  #plt.plot(fp_xs, -fp_sinhy * np.sinh(fp_sinhx * fp_xs))
  #plt.show()
  V1_fp = fsolve(lambda V1: fp_slope * V1 + fp_sinhy * np.sinh(fp_sinhx * V1), 1.05)
  t_fp0=np.zeros(d)
  t_fp1=np.zeros(d)
  t_fp2=np.zeros(d)
  t_fp1[0]=V1_fp
  t_fp1[1]=r4 * V1_fp / r1
  t_fp1[2]=V1_fp / r1
  t_fp2[0]=-t_fp1[0]
  t_fp2[1]=-t_fp1[1]
  t_fp2[2]=-t_fp1[2]

  def func(p_fp):
      func=np.zeros(d)
      out_vec=np.ones(dtot)
      for ii in range(k):
          #out_vec[1+ii*d:1+(ii+1)*d]=p_fp[0:d]
          out_vec[ii*d:(ii+1)*d]=p_fp[0:d]
      cnt=0
      for row in range(dlin):
          for column in range(row,dlin):
            for span in range(column,dlin):
            # important - dlin here, not d (I was making this mistake previously)
              #out_vec[dlin+1+cnt]=out_vec[1+row]*out_vec[1+column]*out_vec[1+span]
              out_vec[dlin+cnt]=out_vec[row]*out_vec[column]*out_vec[span]
              cnt += 1
      func = W_out @ out_vec
      return func

  p_fp1 = fsolve(func, t_fp1)
  #print(' true, predicted, difference first fixed point: \n'+str(t_fp1)+str(p_fp1)+str(np.abs(t_fp1-p_fp1)))
  #print(' normalized L2 distance to first fixed point: '+str(np.sqrt(np.sum((t_fp1-p_fp1)**2)/total_var))+'\n')
  n_fp1_diff=np.sqrt(np.sum((t_fp1-p_fp1)**2)/total_var)
  p_fp1_norm = (t_fp1 - p_fp1) / np.sqrt(total_var)

  p_fp2 = fsolve(func, t_fp2)
  #print(' true, predicted, difference second fixed point: \n'+str(t_fp2)+str(p_fp2)+str(np.abs(t_fp2-p_fp2)))
  #print(' normalized L2 distance to second fixed point: '+str(np.sqrt(np.sum((t_fp2-p_fp2)**2)/total_var)))
  n_fp2_diff=np.sqrt(np.sum((t_fp2-p_fp2)**2)/total_var)
  p_fp2_norm = (t_fp2 - p_fp2) / np.sqrt(total_var)

  p_fp0=fsolve(func, t_fp0)
  n_fp0_diff=np.sqrt(np.sum((t_fp0-p_fp0)**2)/total_var)
  p_fp0_norm = (t_fp0 - p_fp0) / np.sqrt(total_var)
  #print(p_fp0)
  #print(t_fp0)
  #print(n_fp0_diff)

  # return all our local data
  return locals()

def do_plot(dt, x, warmtrain_pts, maxtime_pts, t_eval, warmup_pts, warmup, x_predict, x_test, **extra):
  plottime=200.
  plottime_pts=round(plottime/dt)
  t_linewidth=1.1
  a_linewidth=0.3
  #fig1, axs1 = plt.subplots(4,2)
  plt.rcParams.update({'font.size': 12})
  fig1 = plt.figure()
  fig1.set_figheight(8)
  fig1.set_figwidth(12)

  xlabel=[100,150,200,250,300]
  sectionskip = 2
  h=120 + sectionskip
  w=100
  # top left of grid is 0,0
  axs1 = plt.subplot2grid(shape=(h,w), loc=(0, 9), colspan=22, rowspan=38) 
  axs2 = plt.subplot2grid(shape=(h,w), loc=(52+sectionskip, 0), colspan=42, rowspan=20)
  axs3 = plt.subplot2grid(shape=(h,w), loc=(75+sectionskip, 0), colspan=42, rowspan=20)
  axs4 = plt.subplot2grid(shape=(h,w), loc=(98+sectionskip, 0), colspan=42, rowspan=20)

  axs5 = plt.subplot2grid(shape=(h,w), loc=(0, 61), colspan=22, rowspan=38)
  axs6 = plt.subplot2grid(shape=(h,w), loc=(52+sectionskip, 50),colspan=42, rowspan=20)
  axs7 = plt.subplot2grid(shape=(h,w), loc=(75+sectionskip, 50), colspan=42, rowspan=20)
  axs8 = plt.subplot2grid(shape=(h,w), loc=(98+sectionskip, 50), colspan=42, rowspan=20)
  #plt.subplots_adjust(hspace=0.5)
  #plt.suptitle('training phase, k='+str(k)+' alpha='+str(ridge_param))
  #axs1 = fig1.add_suplot()
  axs1.plot(x[0,warmtrain_pts:maxtime_pts],x[2,warmtrain_pts:maxtime_pts],linewidth=a_linewidth)
  axs1.set_xlabel('$V_1$')
  axs1.set_ylabel('I')
  axs1.set_title('ground truth')
  axs1.text(-.25,.92,'a)', ha='left', va='bottom',transform=axs1.transAxes)
  axs1.axes.set_xbound(-2.1, 2.1)
  axs1.axes.set_ybound(-2.5, 2.5)

  axs2.set_title('training phase') #, k='+str(k)+' alpha='+str(ridge_param))
  axs2.plot(t_eval[warmup_pts:warmtrain_pts]-warmup,x[0,warmup_pts:warmtrain_pts],linewidth=t_linewidth)
  axs2.plot(t_eval[warmup_pts:warmtrain_pts]-warmup,x_predict[0,:],linewidth=t_linewidth, color='r')
  axs2.set_ylabel('$V_1$')
  axs2.text(-.155,0.87,'b)', ha='left', va='bottom',transform=axs2.transAxes)
  axs2.axes.xaxis.set_ticklabels([])
  axs2.axes.set_xbound(-1.5,101.5)
  axs2.axes.set_ybound(-2.1, 2.1)

  axs3.plot(t_eval[warmup_pts:warmtrain_pts]-warmup,x[1,warmup_pts:warmtrain_pts],linewidth=t_linewidth)
  axs3.plot(t_eval[warmup_pts:warmtrain_pts]-warmup,x_predict[1,:],linewidth=t_linewidth,color='r')
  axs3.set_ylabel('$V_2$')
  axs3.text(-.155,0.87,'c)', ha='left', va='bottom',transform=axs3.transAxes)
  axs3.axes.xaxis.set_ticklabels([])
  axs3.axes.set_xbound(-1.5,101.5)
  axs3.axes.set_ybound(-1.1, 1.1)

  axs4.plot(t_eval[warmup_pts:warmtrain_pts]-warmup,x[2,warmup_pts:warmtrain_pts],linewidth=t_linewidth)
  axs4.plot(t_eval[warmup_pts:warmtrain_pts]-warmup,x_predict[2,:],linewidth=t_linewidth,color='r')
  axs4.set_ylabel('I')
  axs4.text(-.155,0.87,'d)', ha='left', va='bottom',transform=axs4.transAxes)
  axs4.set_xlabel('time')
  axs4.axes.set_xbound(-1.5,101.5)
  axs4.axes.set_ybound(-2.5, 2.5)

  #fig2, axs2 = plt.subplots(3)
  #plt.subplots_adjust(hspace=0.43)
  axs5.plot(x_test[0,:],x_test[2,:],linewidth=a_linewidth,color='r')
  axs5.set_xlabel('$V_1$')
  axs5.set_ylabel('I')
  axs5.set_title('NG-RC prediction')
  axs5.text(-.25,0.92,'e)', ha='left', va='bottom',transform=axs5.transAxes)
  axs5.axes.set_xbound(-2.1, 2.1)
  axs5.axes.set_ybound(-2.5, 2.5)

  axs6.set_title('testing phase') #, k='+str(k)+' alpha='+str(ridge_param))
  axs6.set_xticks(xlabel)
  axs6.plot(t_eval[warmtrain_pts-1:warmtrain_pts+plottime_pts-1]-warmup,x[0,warmtrain_pts-1:warmtrain_pts+plottime_pts-1],linewidth=t_linewidth)
  axs6.plot(t_eval[warmtrain_pts-1:warmtrain_pts+plottime_pts-1]-warmup,x_test[0,0:plottime_pts],linewidth=t_linewidth,color='r')
  axs6.set_ylabel('$V_1$')
  axs6.text(-.155,0.87,'f)', ha='left', va='bottom',transform=axs6.transAxes)
  axs6.axes.xaxis.set_ticklabels([])
  axs6.axes.set_xbound(97,303)
  axs6.axes.set_ybound(-2.1, 2.1)

  axs7.set_xticks(xlabel)
  axs7.plot(t_eval[warmtrain_pts-1:warmtrain_pts+plottime_pts-1]-warmup,x[1,warmtrain_pts-1:warmtrain_pts+plottime_pts-1],linewidth=t_linewidth)
  axs7.plot(t_eval[warmtrain_pts-1:warmtrain_pts+plottime_pts-1]-warmup,x_test[1,0:plottime_pts],linewidth=t_linewidth,color='r')
  axs7.set_ylabel('$V_2$')
  axs7.text(-.155,0.87,'g)', ha='left', va='bottom',transform=axs7.transAxes)
  axs7.axes.xaxis.set_ticklabels([])
  axs7.axes.set_xbound(97,303)
  axs7.axes.set_ybound(-1.1, 1.1)

  axs8.set_xticks(xlabel)
  axs8.plot(t_eval[warmtrain_pts-1:warmtrain_pts+plottime_pts-1]-warmup,x[2,warmtrain_pts-1:warmtrain_pts+plottime_pts-1],linewidth=t_linewidth)
  axs8.plot(t_eval[warmtrain_pts-1:warmtrain_pts+plottime_pts-1]-warmup,x_test[2,0:plottime_pts],linewidth=t_linewidth,color='r')
  axs8.set_ylabel('I')
  axs8.text(-.155,0.87,'h)', ha='left', va='bottom',transform=axs8.transAxes)
  axs8.set_xlabel('time')
  axs8.axes.set_xbound(97,303)
  axs8.axes.set_ybound(-2.5, 2.5)

do_plot(**do_nvar())
plt.savefig('predict-dscroll.png')

npts=10
train_nrmse_v=np.zeros(npts)
test_nrmse_v=np.zeros(npts)
n_fp1_diff_v=np.zeros(npts)
n_fp2_diff_v=np.zeros(npts)
n_fp0_diff_v=np.zeros(npts)
p_fp1_norm_v=np.zeros((npts, 3))
p_fp2_norm_v=np.zeros((npts, 3))
p_fp0_norm_v=np.zeros((npts, 3))
for i in range(npts):
  r = do_nvar(warmup=1. + 100. * i)
  train_nrmse_v[i] = r['train_nrmse']
  test_nrmse_v[i] = r['nrmse']
  n_fp1_diff_v[i] = r['n_fp1_diff']
  n_fp2_diff_v[i] = r['n_fp2_diff']
  n_fp0_diff_v[i] = r['n_fp0_diff']
  p_fp1_norm_v[i] = r['p_fp1_norm']
  p_fp2_norm_v[i] = r['p_fp2_norm']
  p_fp0_norm_v[i] = r['p_fp0_norm']

print('\n ridge regression parameter: '+str(r['ridge_param'])+'\n')
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
