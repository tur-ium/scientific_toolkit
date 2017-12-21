# -*- coding: utf-8 -*-
"""
NEWTON-COTES INTEGRATION TOOLS
Includes: Trapezoidal rule

@author: Artur Donaldson

"""
import time   #Used for timing
import numpy as np   #Used for validation
import matplotlib.pyplot as plt   #Used for visual validation
import sys   #Used for float info

#Define quantities useful for floating-point number comparison
machine_accuracy = sys.float_info.epsilon   #Smallest representable number
machine_precision = 2**-23   #Precision to be used in comparisons of floating point numbers

def trapezoidal_step(f,a,b,S,n):
    '''Calculates the nth estimate, S, of the integral of `f` from `a` to `b`\
    using the trapzoidal rule'''
    #SANITATION:
    #None required as this is a local method and should always be called from \
    # another function with proper sanitation. Sanitation could reduce efficiency
    #MAIN METHOD:
    if n == 0:
        return .5*(b-a)*(f(a)+f(b))
    else:
        npoints = 2**(n-1)
        stepsize = (b-a)/npoints  #Distance between new points to be evaluated
        
        #Iteration 1:
        i = 1
        
        x = a + .5*stepsize   #First point
        
        A = f(x) #sum of f(x) on the new points to be added
        
        while i<npoints:
            #Go through the remaining points to be added
            x += stepsize
            A += f(x)
            i += 1
    return .5 * (S + (b-a)*A/npoints)

def extended_trapezoidal(f,a,b,epsilon,maxN=21,validation=False):
    '''Use the extended trapezoidal rule algorithm to integrate `f` from \
`a` to `b` to a precision `epsilon`
    Parameters
    ---
    f: function
        Integrand to be evaluated
    a: float
        Lower bound
    b: float
        Upper bound
    epsilon: 0 < float < 1
        Precision required (determined in terms of relative change between \
estimates (I_new-I_old)/I_old )
    maxN: int (default: 21)
        Maximum number of iterations to perform. Max. number of points which \
will be evaluated is 2^(maxN-1). Default (maxN=21) is to use just over \
a million points.
    validation: True/False
        Perform validation on the result
    Returns
    ---
    I: float
        The value of the integral calculated
    rel_err: float
        The associated relative error
    totN: int
        The total number of points evaluated
    '''
    #SANITATION:
    if not callable(f):  #Check that f is a function
        raise TypeError('f is not a function')
    #Cast a,b,epsilon to float objects to avoid arithmetic issues
    a = float(a)   
    b = float(b)
    epsilon = float(epsilon)
    if abs(epsilon) >= 1:   #Check precision
        raise ValueError("Precision must be between 0 and 1")
    
    #MAIN METHOD:
    start0 = time.time()
    #Iteration 0
    I_old = trapezoidal_step(f,a,b,0,0)   #First estimate
    
    times = list()  #Keep track of runtime per iteration
    Is = list()   #Keep track of estimates of integral
    rel_errs = list()
    
    n=1   #Iteration number
    rel_err = 1   #Set rel. error to max value
    while rel_err > epsilon:
        #Convergence criterion has not been reached
        start = time.time()   #Time each iteration individually to check for inefficiency
        
        I_new = trapezoidal_step(f,a,b,I_old,n)   #Update estimate
        if abs(I_new-I_old) > machine_precision and abs(I_old) > machine_precision:
            rel_err = abs((I_new-I_old)/I_old)   #Update estimated error
        else:
            print('WARNING: Machine accuracy encountered or estimated integral is 0. Previous estimate: {}, current estimate: {}'.format(I_old, I_new))
            rel_err = 1   #Reset relative error to maximum possible value
        I_old = I_new   #Discard old value of integral
        
        
        #VALIDATION: TIMING, CONVERGENCE, COMPARISON WITH NAIVE METHOD.
        
        #Record runtime for iteration
        end = time.time()
        time_it = end-start
        npoints=2**(n-1)
        times.append(time_it/npoints)
        
        if validation:            
            #Record estimate to find convergence
            Is.append(I_new)
            rel_errs.append(rel_err)
            
            print('Iteration {}: Runtime per point: {:.5} s, estimate: {:.10g}, rel. err: {:.10%}'.format(n,time_it/npoints,I_new,rel_err))
            
        n+=1
        if n > maxN:
            #Break loop as maximum number of points to be evaluated has been exceeded
            print("WARNING: Maximum number of iterations ({}) exceeded in `extended_trapeziodal_step`".format(maxN))
            break
    end0 = time.time()
    if validation:
        #Time information
        plt.figure()
        plt.plot(range(1,n),times)
        plt.title('Runtime (s) vs. iteration number')
        plt.ylabel('Runtime per point (s)')
        plt.xlabel('Iteration number')
        #Convergence of integral
        plt.figure()
        plt.plot(range(1,n),Is)
        plt.title('TRAPEZOIDAL: Estimated integral I vs. iteration number')
        plt.ylabel('I')
        plt.xlabel('Iteration number')
        #Error
        plt.figure()
        plt.plot(range(1,n),rel_errs)
        plt.title('TRAPEZOIDAL: Relative error vs. iteration number')
        plt.ylabel('rel_err')
        plt.xlabel('Iteration number')
        #Plot integrand and region of integration
#        plt.figure()
#        xextended = np.arange(-2,4,.01)
#        plt.plot(xextended,f(xextended))
#        plt.axvline(x=a,color='g',linestyle='--')
#        plt.axvline(x=b,color='g',linestyle='--')
#        plt.title('TRAPEZOIDAL: Integrand {}, green bars indicate region of integration'.format(f))
#        plt.xlabel('x')
#        plt.ylabel('f(x)')
    total_runtime=end0-start0
    print('\nTRAPEZOIDAL: Total runtime: {:.5g} s, Av. runtime per point: {:.5g} s'.format(total_runtime,sum(times)/len(times)))
    print('Iterations performed: {}'.format(n))
    totN = 2**n + 1
    print('Total number of points sampled: {}'.format(totN))
    return I_new,rel_err,totN

def extended_simpson(f,a,b,epsilon,maxN=21,validation=False):
    '''use the Extended Simpson's Rule algorithm to integrate `f` from \
    `a` to `b` to a precision `epsilon`
    Parameters
    ---
    f: function
        Integrand to be evaluated
    a: float
        Lower bound
    b: float
        Upper bound
    epsilon: 0 < float < 1
        Precision required (determined in terms of relative change between \
estimates (I_new-I_old)/I_old )
    maxN: int (default: 21)
        Maximum number of iterations to perform. Max. number of points which \
will be evaluated is 2^(maxN-1). Default (maxN=21) is to use just over \
a million points.
    validation: True/False
        Perform validation on the result
    Returns
    ---
    S: float
        The value of the integral calculated
    rel_err: float
        The associated relative error
    totN: int
        The total number of points used
        '''
        
    start0 = time.time()
    
    #Piggyback on trapezoidal method by comparing successive evaluations T_1 and T_2
    T_1 = trapezoidal_step(f,a,b,0,0)   #Estimate from trapezoidal 
    T_2 = trapezoidal_step(f,a,b,T_1,1)
    
    S = (4*T_2 - T_1)/3
    
    times = list()  #Keep track of runtime per iteration
    Ss = list()   #Keep track of estimates of integral
    rel_errs = list()
        
    n=1   #Iteration number
    rel_err = 1   #Set rel. error to max value
    while rel_err > epsilon:
        #Convergence criterion has not been reached
        start = time.time()   #Time each iteration individually to check for inefficiency
        
        T_1 = T_2   #Update old estimate
        T_2 = trapezoidal_step(f,a,b,T_1,n+1)   #Estimate using twice n+1 th iteration of trap
        
        #Check for 0/0 division
        if abs(T_2-T_1) > machine_precision and abs(T_1) > machine_precision:
            rel_err = abs((T_2-T_1)/T_1)   #Update estimated error
        else:
            print('WARNING: Machine accuracy encountered or estimated integral is 0. Previous estimate: {}, current estimate: {}'.format(T_1,T_2))
            rel_err = 1   #Reset relative error to maximum possible value
        
        #Update estimate
        S = (4*T_2 - T_1)/3
        
        n+=1  #Update count
        #VALIDATION: TIMING, CONVERGENCE, COMPARISON WITH NAIVE METHOD.
    
        #Record runtime for iteration
        end = time.time()
        time_it = end-start
        npoints=2**(n-1)
        times.append(time_it/npoints)
        
        if validation:
            #Record estimate to find convergence
            Ss.append(S)
            rel_errs.append(rel_err)
            
            print('Iteration {}: Runtime per point: {:.5} s, estimate: {:.10g}, rel. err: {:.10%}'.format(n,time_it/npoints,S,rel_err))
        if n > maxN:
            #Break loop as maximum number of points to be evaluated has been exceeded
            print("WARNING: Maximum number of iterations ({}) exceeded in `extended_trapeziodal_step`".format(maxN))
            break
    
    end0 = time.time()
    if validation:
        #Time information
        plt.figure()
        plt.plot(range(1,n),times)
        plt.title('Runtime (s) vs. iteration number')
        plt.ylabel('Runtime per point (s)')
        plt.xlabel('Iteration number')
        #Convergence of integral
        plt.figure()
        plt.plot(range(1,n),Ss)
        plt.title('SIMPSON: Estimated integral I vs. iteration number')
        plt.ylabel('I')
        plt.xlabel('Iteration number')
        #Error
        plt.figure()
        plt.plot(range(1,n),rel_errs)
        plt.title('SIMPSON: Relative error vs. iteration number')
        plt.ylabel('rel_err')
        plt.xlabel('Iteration number')
        print('Iterations performed: {}'.format(n))#
        #PLot integrand and region of integration
#        plt.figure()
#        xextended = np.arange(-2,4,.01)
#        plt.plot(xextended,f(xextended))
#        plt.axvline(x=a,color='g',linestyle='--')
#        plt.axvline(x=b,color='g',linestyle='--')
#        plt.title('Integrand, green bars indicate region of integration'.format(f))
#        plt.xlabel('x')
#        plt.ylabel('f(x)')
    total_runtime = end0-start0
    print('SIMPSON: Total runtime: {:.5g} s, Av. runtime per point: {:.5g}'.format(total_runtime,sum(times)/len(times)))
    print('Iterations performed: {}'.format(n))
    totN = 2**n+1
    print('Total number of points sampled: {}'.format(totN))
    return S, rel_err,totN