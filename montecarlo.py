# -*- coding: utf-8 -*-
"""
MONTE CARLO INTEGRATION TOOLS

@author: Artur Donaldson
"""

# Import Libraries
import matplotlib.pyplot as plt   #For plotting
import matplotlib.ticker as ticker
import numpy as np
import time   #For timing

#Set style options
plt.rc('font',**{'family':'sans-serif','weight':'regular','size':12})

def uniform(x,domain):
    '''Uniform pdf (default sampling function for metropolis algorithm)'''
    A = domain  #Normalisation factor
    return 1/A
# GAUSSIAN FUNCTION (for step-size distribution)
def gauss(x,std=.25,mean=0):
    '''Normalised Gaussian distribution with standard deviation `std`, and mean `mean` (default:0)'''
    return 1/(std*np.sqrt(2*np.pi)) * np.exp(-.5*(x-mean)**2*std**-2)

def metropolis1d(f,a,b,epsilon,step_size,x0=None,sampling_function=uniform,show_trace=False,\
                 check_convergence=False):
    '''Use the Metropolis algorithm to integrate a 1d function on the domain a to b,\
to the specified accuracy `epsilon` and using a `step_size` as the standard deviation\
of the Gaussian step distribution. Also returns number of points used and\
plots a histogram of the set of points {x} sampled so as to determine if the \
sampling matches `sampling_function`

Parameters
---
f: function
    Integrand to be evaluated
a: float
    Lower bound
b: float
    Upper bound
epsilon: 0 < float < 1
    Precision required (2*standard deviation of the set of estimated values)
x0: float (default:None)
    Starting point. If None, choose one from a uniform pdf between a and b
sampling_function: function (default: uniform)
    Importance sampling function.
show_trace: boolean (default: False)
    Show a trace of the points sampled (x) against the number of iterations
check_convergence: boolean (default: False)
    Show a plot of the estimate of the integral at each iteration 

Returns
---
I_new: float
    Estimate of the value of the integral
rel_err: float
    Relative error
N: int
    Total number of function evaluations performed
---
    '''
    print("METROPOLIS:")
    #SANITATION:
    if not callable(f):  #Check that f is a function
        raise TypeError('f is not a function')
    #Cast a,b,epsilon to float objects to avoid arithmetic issues
    a = float(a)   
    b = float(b)
    epsilon = float(epsilon)
    if abs(epsilon) >= 1:   #Check precision
        raise ValueError("Precision must be between 0 and 1")
    if b < a:
        raise Exception('The upper limit of the integral, b, is smaller than the lower, a')
    if epsilon < 2**-53: #Precision of float32
        raise Exception('Precision is too high - might result in issues due to limited machine precision')
    
    #MAIN METHOD
    # Keep lists for validation purposes
    qs = list()   #List of values of the integrand
    xs = list()   #List of points in the domain sampled
    rel_errs = list()   #List of relative errors
    Is = list()   #List of estimates of the integral
    domain = b-a
    
    #Keep track of sum of q values, sum of squares and count
    S = 0   
    S_squares = 0
    N = 1   #Number of tries - to avoid infinite loops
    
    P = sampling_function
    P_args = [domain]   #Arguments for the P function
    
    if not x0:
        x = np.random.random()*domain+a
    else:
        x = x0
    xs.append(x)
    qs.append(f(x))
    
    Nmin = 2e3 #Minimum number of iterations to calculate
    Nmax = int(1e9)   #Max. number of iterations to calculate
    
    rel_err = 1    #We have maximal uncertainty about the value of the function
    I_new = qs[0]   #Keep track of the current and previous estimates of the integral
    
    start = time.time()
    while rel_err > epsilon and N <= Nmax:
        I_old = I_new
        
        #Proposal function: gaussian using a user-defined step-size
        x_new = np.random.normal(x,step_size)
        
        #Apply periodic boundary conditions 
        if x_new < a:
            #New point is below the domain of f, so transalte it
            x_new = domain + x_new
        if x_new > b:
            #New point is above the domain of f, so translate it
            x_new = x_new - domain
        
        if P(x_new,*P_args) >= P(x,*P_args):
            x = x_new
            p = P(x,*P_args)
            q = f(x)/p
        else:
            #The sampling function is lower at x_new than x
            #Accept it with probability P(x_new)/P(x)
            r = np.random.random()   #Uniform deviate on the interval [0,1]
            prob_ratio = P(x_new,*P_args)/P(x,*P_args)
            if r < prob_ratio:
                #Accept the new step
                x = x_new
                p = P(x,*P_args)
                q = f(x)/p
            else:
                #Don't accept the new step, stay at the same point
                q = f(x)

        
        #Increase iteration count
        N += 1
        
        # Update lists and sums
        xs.append(x)
        qs.append(q)
            
            
        if N > Nmin:
            #After 'burn in'
            S+=q
            S_squares += q**2
    
            if N > Nmax:
                print('WARNING: Maximum number of iterations exceeded')
            
            q_mean = S/N    # Mean value of Q
            I_new = q_mean   #Estimate of the integral of f over the domain is the mean value of Q
            Is.append(I_new)            
        
            #Uncertainty
            #Calculate variance of q
            q_variance = (S_squares - q_mean**2) / (N-1)

            I_sd = N ** -.5 * q_variance ** .5   # standard deviation of I
            rel_err = abs(.5*I_sd/I_new) #Use a confidence interval of 2 standard deviations (5% sig. level)
            rel_errs.append(rel_err)
            #Print the error every so often for code verification
            if N % 100000 == 0:
                print('Err: {}'.format(rel_err))
    
    #TIMING
    end =time.time()
    print('Runtime: {:.3g} s'.format(end-start))
    
    #VALIDATION AND VISUALISATION
    #Plot the path traced:
    if show_trace:
        plt.figure()
        plt.plot(range(len(xs)),xs)
        plt.xlabel('Step number')
        plt.ylabel('x')
        plt.title('Trace of random walk across domain')
        plt.show()
   
    #Plot histogram of where we have sampled (should be the sample function)
    plt.figure()
    plt.hist(xs,normed=True,label='Observed distr')   #Normalize for comparison with sampling function
    
    #Plot the sampling function
    xrange = np.linspace(a,b,100)
    plt.plot(xrange,[P(xx,*P_args) for xx in xrange],label='Sampling Function')
    plt.legend()
    plt.ylabel('Probability')
    plt.xlabel('x')
    plt.title('Sampling comparison')
    plt.show()
    
    print('Number of points used in estimate: {}'.format(len(Is)))
    print('Number of iterations: {}'.format(N))
    
    #Check for convergence
    if check_convergence:
        fig = plt.figure(figsize=(10,4))
        ax = fig.add_subplot(111)
        ax.plot(range(len(rel_errs)),rel_errs)
        ax.set_yscale('log')
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1g'))
        plt.title('Estimated integral I vs. iteration number')
        plt.ylabel('I')
        plt.xlabel('Iteration number')
    
    return I_new, rel_err, N