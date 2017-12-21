'''
project-B2.py
Main module for demonstrating results of project-B2
'''

#Import modules
import newton_cotes   #Contains Trapezoidal and Simpson's methods
import montecarlo   #Metropolis algorithm
import matplotlib.pyplot as plt   #Used for plotting
import numpy as np

#Set style options
plt.rc('font',**{'family':'sans-serif','weight':'regular','size':12})

def psi(z):
    '''Wavefunction for a well-localised particle at a position z in space. a(z) is the phase function'''
    return 1/(np.pi)**.25 * np.exp(1j*a(z)) * np.exp(-.5*z**2)
def psi_sq(z):
    '''Wavefunction for a well-localised particle at a position z in space. a(z) is the phase function'''
    return abs(psi(z)**2)
def a(z):
    '''Phase function'''
    return 0

# SAMPLING FUNCTIONS
def uniform(x,domain):
    A = domain  #Normalisation factor
    return 1/A
def linear(x,domain):
    A = -.48  #Suggested parameters
    B = .98
    return A*x + B

#Evaluate Trapezoidal and Simpson's Rules
I_t, err_t, N_t = newton_cotes.extended_trapezoidal(psi_sq,0,2,1e-6)
I_s, err_s, N_s = newton_cotes.extended_simpson(psi_sq,0,2,1e-6)

#Parameters for the Metropolis methods
step_size = .1
epsilon = 5e-4
x0 = np.random.random()*2.

#Evaluate Metropolis Algorithm using a uniform and linear sampling function
I_mc_uniform,err_mc_uniform,N_mc_uniform = montecarlo.metropolis1d(psi_sq,0.,2.,step_size=step_size,epsilon=epsilon,x0=x0,sampling_function=uniform)
I_mc_linear,err_mc_linear,N_mc_linear = montecarlo.metropolis1d(psi_sq,0.,2.,step_size=step_size,epsilon=epsilon,x0=x0,sampling_function=linear)

#Print results
print("***")
print("Trapezoidal rule      : {:.10f} +- {:.10f}, N={}".format(I_t,err_t,N_t))
print("Simpson's Rule        : {:.10f} +- {:.10f}, N={}".format(I_s,err_s,N_s))
print("Metropolis (uniform)  : {:.10f} +- {:.10f}, N={}".format(I_mc_uniform,err_mc_uniform,N_mc_uniform))
print("Metropolis (linear)   : {:.10f} +- {:.10f}, N={}".format(I_mc_linear,err_mc_linear,N_mc_linear))
print("***")

#Plot integrand
xrange=np.linspace(0,2,100)
plt.figure()
xextended = np.arange(-1,3,.01)   #Extend range
plt.plot(xextended,psi_sq(xextended),'black',label='psi_sq(z)')
#Plot 1st two iterations of trapezoidal rule
plt.plot([0,1,2],[psi_sq(0),psi_sq(1),psi_sq(2)],'r-',label='Trapezoidal n=1')
plt.plot([0,.5,1,1.5,2],[psi_sq(0),psi_sq(.5),psi_sq(1),psi_sq(1.5),psi_sq(2)],'b-',label='Trapezoidal n=2')
#Plot domain of integration
plt.axvline(x=0,color='g',linestyle='--')
plt.axvline(x=2,color='g',linestyle='--')
plt.title('psi_sq(z)')
plt.xlabel('z')
plt.ylabel('y')
plt.grid()
plt.legend()