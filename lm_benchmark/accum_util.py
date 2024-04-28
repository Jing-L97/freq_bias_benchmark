"""
util functions for accum model
"""

import math
import numpy as np
from scipy.special import comb
from scipy.stats import norm

# function definitions useful for estimating probabilities of generations in the accumulator model

def p_miss(p,n):
    """Computes the probability of observing zero instances of an event with probability p after n draws
    Application: probability of not seeing a word with probability p in a corpus of size n words
    """
    return (1-p)**n


def p_obs_k_e(k,p,n):
    """
    Computes the probability of observing k instances of an event with probability p after n draws
    Exact formula using Bernouilli coefficients; crashes for k>50 and large n
    prob= C(n,k) * p^k * (1-p)^(n-k)
    """
    prob=np.nan
    if (k<50) or (k>n-50) or (n<1000000):
        prob=comb(n,k)*(p**k)*((1-p)**(n-k))
    return prob

def p_obs_k_p(k,p,n):
    """
    Approximates the probability of observing k instances of an event with probability p after n draws
    Poisson approximation, crashes for k>150; valid only for p small, n large, np moderate
    prob= (l^k) * e^-l / n!   with l=n*p
    """
    lambd=p*n
    prob=np.nan
    #print(lambd,k)
    if (k<150) and (np.log10(lambd)*k<300):
        prob=(lambd**k)*np.exp(-lambd)/math.factorial(k)
    return prob

def p_obs_k_ps(k,p,n):
    """
    Approximates the probability of observing k instances of an event with probability p after n draws
    Poisson approximation with Stirling approximation for factorial, crashes for k>800; valid only for p small, n large, np moderate
    prob= ((l/k*e)^k) * e^-l / sqrt(2k*pi)   with l=n*p
    """
    lambd=p*n
    prob=np.nan
    if (k>0) and (np.log10(lambd/k*np.e)*k<300):
        prob=((lambd/k*np.e)**k)*np.exp(-lambd)/np.sqrt(2*k*np.pi)
    return prob

def phi(mu,sd,x):
    """Cumulative function for a gaussian of mean mu and standard deviation sd"""
    return norm.cdf((x-mu)/sd)

def p_obs_k_g(k,p,n):
    """
    Approximates the probability of observing k instances of an event with probability p after n draws
    Gaussian approximation, never crashes; valid only for np>5 and n(1-p)>5
    prob==phi(k+0.5)-phi(k-0.5); where phi(x)=unitnormalCDF((x-mu)/sd), mu=n*p, sd=np.sqrt(n*p*(1-p))
    """
    mu=n*p
    sd=np.sqrt(n*p*(1-p))
    return phi(mu,sd,k+0.5)-phi(mu,sd,k-0.5)


def p_obs_k(k,p,n):
    """Computes the probability of observing k instances of an event with probability p after n draws
    Application: probability of seeing a word k times, assuming it has probability p, in a corpus of size n words
    Uses various approximations to avoid crashing, and being approx correct
    """

    prob=p_obs_k_e(k,p,n) # exact bernouilli
    if np.isnan(prob):
          prob=p_obs_k_p(k,p,n) # Poisson Approx
    if np.isnan(prob):
          prob=p_obs_k_ps(k,p,n) # Poisson Stirling Approx
    if np.isnan(prob):
          prob=p_obs_k_g(k,p,n) # Gaussian Approx
    return prob


def p_obs_less_than_k(k,p,n):
    """Computes the probability of observing strictly less than k instances of an event with probability p after n draws
    Application: probability of seeing a word k times, assuming it has probability p, in a corpus of size n words
    Uses various approximations to avoid crashing, and being approx correct
    """
    pless=0
    for i in range(k):
           pless+=p_obs_k(i,p,n)
    return pless


def p_obs_k_list(k,p,n):
    """function used to debug"""
    return {'exact':p_obs_k_e(k,p,n),'Poisson':p_obs_k_p(k,p,n),'PoissonStir':p_obs_k_ps(k,p,n),'Gaussian':p_obs_k_g(k,p,n),'total':p_obs_k(k,p,n)}

