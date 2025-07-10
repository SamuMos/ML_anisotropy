from matplotlib import pyplot as plt 
import numpy as np
from scipy.optimize import curve_fit
import multiprocessing as mp
from time import time
from functools import partial
from inspect import signature

#Functions for fitting
def log_squared(x,a,b,c):
    return a + b * np.log(x) + c * np.log(x)**2

def D_log_squared(x,a,b,c):
    return b / x + 2 * c * np.log(x) / x


def businger(x,a,b,c):
    return a + b * x + c * np.log(x)

def D_businger(x,a,b,c):
    return b + c / x


def quadratic(x,a,b,c,d):
    return a + b * x + c * x**2 + d * np.log(x)

def D_quadratic(x,a,b,c,d):
    return b  + 2 * c * x + d / x


def full(x,a,b,c,d,e):
    return a + b * x + c * x**2 + d * np.log(x) + e * np.log(x)**2

def D_full(x,a,b,c,d,e):
    return b + 2 * c * x + d / x + 2 * e * np.log(x) / x

def poly2(x,a,b,c):
    return a + b * x + c * x ** 2

def D_poly2(x,a,b,c):
    return b  + c * x * 2

#functions that include roughness length

def businger_rough(x, b, c, z0):
    return b * x + c * np.log(x/z0)

def D_businger_rough(x, b, c, z0):
    return b + c / x


def quadratic_rough(x, b, c, d, z0):
    return b * x + c * x ** 2 + d * np.log(x/z0)

def D_quadratic_rough(x, b, c, d, z0):
    return b + 2 * c * x + d / x

#----------------------------------------------------------

def Fit(Function, X, Y, N_par, bounds):
    '''
    Function to be handled by multiprocessing,
    Fits only if the number of parameters is smaller than the number of points, returns null array otherwise
    '''
    #need initial parameters guess to have n_par when using partial, might not work with certain boundaries, to be later modified
    p0 = np.full(N_par, 1)
    #fit profiles
    if len(X) >= N_par:
        param, covar = curve_fit(Function, X, Y, bounds=bounds, p0=p0)
    else :
        param = np.zeros(N_par)

    return param

#--------------------------------------------------

#Main functions for gradients

def Gradient(profile, heights, Function, Derivative, return_fit = False, bounds = (-np.inf,np.inf)):
    '''
    Calculates gradient of the profile by analytical fitting and derivative
    '''

    length = len(profile)
    
    #prepare data: list of arrays without nans, each with its own length
    Y = [[] for i in range(length)]
    X = [[] for i in range(length)]
    for i in range(length):
        row = profile[i,:]
        notnan = ~np.isnan(row)
        Y[i] = np.array(row[notnan])
        X[i] = np.array(heights[notnan])

    #get number of free parameters
    # (avoiding to count the 'only keyboard params' from functools.partial)
    # of the fitting function (-1 as the X entry)
    N_par = sum(1 for param in signature(Function).parameters.values() if param.kind == param.POSITIONAL_OR_KEYWORD) - 1
    N_par_D = sum(1 for param in signature(Derivative).parameters.values() if param.kind == param.POSITIONAL_OR_KEYWORD) - 1
    if N_par != N_par_D:
        raise ValueError(
            'Function and the derivative don\'t have the same N_par: ({} , {})'.format(N_par, N_par_D))

    #check boundaries size is correct
    for bound in bounds:
        if np.size(bound) > 1 and np.size(bound) != N_par:
            raise ValueError(
                    'The provided boundaries {} have different size than the number of free parameters {} of the fitting function'.format(bounds,N_par))


    #fit profiles - PARALLEL COMPUTED
    fit_res=[]
    
    pool = mp.Pool(mp.cpu_count())
    fit_res = pool.starmap(Fit, [(Function, X[i], Y[i], N_par, bounds) for i in range(length)])
    pool.close()
    
    #convert list to np array
    fit_res = np.array(fit_res)

    # calculate gradient
    grad = np.full((length, len(heights)), np.nan)
    for i in range(length):
        grad[i] = Derivative(heights, *fit_res[i])

    #return gradient
    if return_fit :
        return fit_res,grad
    else: return grad


def MultiFunc_Gradient(profile, heights, Functions, Derivatives, func_index, return_fit = False, bounds = None):
    '''
    Calculates gradient of the profile by analytical fitting and differentiation
    Uses different functions as provided, indexed by func_index
    '''
    N_funcs = len(Functions)

    #check
    if not (N_funcs == len(Derivatives)):
        raise ValueError('The sizes of the function arrays don\'t match')
    if (np.invert(np.isin( func_index, np.arange(N_funcs)))).any():
        raise ValueError('func_index has values different than 0,..,n_funcs-1')


    # get number of free parameters
    # (avoiding to count the 'only keyboard params' from functools.partial)
    # of the fitting functions (-1 as the X entry)
    N_par = np.zeros(N_funcs, dtype=int)
    N_par_D = np.zeros(N_funcs, dtype=int)
    for i in range(N_funcs):
        N_par[i] = sum(
            1 for param in signature(Functions[i]).parameters.values()
            if param.kind == param.POSITIONAL_OR_KEYWORD) - 1
        N_par_D[i] = sum(
            1 for param in signature(Derivatives[i]).parameters.values()
            if param.kind == param.POSITIONAL_OR_KEYWORD) - 1
        if N_par[i] != N_par_D[i]:
            raise ValueError('Function and the derivative don\'t have the same N_par: ({} , {})'.format(N_par[i], N_par_D[i]))

    #initialize boundary if not provided
    if bounds == None:
        bounds = [(-np.inf,np.inf) for i in range(N_funcs)]

    # check if boundaries size is correct
    for i in range(len(bounds)):
        for bound in bounds[i]:
            if np.size(bound) > 1 and np.size(bound) != N_par[i]:
                raise ValueError(
                    'The provided boundaries {} have different size than the number of free parameters {} of the fitting functions'.format(bounds,N_par))
        
    length = len(profile)

    # prepare data: list of arrays without nans, each with its own length
    Y = [[] for i in range(length)]
    X = [[] for i in range(length)]
    for i in range(length):
        row = profile[i, :]
        notnan = ~np.isnan(row)
        Y[i] = np.array(row[notnan])
        X[i] = np.array(heights[notnan])

    # fit profiles - PARALLEL COMPUTED
    fit_res = []

    pool = mp.Pool(mp.cpu_count())

    fit_res = pool.starmap(Fit,
                               [(Functions[func_index[i]], X[i], Y[i], N_par[func_index[i]], bounds[func_index[i]])
                                for i in range(length)])
    pool.close()

    # calculate gradient
    grad = np.full((length,len(heights)), np.nan)
    for i in range(length):
        # if N_par[func_index[i]] != np.size(fit_res[i]):
        #raise ValueError('At index {}, N_par_D is {} but fit_res contains {}'.format(i, N_par[func_index[i]], fit_res[i]))
        grad[i] = Derivatives[func_index[i]] (heights, *fit_res[i])


    # return gradient
    if return_fit:
        return fit_res, grad
    else:
        return grad

