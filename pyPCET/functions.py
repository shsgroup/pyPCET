import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import splrep, BSpline
import types
import functools
from . import units

def Morse(r, r0, De, beta):
    return De*(1 - np.exp(-beta*(r-r0)))**2

def Morse_inverted(r, r0, De, beta):
    return De*(1 - np.exp(beta*(r-r0)))**2

def Gaussian(x, x0, sigma2):
    return 1/np.sqrt(2*np.pi*sigma2)*np.exp(-(x-x0)**2/(2*sigma2))

def poly6(x,a,b,c,d,e,f,g):
    return a*x**6+b*x**5+c*x**4+d*x**3+e*x**2+f*x+g

def poly8(x,a,b,c,d,e,f,g,h,i):
    return a*x**8+b*x**7+c*x**6+d*x**5+e*x**4+f*x**3+g*x**2+h*x+i

#====================================================
# functions that returns a callable function
# which only takes one arguments to use in pyPCET
#====================================================

def gen_Morse(r0, De, beta):
    """
    This function generate a callable function of a Morse potential 
    """
    def func(r):
        return De*(1 - np.exp(-beta*(r-r0)))**2
    return func

def gen_Morse_inverted(r0, De, beta):
    """
    This function generate a callable function of an inverted Morse potential 
    """
    def func(r):
        return De*(1 - np.exp(beta*(r-r0)))**2
    return func

def gen_double_well(De1, De2, beta1, beta2, R0, Delta, VPT0, smooth='poly8', s=8, NGrid=500):
    """
    This function generate a callable function for the double well potential 
    This is done by diagonalizing a 2x2 matrix, where the diagonal elements are two Morse potentials
    and the coupling term is a Gaussian function which has maximun value VPT0 at the crossing point of the two Morse potentials
    The parameter smooth determine how the calculated potential can be smoothened, possible choises are 'poly8' and 'BSpline'
    When using 'BSpline', s is a parameter used to controll the smoothness of the final function, increasing s will smoothen the function
    but may also remove some important features, s should be as small as possible
    s is not used if smooth == 'poly8' or 'poly6'
    """
    r = np.linspace(-1.2*R0, 1.2*R0, NGrid)
    ED = Morse(r, -R0/2, De1, beta1)
    EA = Morse_inverted(r, R0/2, De2, beta2) + Delta

    roots = find_roots(r, ED-EA)
    if len(roots) != 3:
        raise RuntimeError("These input parameters lead to a single-well potential. Please use a different set of parameters. ")
    else:
        VPT = VPT0*Gaussian(r, roots[1], R0*R0)

    E1 = 0.5*(EA + ED - np.sqrt((EA-ED)**2+4*VPT**2))
    E2 = 0.5*(EA + ED + np.sqrt((EA-ED)**2+4*VPT**2))

    switch = np.heaviside(r-roots[0], 0*r) + np.heaviside(roots[2]-r, 0*r) - 1

    E = switch*E1 + (1-switch)*E2

    # due to avoided crossing, E(r) may have some bumps
    # the following codes are used to smoothen E(r)
    if smooth == 'poly8':
        # fit the potential to 8th-order polynomial
        fitted_E = fit_poly8(r, E)
        def double_well_func(rr):
            return fitted_E(rr) - np.min(fitted_E(r))
    elif smooth == 'poly6':
        # fit the potential to 6th-order polynomial
        fitted_E = fit_poly6(r, E)
        def double_well_func(rr):
            return fitted_E(rr) - np.min(fitted_E(r))
    elif smooth == 'BSpline':
        # an alternative way, use spline fit 
        tck = splrep(r, E, s=s)
        def double_well_func(rr):
           return BSpline(*tck)(rr) - np.min(BSpline(*tck)(r))
    else:
        raise ValueError("Unrecogonized input, please set smooth = 'poly8', 'poly6', or 'BSpline'. ")

    return double_well_func 

def fit_poly6(xdata, ydata):
    """
    This function fit the given data to a 6th-order polynomial
    """
    aa,bb,cc,dd,ee,ff,gg = curve_fit(poly6, xdata, ydata-np.min(ydata))[0]
    xdata_tmp = np.linspace(np.min(xdata),np.max(xdata),500)
    ydata_tmp = poly6(xdata_tmp,aa,bb,cc,dd,ee,ff,gg)
    aa,bb,cc,dd,ee,ff,gg = curve_fit(poly6, xdata_tmp, ydata_tmp-np.min(ydata_tmp))[0]
    def func(x):
        return aa*x**6+bb*x**5+cc*x**4+dd*x**3+ee*x**2+ff*x+gg
    return func

def fit_poly8(xdata, ydata):
    """
    This function fit the given data to a 8th-order polynomial
    """
    aa,bb,cc,dd,ee,ff,gg,hh,ii = curve_fit(poly8, xdata, ydata-np.min(ydata))[0]
    xdata_tmp = np.linspace(np.min(xdata),np.max(xdata),500)
    ydata_tmp = poly8(xdata_tmp,aa,bb,cc,dd,ee,ff,gg,hh,ii)
    aa,bb,cc,dd,ee,ff,gg,hh,ii = curve_fit(poly8, xdata_tmp, ydata_tmp-np.min(ydata_tmp))[0]
    def func(x):
        return aa*x**8+bb*x**7+cc*x**6+dd*x**5+ee*x**4+ff*x**3+gg*x**2+hh*x+ii
    return func

def bspline(xdata, ydata, s=5e-3):
    """
    This function spline the given data using BSpline
    """
    tck = splrep(xdata, ydata, s=s)
    xdata_tmp = np.linspace(np.min(xdata),np.max(xdata),500)
    def func(x):
       return BSpline(*tck)(x) - np.min(BSpline(*tck)(xdata_tmp))
    return func

#====================================================
# some useful functions
#====================================================

def find_roots(xdata, ydata):
    # find all roots for y(x) with in the data range
    xo = xdata[0]
    yo = ydata[0]
    roots = []
    for xi, yi in zip(xdata[1:], ydata[1:]):
        if np.sign(yi) != np.sign(yo):
            if np.abs(yi) < np.abs(yo):
                roots.append(xi)
            else:
                roots.append(xo)
        else:
            pass
        xo = xi
        yo = yi

    return roots

def isnumber(dat):
    return (isinstance(dat, int) or isinstance(dat, float))

def isarray(dat):
    return (isinstance(dat, tuple) or isinstance(dat, list) or isinstance(dat, np.ndarray))

def copy_func(f):
    """Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)"""
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__,
                           argdefs=f.__defaults__,
                           closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g
