import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy import integrate

T     = 10
w0    = 2*np.pi/T
tvec  = np.linspace(-2*T, 2*T, 1000, endpoint = False)
N     = 50
nvec  = np.arange(-N, N)

def realintegrand(t, f, n, b) :
    return f(t-b)*np.cos(n*w0*t)/T

def imagintegrand(t, f, n, b) :
    return f(t-b)*np.sin(n*w0*t)/T

def compute_integrands(f, n, a, b) :
    realans , errR = integrate.quad(realintegrand, a, T+a, args=(f, n, b))
    imagans , errI = integrate.quad(imagintegrand, a, T+a, args=(f, n, b))
    return realans - 1.0j*imagans
    
def fourier_calculation(t, x, a = 0, b = 0) :
    xnum    = sp.lambdify(t, x)
    fourier = np.vectorize(compute_integrands)
    X       = fourier(xnum, nvec, a, b)
    return X

def reconstruction(X) :
    recon  = np.zeros_like(tvec, dtype=complex)
    count1 = 0

    for count in nvec :
        cexp    = np.exp(1.0j*count*w0*tvec)
        recon   = recon + X[count1]*cexp
        count1  = count1 + 1

    plt.plot(tvec, np.real(recon))
    plt.show()

def linearity (t, x, y) :
    a = 4
    b = 10
    X = Fourier_calculation(t, x)
    Y = Fourier_calculation(t, y)
    aXbY  = a*X + b*Y
    aXbYf = Fourier_calculation(t, a*x+b*y)
    L =  aXbY - aXbYf

    plt.plot(nvec, L)
    plt.show()

def limits_shift (t, x, a) :
    X_a = Fourier_calculation(t, x, a)
    X   = Fourier_calculation(t, x)

    plt.plot(nvec, X_a-X)
    plt.show()

def Conjugate_symmetry (t, x) :
    X = Fourier_calculation(t, x)
    Y = Fourier_calculation(t, np.conj(x))

    plt.plot(nvec, np.imag(Y))
    plt.plot(nvec, np.imag(X))
    plt.show()

def time_shifting (t, x) :
    X = Fourier_calculation(t, x)
    Y = Fourier_calculation(t, x, 0, 5)

    plt.plot(nvec, np.real(X))
    plt.plot(nvec, np.real(Y))
    plt.legend(['non shifted', 'shifted'])
    plt.show()

    plt.plot(nvec, np.imag(X))
    plt.plot(nvec, np.imag(Y))
    plt.legend(['non shifted', 'shifted'])
    plt.show()

def Product_Convolution(t, x, y) :
    X = Fourier_calculation(t, x)
    Y = Fourier_calculation(t, y)
    Z = Fourier_calculation(t, x*y)
    Z_ = []

    for n in range(-N, N, 1) :
        Zn = 0
        for m in range(-N, N, 1) :
            Zn += X[m + 50]*Y[n-m]
        Z_.append(Zn)

    reconstruction(Z)
    reconstruction(Z_)
              

# Fourier run part 

t = sp.symbols("t")
x = sp.Piecewise((t - (t//T)*T, t%T<=T/4), (T/2 + (t//T)*T - t, (t%T>T/4) & (t%T<=3*T/4)), (t-T - (t//T)*T, (t%T>3*T/4) & (t%T<=T)))
y = sp.Piecewise((1, t%T<=T/2), (0, t%T>T/2))


# reconstruction(Fourier_calculation(t, x))
# linearity(t, x, y)
# limits_shift(t, x, 5)
# limits_shift(t, x, 10)
# Conjugate_symmetry(t, y)
# time_shifting(t, x)
# Product_Convolution(t, x, y)
