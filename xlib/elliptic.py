import numpy as np
from pdb import set_trace
from numpy import sin, cos, exp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
#from numba import njit, jit

############################
# Plots
############################
# Plot figure without boundary


def plot1(u, h, number):
    x = np.arange(0.0+h, 1.0, h)
    y = np.arange(0.0+h, 1.0, h)
    X, Y = np.meshgrid(x, y)  # grid of point
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(X, Y, u, rstride=1, cstride=1,
                           cmap=cm.RdBu, linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.show()
    plt.savefig("figure_"+str(number)+".png")
# Plot figure with boundary


def plot2(u, h, n):
    uu = np.zeros((n+1, n+1))
    uu[1:n, 1:n] = u  # be careful with the indices in python
    x = np.arange(0.0, 1.0+h, h)
    y = np.arange(0.0, 1.0+h, h)
    X, Y = np.meshgrid(x, y)  # grid of point
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(X, Y, uu, rstride=1, cstride=1,
                           cmap=cm.RdBu, linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig("figure_boundary_"+str(n)+".png")

# @njit()


def gauss_seidel(A, b, err, nlimit):
    n = len(b)
    x = np.zeros((n,), dtype=float)
    x[:] = b[:]
    for i in range(nlimit):
        x_old = np.copy(x)
        for j in range(n):
            v = 0.0
            for k in range(j):
                v += A[j, k]*x[k]
            for k in range(j+1, n):
                v += A[j, k]*x[k]

            x[j] = (b[j] - v)/A[j, j]

        if np.sqrt(np.linalg.norm(x - x_old))/n < err:
            return x
############################
# FIRST EXAMPLE
############################
# Create matrix Lh


def f_example(n):
    h = 1/n
    I = np.eye(n-1)
    U = np.eye(n-1, k=1)
    L = np.eye(n-1, k=-1)
    T = 20*np.eye(n-1) - 4*np.eye(n-1, k=1) - 4*np.eye(n-1, k=-1)
    B = -4*np.eye(n-1) - np.eye(n-1, k=1) - np.eye(n-1, k=-1)
    Lh = (np.kron(I, T) + np.kron(L, B) + np.kron(U, B)) / (h**2) / 6

    # Adjust fh too
    T2 = 8*np.eye(n-1) + np.eye(n-1, k=1) + np.eye(n-1, k=-1)
    Rh = (np.kron(I, T2) + np.kron(U, I) + np.kron(L, I))/12

    def adjust_bc(n, rhs, g, f):
        a = n*n/6

        eps = np.arange(1, n)/n

        for j in range(n-1):
            rhs[j] += f((j+1)*h, 0)/12
            rhs[j - (n-1)] += f((j+1)*h, 1)/12
            rhs[j*(n-1)] += f(0, (j+1)*h)/12
            rhs[(j+1)*(n-1)-1] += f(1, (j+1)*h)/12

        rhs[:n-1] += -((B @ g(eps, 0).reshape((n-1, 1)))*a).ravel()
        rhs[-(n-1):] += -((B @ g(eps, 1).reshape((n-1, 1)))*a).ravel()
        rhs[::n-1] += -((B @ g(0, eps).reshape((n-1, 1)))*a).ravel()
        rhs[n-2::n-1] += -((B @ g(1, eps).reshape((n-1, 1)))*a).ravel()

        rhs[0] += g(0, 0)*a
        rhs[n-2] += g(1, 0)*a
        rhs[-(n-1)] += g(0, 1)*a
        rhs[-1] += g(1, 1)*a

    # f : fonte em (-Du = f)
    def f(x, y): return -200*x**2*(x - 1) - 200*y*(3*x - 1)*(y - 1) - 4
    #f = lambda x,y: -2*x*y*(1 - x)*(1 - y)*cos(x + y) + x*y*(1 - x)*sin(x + y) + x*y*(1 - y)*sin(x + y) - x*(1 - x)*(1 - y)*sin(x + y) - x*exp(x*y) - y*(1 - x)*(1 - y)*sin(x + y) - y*exp(x*y)
    # g : BC em u = g em Gamma
    def g(x, y): return + x**2 + y**2
    #g = lambda x,y: exp(x*y)

    def f_mesh(n, f):
        x = np.arange(1, n)/n
        y = np.arange(1, n)/n
        X, Y = np.meshgrid(x, y)  # grid of point
        return f(X, Y)

    rhs = f_mesh(n, f).reshape((n-1)**2)
    rhs = Rh @ rhs

    adjust_bc(n, rhs, g, f)

    # resultado do valor numérico
    # u = np.linalg.solve(Lh, rhs) #<===para validar o método
    u = gauss_seidel(Lh, rhs, h**4, n**4)
    u = u.reshape((n-1, n-1))  # vetor convertido em matriz

    # CONVERGENCIA
    # solução exata - na prática testamos o modelos comparando-o com uma solução conhecida
    def u_ex(x, y): return 100*x**2*y*(1 - x)*(1 - y) + x**2 + y**2
    #u_ex= lambda x,y : x*y*(1 - x)*(1 - y)*sin(x + y) + exp(x*y)

    u_exact = f_mesh(n, u_ex)

    ################################################################################
    udiff = u - u_exact
    ################################################################################

    number = n
    try:
        plot1(u, h, number)
    # plot1(u_ex,h,number)
    except:
        pass

    # plot1(u_exact,h)
    ##plot1(udiff, h)

    return np.linalg.norm(udiff.ravel(), np.inf)
