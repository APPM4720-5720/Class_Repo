"""
    Summary: LU decomposition of Hilbert matrix.
    """


import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.linalg import hilbert, cholesky


def solveLU(L, U, b):
    L = np.array(L, float)
    U = np.array(U, float)
    b = np.array(b, float)
    n = len(L)
    y = np.zeros(n)
    x = np.zeros(n)

    for i in range(n):
        sumj = 0
        for j in range(i):
            sumj += L[i, j]*y[j]
        y[i] = (b[i]-sumj)/L[i, i]

    for i in range(n-1, -1, -1):
        sumj = 0
        for j in range(i+1, n):
            sumj += U[i, j] * x[j]
        x[i] = (y[i]-sumj)/U[i, i]
    return x


def plot2(inp):
    sol = []
    sizes = []
    for k in range(2, inp+1):

        L = cholesky(hilbert(k), lower=True)
        LT = np.transpose(L)
        b = np.dot(hilbert(k), np.ones(k))

        x = solveLU(L, LT, b)
        xNorm = LA.norm(x-1, np.inf)
        sizes.append(float(k))
        sol.append(xNorm)
    return sol, sizes


p = plot2(13)
print(p[0], p[1])


plt.figure()
plt.title("Semilog plot of ")
plt.ylabel('Infinity Norm value')
plt.xlabel('Matrix size')
plt.semilogy(p[1], p[0])
plt.show()
