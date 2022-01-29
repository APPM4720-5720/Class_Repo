""" Author: Tyler Reiser
    Date Modified: January 21, 2022 @ 8:49 MST

    Summary: 3D model of the Lorenz ODE system approximated using the 4th-order Runge-Kutta method.
"""


import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

x = 1.  # initial condition can be changed for different results
y = 5.
z = 10.

t0 = 0.
tf = 100.
dt = 0.008  # time step sufficiently small
tmax = 100.
vector_t = np.arange(t0, tmax, dt)
n = len(vector_t)

rho = 28.  # check wiki page for info of rho depencancy/interesting values
sigma = 10.
beta = 8./3.

# initialize
k1 = np.array([0., 0., 0.])
k2 = np.array([0., 0., 0.])
k3 = np.array([0., 0., 0.])
k4 = np.array([0., 0., 0.])


# Three functions for Lorenz Equations
def equation_1(t, x, y, z):
    return sigma*(y - x)


def equation_2(t, x, y, z):
    return rho*x - y - x*z


def equation_3(t, x, y, z):
    return x*y - beta*z


def RK4(t, x, y, z, equation_1, equation_2, equation_3, dt):

    k1[0] = dt*equation_1(t, x, y, z)
    k1[1] = dt*equation_2(t, x, y, z)
    k1[2] = dt*equation_3(t, x, y, z)

    k2[0] = dt*equation_1(t + dt/2., x + k1[0]/2., y + k1[1]/2., z + k1[2]/2.)
    k2[1] = dt*equation_2(t + dt/2., x + k1[0]/2., y + k1[1]/2., z + k1[2]/2.)
    k2[2] = dt*equation_3(t + dt/2., x + k1[0]/2., y + k1[1]/2., z + k1[2]/2.)

    k3[0] = dt*equation_1(t + dt/2., x + k2[0]/2., y + k2[1]/2., z + k2[2]/2.)
    k3[1] = dt*equation_2(t + dt/2., x + k2[0]/2., y + k2[1]/2., z + k2[2]/2.)
    k3[2] = dt*equation_3(t + dt/2., x + k2[0]/2., y + k2[1]/2., z + k2[2]/2.)

    k4[0] = dt*equation_1(t + dt, x + k3[0], y + k3[1], z + k3[2])
    k4[1] = dt*equation_2(t + dt, x + k3[0], y + k3[1], z + k3[2])
    k4[2] = dt*equation_3(t + dt, x + k3[0], y + k3[1], z + k3[2])

    x = x + (1./6.)*(k1[0] + 2.*k2[0] + 2.*k3[0] + k4[0])
    y = y + (1./6.)*(k1[1] + 2.*k2[1] + 2.*k3[1] + k4[1])
    z = z + (1./6.)*(k1[2] + 2.*k2[2] + 2.*k3[2] + k4[2])

    return np.array([x, y, z])


term = np.zeros((n, 3))
term[0, 0] = x  # convective intensity, check WolframAlpha page for info on these
term[0, 1] = y  # temp difference btwn ascending/descending currents
term[0, 2] = z  # dif in vertical temp


for i in range(n - 1):  # loop the stages
    term[i+1, :] = RK4(vector_t[i], term[i, 0], term[i, 1],
                       term[i, 2], equation_1, equation_2, equation_3, dt)


fig = plt.figure('Lorenz System of ODEs', figsize=(12, 8), dpi=100)
ax = fig.gca(projection='3d')



def animate(i):  # Anim function; original version set default grid to off
    ax.view_init(20, i/3) # sets rotation by i/3 for each step
    ax.clear()
    ax.set(facecolor='white')
    ax.plot(term[:i, 0], term[:i, 1], term[:i, 2],
            color='purple', lw=0.5, label='RK4 approximation (xyz)')  # plots x,y,z
    ax.plot(term[:i, 0], term[:i, 1],
            color='green', lw=0.5, label='RK4 approximation (xy)')  # plots x,y
    ax.legend(fontsize=15)


ani = animation.FuncAnimation(
    fig, animate, np.arange(2000), interval=5, repeat=False)  # anim object; 2k frame, 5ms delay
plt.show()
