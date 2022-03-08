import matplotlib.animation as animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Lorenz_RK4 import *

poinc_x = []
poinc_y = []
poinc_z = []
count = []


def plot(i):
    for i in range(0, n-1, 1):
        if term[i, 1] < 3.5 and term[i, 1] > 3:
            poinc_x.append(float(term[i, 0]))
            poinc_y.append(float(term[i, 1]))
            poinc_z.append(float(term[i, 2]))
            count.append(float(i))

    return np.array([poinc_x, poinc_y, poinc_z, count])


p = plot(2000)


# 2D plot
plt.figure()
plt.title("Poncare Map of Lorenz Equations")
plt.xlabel('x value')
plt.ylabel('y value')
plt.scatter(p[0], p[1])
plt.show()


# animate
fig = plt.figure('Lorenz System of ODEs', figsize=(12, 8), dpi=100)
ax = fig.gca(projection='3d')


def animate(i):  # Anim function; original version set default grid to off
    ax.view_init(20, i/3)  # sets rotation by i/3 for each step
    ax.clear()
    ax.set(facecolor='white')
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.plot(term[:i, 0], term[:i, 1], term[:i, 2],
            color='purple', lw=0.5, label='RK4 approximation (xyz)')  # plots x,y,z
    ax.scatter(p[0], p[1], p[2],
               color='green', lw=0.5, label='RK4 approximation (xy)')  # plots x,y
    ax.legend(fontsize=15)


ani = animation.FuncAnimation(
    fig, animate, np.arange(2000), interval=5, repeat=False)  # anim object; 2k frame, 5ms delay
plt.show()
