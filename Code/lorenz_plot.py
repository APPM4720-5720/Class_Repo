import matplotlib.animation as animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Lorenz_RK4 import *


fig = plt.figure('Lorenz System of ODEs', figsize=(12, 8), dpi=100)
ax = fig.gca(projection='3d')


def animate(i):  # Anim function; original version set default grid to off
    ax.view_init(20, i/3)  # sets rotation by i/3 for each step
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
