import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def sin():
    fig, ax = plt.subplots()
    xdata, ydata = [], []
    ln, = plt.plot([], [], 'ro', animated=True)

    def init():
        ax.set_xlim(0, 2 * np.pi)
        ax.set_ylim(-1, 1)
        return ln,

    def update(frame):
        xdata.append(frame)
        ydata.append(np.sin(frame))
        ln.set_data(xdata, ydata)
        return ln,

    # noinspection PyTypeChecker
    ani = FuncAnimation(fig, update, frames=np.linspace(0, 2 * np.pi, 128),
                        init_func=init, blit=True)
    plt.show()


def sin_dot():
    """
    animation example 2
    author: Kiterun
    """

    fig, ax = plt.subplots()
    x = np.linspace(0, 2 * np.pi, 200)
    y = np.sin(x)
    l = ax.plot(x, y)
    dot, = ax.plot([], [], 'ro')

    def init():
        ax.set_xlim(0, 2 * np.pi)
        ax.set_ylim(-1, 1)
        return l

    def gen_dot():
        for i in np.linspace(0, 2 * np.pi, 200):
            newdot = [i, np.sin(i)]
            yield newdot

    def update_dot(newd):
        dot.set_data(newd[0], newd[1])
        return dot,

    ani = FuncAnimation(fig, update_dot, frames=gen_dot, interval=100, init_func=init)
    ani.save('sin_dot.gif', writer='imagemagick', fps=30)

    plt.show()


if __name__ == '__main__':
    pass