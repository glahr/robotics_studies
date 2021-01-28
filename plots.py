import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#Data for plotting
# t = (start point, end point, step)
def main():
    t = np.arange(0.0, 2.0, 0.01)
    s = 1.0 + np.sin(2.0*np.pi*t)
    fig, ax = plt.subplots()
    ax.plot(t, s)

    ax.set(xlabel='time', ylabel='voltage', title='simple')
    ax.grid()

    #fig.savefig("test.png")
    plt.show()

def displacement_plot(time, displacement, box):
    title = "stiffness of" + " " + str(box)
    plt.plot(time, displacement, 'b', linewidth=2.0)
    #plt.setp(color='b', linewidth=2.0)
    plt.xlabel('time of iteration')
    plt.ylabel('displacement')
    plt.title(title)

    plt.grid(True)
    plt.savefig(title)
    plt.show()



if __name__ == '__main__':
    main()
