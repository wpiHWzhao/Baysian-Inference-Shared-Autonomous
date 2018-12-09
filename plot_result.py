import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

style.use('fivethirtyeight')

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

# graph_data_ = open('example.txt', 'r').read()
# lines_ = graph_data_.split('\n')

def animate(i):
    graph_data = open('trajectory_new.csv', 'r').read()
    lines = graph_data.split('\n')
    # xs = []
    ys = []
    for line in lines:
    # for k in range(i):
        if len(line) > 1:
            # x, y = lines[k].split(',')
            linedata = line.split(',')
            # xs.append(float(x))
            ys.append(float(linedata[2]))
    ax1.clear()
    ax1.plot(ys)
    plt.xlim(0, len(lines))
    plt.ylim(0.0, 1.0)

frames = 5000
ani = animation.FuncAnimation(fig, animate, interval=1000, frames = 5000)
plt.show()