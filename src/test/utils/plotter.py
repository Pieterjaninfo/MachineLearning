import matplotlib.pyplot as plt


def lineplot(x, y, options=None):
    plt.plot(x, y)
    plt.show()

def scatterplot(x, y, options=None):
    plt.plot(x, y)
    plt.show()

if __name__ == 'builtins':
    scatterplot([1,2,3], [5,6,7])