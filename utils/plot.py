import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'STSong'

def plot_path(x:list, y:list, path:list, savefig="results\\result_path.png"):
    '''
    Plot the best path.
    '''
    plt.figure()
    plt.plot(x, y, '-o')
    plt.title("Best path graph")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    for i in range(len(path)):
        plt.annotate(path[i], xy=(x[i], y[i]), xytext=(x[i] + 0.3, y[i] + 0.3))
    plt.savefig(savefig)
    
def plot_dist(distance:list, savefig="results\\result_dist.png"):
    '''
    Plot the distance iteration condition.
    '''
    plt.figure()
    plt.plot(range(1, len(distance) + 1), distance)
    plt.title("Distance iteration graph")
    plt.xlabel("Number of iterations")
    plt.ylabel("Distance value")
    plt.savefig(savefig)