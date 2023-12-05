from datas.load_data import CityData
from utils import plot
from methods.ACO import ACO

def experiment_Chinese():
    # Get the data
    city_data = CityData()
    city_name, city_pos, distance_table = city_data.get_Chinese_cities()

    # Load the model
    aco = ACO(ant_count=200)
    aco.input_data(city_pos, distance_table)
    path_best, distance_best = aco.serial_iteration("constant")

    # plot the result
    path = [city_name[i] for i in path_best[-1]]
    path.append(path[0])
    print("The best path is", path)
    print("The shorest distance is", distance_best[-1])

    x = []
    y = []
    for i in range(len(path_best[-1])):
        x.append(city_pos[int(path_best[-1][i])][0])
        y.append(city_pos[int(path_best[-1][i])][1])
    x.append(x[0])
    y.append(y[0])

    plot.plot_path(x, y, path, "results\\Chinese_city_ant_constant_path.png")
    plot.plot_dist(distance_best, "results\\Chinese_city_ant_constant_distance.png")

def experiment_serial(dataset:str, method:str, ant_count=100, alpha=1, beta=6, rho=0.2, Q=50, MAX_iter=200, use_CPUs=10):
    # Get the data
    city_data = CityData()
    city_name, city_pos, distance_table = city_data.get_my_cities("datas\\"+dataset+".txt")

    # Load the model
    aco = ACO(ant_count, alpha, beta, rho, Q, MAX_iter, use_CPUs)
    aco.input_data(city_pos, distance_table)
    path_best, distance_best = aco.serial_iteration(method)

    # plot the result
    path = [city_name[i] for i in path_best[-1]]
    path.append(path[0])
    print("The best path is", path)
    print("The shortest distance is", distance_best[-1])

    with open("results\\result.log", "a+") as f:
        f.write("ant_"+ method + "_shortest_distance_" + dataset + " is "+ str(distance_best[-1]) + "\n")

    x = []
    y = []
    for i in range(len(path_best[-1])):
        x.append(city_pos[int(path_best[-1][i])][0])
        y.append(city_pos[int(path_best[-1][i])][1])
    x.append(x[0])
    y.append(y[0])

    plot.plot_path(x, y, path, "results\\"+ dataset +"ant_"+ method + "_path_" + ".png")
    plot.plot_dist(distance_best, "results\\"+ dataset +"ant_"+ method + "_distance_" +".png")

if __name__ == "__main__":
    experiment_Chinese()

    # datasets = ["oliver30","dantzig42","eil51","berlin52","st70","pr107","tsp225"]
    # methods = ["quantity","density","cycle","constant"]
    # for i in datasets:
    #     for j in methods:
    #         experiment_serial(i,j)