from datas.load_data import CityData
from utils import plot
from methods.ACO import ACO

def experiment_Chinese():
    # Get the data
    city_data = CityData()
    city_name, city_pos, distance_table = city_data.get_Chinese_cities()

    # Load the model
    aco = ACO(ant_count=50)
    aco.input_data(city_pos, distance_table)
    path_best, distance_best = aco.serial_iteration("cycle")

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

    plot.plot_path(x, y, path, "results\\ant_cycle_path_Chinese_city.png")
    plot.plot_dist(distance_best, "results\\ant_cycle_distance_Chinese_city.png")

def experiment_oliver30():
    # Get the data
    city_data = CityData()
    city_name, city_pos, distance_table = city_data.get_my_cities("datas\\oliver30.txt")

    # Load the model
    aco = ACO(ant_count=100, alpha=1, beta=6, rho=0.2, Q=50, MAX_iter=200)
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

    plot.plot_path(x, y, path, "results\\ant_constant_path_oliver30_1.png")
    plot.plot_dist(distance_best, "results\\ant_constant_distance_oliver30_1.png")

if __name__ == "__main__":
    # experiment_Chinese()
    experiment_oliver30()