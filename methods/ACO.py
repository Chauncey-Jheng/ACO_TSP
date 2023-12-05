import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import Array
import ctypes

class ACO:
    def __init__(self, ant_count = 100, alpha = 1, beta = 2,
                 rho = 0.1, Q = 1, MAX_iter = 200, use_CPUs = 10) -> None:
        '''
        ant_count: The total number of the ants
        alpha: The weight index factor of the pheromone
        beta:  The weight index factor of the heuristic function
        rho: Volatilization rate
        Q: The amount of pheromone
        MAX_iter: Maximum iteration number
        '''
        self.ant_count = ant_count
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.MAX_iter = MAX_iter
        self.use_CPUs = use_CPUs
    
    def input_data(self, city_pos:np.ndarray, distance_table:np.ndarray):
        '''
        input the data, include city_pos and distance_table.
        city_pos: NDArray(n*2),
        distance_table: NDArray(n*n)
        '''
        self.city_pos = city_pos
        self.distance_table = distance_table
        self.city_count = len(city_pos)
        self.pheromone_table = np.ones((self.city_count, self.city_count))
        self.path_best = np.zeros((self.MAX_iter, self.city_count), dtype = int)
        self.distance_best = np.zeros(self.MAX_iter)
        self.reciprocal_dist = 1.0 / self.distance_table

    def serial_iteration(self, method = "cycle"):
        '''
        The main body of the aco.
        It will excute serially.
        the method can be choosed in the following list:

        "density": incre_pheromone = Q

        "quantity": incre_pheromone = Q / dist(i,j)

        "cycle": incre_pheromone = Q / L(k)

        "constant": incre_pheromone = dist(i,j) * Q / L(k)

        Return the path_best list combined by the best path in each iteration,
        and the distance_best list combinde by the shortest distance according to the best path in each iteration.
        '''
        candidate = np.zeros((self.ant_count, self.city_count), dtype = int)
        print("ACO serial iteration progress:")
        for now_iter in tqdm(range(self.MAX_iter)):
            # select the initial city
            if self.ant_count <= self.city_count:
                candidate[:, 0] = np.random.permutation(range(self.city_count))[:self.ant_count]
            else:
                m = self.ant_count
                n = 1
                while m > self.city_count:
                    candidate[self.city_count*(n-1):self.city_count*n, 0] = np.random.permutation(range(self.city_count))[:]
                    m -= self.city_count
                    n += 1
                candidate[self.city_count*(n-1):self.ant_count, 0] = np.random.permutation(range(self.city_count))[:m]
                length = np.zeros(self.ant_count)
            
            # select the path
            for i in range(self.ant_count):
                # remove the initial city
                unvisit = list(range(self.city_count))
                visit = candidate[i, 0]
                unvisit.remove(visit)
                for j in range(1, self.city_count):
                    # compute the probabilities of transfer to unvisit cities.
                    probability_trans = np.zeros(len(unvisit))
                    for k in range(len(unvisit)):
                        probability_trans[k] = np.power(self.pheromone_table[visit][unvisit[k]], self.alpha) * \
                                               np.power(self.reciprocal_dist[visit][unvisit[k]], self.beta)
                    # roulette wheel selection
                    cumsum_prob_trans = (probability_trans / sum(probability_trans)).cumsum()
                    cumsum_prob_trans -= np.random.rand()
                    k = unvisit[list(cumsum_prob_trans > 0).index(True)]

                    candidate[i, j] = k
                    unvisit.remove(k)
                    length[i] += self.distance_table[visit][k]
                    visit = k
                length[i] += self.distance_table[visit][candidate[i, 0]]

            # update the best path
            if now_iter == 0:
                self.distance_best[now_iter] = length.min()
                self.path_best[now_iter] = candidate[length.argmin()].copy()
            else:
                if length.min() > self.distance_best[now_iter -1]:
                    self.distance_best[now_iter] = self.distance_best[now_iter -1]
                    self.path_best[now_iter] = self.path_best[now_iter -1].copy()
                else:
                    self.distance_best[now_iter] = length.min()
                    self.path_best[now_iter] = candidate[length.argmin()].copy()

            # update the pheromone
            incre_pheromone = np.zeros((self.city_count, self.city_count))
            if method == "quantity":
                for i in range(self.ant_count):
                    for j in range(self.city_count - 1):
                        incre_pheromone[candidate[i, j]][candidate[i, j+1]] += \
                            self.Q / self.distance_table[candidate[i, j]][candidate[i,j + 1]]
                    incre_pheromone[candidate[i, j+1]][candidate[i, 0]] += \
                        self.Q / self.distance_table[candidate[i, j+1]][candidate[i, 0]]
            elif method == "density":
                for i in range(self.ant_count):
                    for j in range(self.city_count - 1):
                        incre_pheromone[candidate[i, j]][candidate[i, j+1]] += self.Q
                    incre_pheromone[candidate[i, j+1]][candidate[i, 0]] += self.Q
            elif method == "cycle":
                for i in range(self.ant_count):
                    for j in range(self.city_count - 1):
                        incre_pheromone[candidate[i, j]][candidate[i, j+1]] += self.Q / length[i]
                    incre_pheromone[candidate[i, j+1]][candidate[i, 0]] += self.Q / length[i]
            elif method == "constant":
                for i in range(self.ant_count):
                    for j in range(self.city_count - 1):
                        incre_pheromone[candidate[i, j]][candidate[i, j+1]] += \
                            self.distance_table[candidate[i, j]][candidate[i, j+1]] * self.Q / length[i]
                    incre_pheromone[candidate[i, j+1]][candidate[i, 0]] += \
                        self.distance_table[candidate[i, j+1]][candidate[i, 0]] * self.Q / length[i]
            else:
                raise ValueError("ACO method does not support!")
            self.pheromone_table = (1 - self.rho) * self.pheromone_table + incre_pheromone
        
        return self.path_best, self.distance_best

    def select_path(self, i):
        '''
        private methods, used in the parallel_iteration, don't use it outside.
        '''
        # remove the initial city
        unvisit = list(range(self.city_count))
        visit = candidate[i * self.city_count + 0]
        unvisit.remove(visit)
        for j in range(1, self.city_count):
            # compute the probabilities of transfer to unvisit cities.
            probability_trans = np.zeros(len(unvisit))
            for k in range(len(unvisit)):
                probability_trans[k] = np.power(self.pheromone_table[visit][unvisit[k]], self.alpha) * \
                                        np.power(self.reciprocal_dist[visit][unvisit[k]], self.beta)
            # roulette wheel selection
            cumsum_prob_trans = (probability_trans / sum(probability_trans)).cumsum()
            cumsum_prob_trans -= np.random.rand()
            k = unvisit[list(cumsum_prob_trans > 0).index(True)]
            # fill the candidate table
            candidate[i * self.city_count + j] = k
            unvisit.remove(k)
            length[i] += self.distance_table[visit][k]
            visit = k
        length[i] += self.distance_table[visit][candidate[i * self.city_count + 0]]

    def init_pool_processes(self, the_length, shared_array):
        '''
        private methods, used in the parallel_iteration, don't use it outside.
        '''
        global length
        length = the_length
        global candidate
        candidate = shared_array

    def parallel_iteration(self, method = "cycle"):
        '''
        The main body of the aco.
        It will excute parallelly.
        the method can be choosed in the following list:

        "density": incre_pheromone = Q

        "quantity": incre_pheromone = Q / dist(i,j)

        "cycle": incre_pheromone = Q / L(k)

        Return the path_best list combined by the best path in each iteration,
        and the distance_best list combinde by the shortest distance according to the best path in each iteration.
        '''
        shared_array = Array(ctypes.c_int64, self.ant_count * self.city_count)
        for i in range(len(shared_array)):
            shared_array[i] = 0
        shared_np_array = np.frombuffer(shared_array.get_obj(), dtype=np.int64)
        candidate = shared_np_array.reshape((self.ant_count, self.city_count))
        print("ACO parallel iteration progress:")
        for now_iter in tqdm(range(self.MAX_iter)):
            # select the initial city
            if self.ant_count <= self.city_count:
                candidate[:, 0] = np.random.permutation(range(self.city_count))[:self.ant_count]
            else:
                m = self.ant_count
                n = 1
                while m > self.city_count:
                    candidate[self.city_count*(n-1):self.city_count*n, 0] = np.random.permutation(range(self.city_count))[:]
                    m -= self.city_count
                    n += 1
                candidate[self.city_count*(n-1):self.ant_count, 0] = np.random.permutation(range(self.city_count))[:m]
            
            length = Array(ctypes.c_double, self.ant_count)
            for i in range(len(length)):
                length[i] = 0
            
            # select the path
            p = Pool(processes=self.use_CPUs, initializer=self.init_pool_processes, initargs=(length, shared_array,))
            p.map(self.select_path, list(range(self.ant_count)))
            p.close()    
            p.join()
            length = np.array(length)
            # update the best path
            if now_iter == 0:
                self.distance_best[now_iter] = length.min()
                self.path_best[now_iter] = candidate[length.argmin()].copy()
            else:
                if length.min() > self.distance_best[now_iter -1]:
                    self.distance_best[now_iter] = self.distance_best[now_iter -1]
                    self.path_best[now_iter] = self.path_best[now_iter -1].copy()
                else:
                    self.distance_best[now_iter] = length.min()
                    self.path_best[now_iter] = candidate[length.argmin()].copy()

            # update the pheromone
            incre_pheromone = np.zeros((self.city_count, self.city_count))
            if method == "quantity":
                for i in range(self.ant_count):
                    for j in range(self.city_count - 1):
                        incre_pheromone[candidate[i, j]][candidate[i, j + 1]] += \
                            self.Q / self.distance_table[candidate[i, j]][candidate[i,j + 1]]
                    incre_pheromone[candidate[i, j + 1]][candidate[i, 0]] += \
                        self.Q / self.distance_table[candidate[i, j + 1]][candidate[i, 0]]
            elif method == "density":
                for i in range(self.ant_count):
                    for j in range(self.city_count - 1):
                        incre_pheromone[candidate[i, j]][candidate[i, j + 1]] += self.Q
                    incre_pheromone[candidate[i, j + 1]][candidate[i, 0]] += self.Q
            elif method == "cycle":
                for i in range(self.ant_count):
                    for j in range(self.city_count - 1):
                        incre_pheromone[candidate[i, j]][candidate[i, j + 1]] += self.Q / length[i]
                    incre_pheromone[candidate[i, j + 1]][candidate[i, 0]] += self.Q / length[i]
            elif method == "constant":
                for i in range(self.ant_count):
                    for j in range(self.city_count - 1):
                        incre_pheromone[candidate[i, j]][candidate[i, j+1]] += \
                            self.distance_table[candidate[i, j]][candidate[i, j+1]] * self.Q / length[i]
                    incre_pheromone[candidate[i, j+1]][candidate[i, 0]] += \
                        self.distance_table[candidate[i, j+1]][candidate[i, 0]] * self.Q / length[i]
            else:
                raise ValueError("ACO method does not support!")
            self.pheromone_table = (1 - self.rho) * self.pheromone_table + incre_pheromone
        
        return self.path_best, self.distance_best
