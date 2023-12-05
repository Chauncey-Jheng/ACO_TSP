import numpy as np
import random
import math
class CityData:
    def __init__(self) -> None:
        self.city_name = []
        self.city_pos = []

    def __compute_distance_matrix__(self):
        city_count = len(self.city_name)
        self.Distance = np.zeros((city_count, city_count))
        for i in range(city_count):
            for j in range(city_count):
                if i != j:
                    self.Distance[i][j] = math.sqrt((self.city_pos[i][0] -
                                                self.city_pos[j][0]) ** 2 +
                                               (self.city_pos[i][1] -
                                                self.city_pos[j][1]) ** 2)
                else:
                    self.Distance[i][j] = 9999999

    def get_Chinese_cities(self):
        '''
        Return the 34 Chinese cities names and positions.
        Return format: (list, NDarray(34*2), NDarray(34*34))
        '''
        with open('datas\\ChineseCities.txt','r',encoding="UTF-8") as f:
            lines = f.readlines()
            for line in lines:
                words = line.split('\n')[0]
                words = words.split(",")
                self.city_name.append(words[0])
                self.city_pos.append([float(words[1]),float(words[2])])
        self.__compute_distance_matrix__()
        return self.city_name, np.array(self.city_pos), self.Distance

    def get_my_cities(self, load_file_path:str):
        '''
        Return the n cities names and positions defined in the mycites.txt.
        Return format: (list, NDarray(n*2), NDarray(n*n))
        '''
        with open(load_file_path,"r",encoding="UTF-8") as f:
            lines = f.readlines()
            for line in lines:
                words = line.split("\n")[0]
                words = words.split(",")
                self.city_name.append(words[0])
                self.city_pos.append([float(words[1]),float(words[2])])
        self.__compute_distance_matrix__()
        return self.city_name, np.array(self.city_pos), self.Distance

    def get_random_cities_float(self, n:int, MaxLongitude: float, MaxLatitude: float, nd=2):
        '''
        Return the n cities names and positions generated randomly. 
        The position coordinates will be float.
        nd represent nd decimal places of the float coordinates.
        Return format: (list, NDarray(n*2), NDarray(n*n))
        '''
        for i in range(n):
            self.city_name.append(str(i))
            self.city_pos.append([round(random.uniform(0,MaxLongitude), nd),
                                  round(random.uniform(0,MaxLatitude), nd)])
        self.__compute_distance_matrix__()
        return self.city_name, np.array(self.city_pos), self.Distance
    
    def get_random_cities_int(self, n:int, MaxLongitude: int, MaxLatitude: int):
        '''
        Return the n cities names and positions generated randomly. 
        The position coordinates will be int.
        Return format: (list, NDarray(n*2), NDarray(n*n))
        '''
        for i in range(n):
            self.city_name.append(str(i + 1))
            self.city_pos.append([random.randint(0,MaxLongitude),random.randint(0,MaxLatitude)])
        self.__compute_distance_matrix__()
        return self.city_name, np.array(self.city_pos), self.Distance
       
    def print_cities_data(self):
        for i in range(len(self.city_name)):
            print(self.city_name[i] + "," + ",".join([str(j) for j in self.city_pos[i]]))

    def save_to_file(self, save_file_path:str):
        '''
        Save the n cities to the file named by the given save_file_path.
        The city data format is as follows:
        (cityname 1),(longtitude 1),(latitude 1)
        (cityname 2),(longtitude 2),(latitude 2)
        ...
        (cityname n),(longtitude n),(latitude n)
        '''
        newlines = []
        for i in range(len(self.city_name)):
            newline = ""
            newline += self.city_name[i] + "," + ",".join([str(j) for j in self.city_pos[i]]) + "\n"
            newlines.append(newline)
        with open(save_file_path, "w", encoding="UTF-8") as f:
            f.writelines(newlines)

def main():
    city_data = CityData()
    # city_data.get_Chinese_cities()
    # city_data.print_cities_data()
    city_data.get_random_cities_int(50, 100, 100)
    city_data.print_cities_data()
    city_data.save_to_file("datas\\50_cities.txt")

if __name__ == "__main__":
    main()