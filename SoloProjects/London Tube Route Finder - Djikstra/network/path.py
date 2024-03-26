from network.graph import NeighbourGraphBuilder

class PathFinder:
    """
    Task 3: Complete the definition of the PathFinder class by:
    - completing the definition of the __init__() method (if needed)
    - completing the "get_shortest_path()" method (don't hesitate to divide 
      your code into several sub-methods)
    """

    def __init__(self, tubemap):
        """
        Args:
            tubemap (TubeMap) : The TubeMap to use.
        """
        self.tubemap = tubemap
        graph_builder = NeighbourGraphBuilder()
        
        self.graph = graph_builder.build(self.tubemap)
        
        # Feel free to add anything else needed here.
        
        
    def station_checker(self, start_station_name, end_station_name):
        """ Check if the input stations are valid
        
        Args:
            start_station_name (str) : The name of the starting station
            end_station_name (str) : The name of the ending station
            
        Returns:
            boolean [True] : if the stations do not exist a flag is sent back to the main function
            int [start_id] : The corresponding id of the starting station
            int [end_id] : The corresponding id of the ending staion
        """
        # Initialising our check if the start/end stations are real
        start_real = False
        end_real = False
        
        # We extract all IDs and station instances
        for station_id, instance in self.tubemap.stations.items():
            #if the current instance matches a start/end station we set our checks to True and extract IDs
            if instance.name == start_station_name:
                start_real = True
                start_id = str(station_id)
            elif instance.name == end_station_name:
                end_real = True
                end_id = str(station_id)
        
        # If either station is not real then we return a boolean flag to the main function, else we return IDs
        if start_real == False or end_real == False:
            return False
        else:
            return start_id, end_id
        
        
    def dijkstra_initialiser(self, start_id):
        """ Initialising the Dijkstra algorithm
        
        Args:
            start_id (int) : The id of the starting station
                
        Returns:
            dict [distances] : To store the distances it takes to travel to stations
            list [visited] : To store all previously visited stations
            dict [previous_station] : To store the details of all previously visited stations      
        """
        # Setting all distances to infinity as per Dijkstra algorithm
        distances = {station_id: float('inf') for station_id in self.graph} 
        # Set station 1 distance to 0
        distances[start_id] = 0
        # We define our visited list to include the starting station
        visited = [] 
        previous_station = {}
        return distances, visited, previous_station
    

    
    def nearest_station_finder(self, distances, visited):
        """ Finding the closest station
        
        Args:
            distances (dict) : The distances it takes to travel to stations
            visited (list) : A list of all visited stations
            
        returns:
            str (min_distance_station) : The Id of the nearest station
        """
        min_distance_station = None
        for station_id in self.graph:
                # If we haven't visited the station and we do not have a smallest distance/this is the smallest distance we continue
                if station_id not in visited and (min_distance_station is None or distances[station_id] < distances[min_distance_station]):
                    # Setting the new smallest distance
                    min_distance_station = station_id
        return min_distance_station
    
    
    def dijkstra_algorithm(self, min_distance_station, end_id, distances, previous_station, visited):
        """ Main body of path-finding dijkstra algorithm
        
        Args:
            min_distance_station (str) : The id of the nearest station
            end_id (str) : The id of the final station
            start_id (str) : The id of the starting station
            distances (dict) : The distances between stations
            previous_station (dict) : A dictionary containing all the stations in the optimal route
            visited (list) : A list of all visited stations
            
        Returns:
            dict (previous_station) : A filled dictionary containing all stations in the optimal route
        
        """
        visited.append(min_distance_station)  # Append the minimum distance station to stations we have visited
        if min_distance_station == end_id:
                # We break if we have reached our destination
                return True
            
        for neighbour_station, connections in self.graph[min_distance_station].items():
            # For neighbours with multiple connecting lines we find the minimum time travelled
            times = [connection.time for connection in connections]
            travel_time = min(times)
            if distances[min_distance_station] == float('inf'):
                continue  # Skip if the distance is infinity.
            distance = distances[min_distance_station] + travel_time
            # If current time is less than the last connection, we set a new distance and new station on the route
            if distance < distances[neighbour_station]:
                distances[neighbour_station] = distance
                previous_station[neighbour_station] = min_distance_station 
        return previous_station
   
    
    def path_builder(self, start_id, end_id, previous_station):
            """ Building the "path" that will be displayed as the quickest route between stations
            
            Args:
                start_id (int) : The corresponding ID of the starting station
                end_id (int): The corresponding ID of the ending station
                previous_station (dict): A collection of all the stations visited between the start and the end
                
            Returns:
                list [path] : The final route taken between the two stations
            
            """
            # Defining an empty list to store the path
            path = []
            current_station = end_id
            while current_station != start_id:
                # We iterate through our optimal stations backwards in order to construct the route taken
                path.insert(0, self.tubemap.stations[current_station])
                current_station = previous_station[current_station]
            path.insert(0, self.tubemap.stations[start_id])
            return path
        
        
    def get_shortest_path(self, start_station_name, end_station_name):
        """ Find ONE shortest path from start_station_name to end_station_name.
        
        The shortest path is the path that takes the least amount of time.

        For instance, get_shortest_path('Stockwell', 'South Kensington') 
        should return the list:
        [Station(245, Stockwell, {2}), 
         Station(272, Vauxhall, {1, 2}), 
         Station(198, Pimlico, {1}), 
         Station(273, Victoria, {1}), 
         Station(229, Sloane Square, {1}), 
         Station(236, South Kensington, {1})
        ]

        If start_station_name or end_station_name does not exist, return None.
        
        You can use the Dijkstra algorithm to find the shortest path from
        start_station_name to end_station_name.

        Find a tutorial on YouTube to understand how the algorithm works, 
        e.g. https://www.youtube.com/watch?v=GazC3A4OQTE
        
        Alternatively, find the pseudocode on Wikipedia: https://en.wikipedia.org/wiki/Dijkstra's_algorithm#Pseudocode

        Args:
            start_station_name (str): name of the starting station
            end_station_name (str): name of the ending station

        Returns:
            list[Station] : list of Station objects corresponding to ONE 
                shortest path from start_station_name to end_station_name.
                Returns None if start_station_name or end_station_name does not 
                exist.
                Returns a list with one Station object (the station itself) if 
                start_station_name and end_station_name are the same.
        """
        
        valid_stations = self.station_checker(start_station_name, end_station_name)

        valid_stations = False
        if valid_stations == False:
            return None
        else:
            start_id = valid_stations[0]
            end_id = valid_stations[1]

        distances, visited, previous_station = self.dijkstra_initialiser(start_id)
        
        #We loop until we have visited each station
        while len(visited) < len(self.graph):   
            min_distance_station = self.nearest_station_finder(distances, visited)     
            results = self.dijkstra_algorithm(min_distance_station, end_id, distances, previous_station, visited)
            if results == True:
                break
            else:
                previous_station = results
        path = self.path_builder(start_id, end_id, previous_station)        
        return path
    


def test_shortest_path():
    from tube.map import TubeMap
    
    tubemap = TubeMap()
    tubemap.import_from_json("data/london.json")
    
    path_finder = PathFinder(tubemap)
    stations = path_finder.get_shortest_path("Covent Garden", "Green Park")
    print(stations)
    
    station_names = [station.name for station in stations]
    expected = ["Covent Garden", "Leicester Square", "Piccadilly Circus", 
                "Green Park"]
    assert station_names == expected


if __name__ == "__main__":
    test_shortest_path()
