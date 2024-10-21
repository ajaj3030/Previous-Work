class NeighbourGraphBuilder:
    """
    Task 2: Complete the definition of the NeighbourGraphBuilder class by:
    - completing the "build" method below (don't hesitate to divide your code 
      into several sub-methods, if needed)
    """

    def __init__(self):
        pass


    def valid_tube_check(self, connections, stations, lines):
        if len(connections) != 406 or len(stations) != 302 or len(lines) != 13:
            return False
        else:
            return True

    def neighbour_finder(self, stations, connections, graph):
        for station_key, station_instance in stations.items():
                connections_for_station = {}    
                for station_id, other_station_instance in stations.items():
                    if station_id == station_key:
                        continue  # Skip the same station
                    connection_list = [connection for connection in connections if station_instance in connection.stations and other_station_instance in connection.stations]
                    if connection_list:
                        connections_for_station[station_id] = connection_list    
                graph[station_key] = connections_for_station
        return graph
        

    def build(self, tubemap):
        """ Builds a graph encoding neighbouring connections between stations.

        ----------------------------------------------

        The returned graph should be a dictionary having the following form:
        {
            "station_A_id": {
                "neighbour_station_1_id": [
                                connection_1 (instance of Connection),
                                connection_2 (instance of Connection),
                                ...],

                "neighbour_station_2_id": [
                                connection_1 (instance of Connection),
                                connection_2 (instance of Connection),
                                ...],
                ...
            }

            "station_B_id": {
                ...
            }

            ...

        }

        ----------------------------------------------

        For instance, knowing that the id of "Hammersmith" station is "110",
        graph['110'] should be equal to:
        {
            '17': [
                Connection(Hammersmith<->Barons Court, District Line, 1),
                Connection(Hammersmith<->Barons Court, Piccadilly Line, 2)
                ],

            '209': [
                Connection(Hammersmith<->Ravenscourt Park, District Line, 2)
                ],

            '101': [
                Connection(Goldhawk Road<->Hammersmith, Hammersmith & City Line, 2)
                ],

            '265': [
                Connection(Hammersmith<->Turnham Green, Piccadilly Line, 2)
                ]
        }

        ----------------------------------------------

        Args:
            tubemap (TubeMap) : tube map serving as a reference for building 
                the graph.

        Returns:
            graph (dict) : as described above. 
                If the input data (tubemap) is invalid, 
                the method should return an empty dict.
        """ 
        
        
        graph = {}
        connections = tubemap.connections
        stations = tubemap.stations
        lines = tubemap.lines
        
        if self.valid_tube_check(connections, stations, lines):  
            graph = self.neighbour_finder(stations, connections, graph)
            return graph
        
        else:
            return graph

def test_graph():
    from tube.map import TubeMap
    tubemap = TubeMap()
    tubemap.import_from_json("data/london.json")
    

    graph_builder = NeighbourGraphBuilder()
    graph = graph_builder.build(tubemap)

    print(graph)

if __name__ == "__main__":
    test_graph()
