import json
import math as m
from tube.components import Line, Station, Connection

class TubeMap:
    """
    Task 1: Complete the definition of the TubeMap class by:
    - completing the "import_from_json()" method

    Don't hesitate to divide your code into several sub-methods, if needed.

    As a minimum, the TubeMap class must contain these three member attributes:
    - stations: a dictionary that indexes Station instances by their id 
      (key=id (str), value=Station)
    - lines: a dictionary that indexes Line instances by their id 
      (key=id, value=Line)
    - connections: a list of Connection instances for the TubeMap 
      (list of Connections)
    """

    def __init__(self):
        self.stations = {}  # key: id (str), value: Station instance
        self.lines = {}  # key: id (str), value: Line instance
        self.connections = []  # list of Connection instances


    def station_builder(self, station):
        # TO DO - DOC STRING
        id_data = station['id']
        name_data = station['name']
        zone_data = float(station['zone'])  
        if zone_data != int(zone_data):
            zone_data = set((m.floor(zone_data), m.ceil(zone_data)))
        else:
            zone_data = int(zone_data)
        return {id_data: Station(id_data, name_data, zone_data)}


    def line_builder(self, line):
        # TO DO - DOC STRING
        line_data = line['line']
        name_data = line['name']    
        return {line_data: Line(line_data, name_data)}


    def connection_builder(self, connection):
        # TO DO - DOC STRING
        station1 = connection['station1']
        station2 = connection['station2']
            
        line = connection['line']
        time = int(connection['time'])
    
        return Connection(set((self.stations[station1], self.stations[station2])), self.lines[line], time)
        
        
    def import_from_json(self, filepath):
        """ Import tube map information from a JSON file.
        
        During the import process, the `stations`, `lines` and `connections` 
        attributes should be updated.

        You can use the `json` python package to easily load the JSON file at 
        `filepath`

        Note: when the indicated zone is not an integer (for instance: "2.5"), 
            it means that the station belongs to two zones. 
            For example, if the zone of a station is "2.5", 
            it means that the station is in both zones 2 and 3.

        Args:
            filepath (str) : relative or absolute path to the JSON file 
                containing all the information about the tube map graph to 
                import. If filepath is invalid, no attribute should be updated, 
                and no error should be raised.

        Returns:
            None
        """        
        
        
        try:
            with open(filepath, "r") as jsonfile:
                data = json.load(jsonfile)         
            for station in data['stations']:
                self.stations.update(self.station_builder(station))            
            for line in data['lines']:
                self.lines.update(self.line_builder(line))
            for connection in data['connections']:
                self.connections.append(self.connection_builder(connection))    
            return  
        except FileNotFoundError:
            print("Invalid file!")
            return


def test_import():
    tubemap = TubeMap()
    tubemap.import_from_json("data/london.json")
    
    try:
        # view one example Station
        print(tubemap.stations[list(tubemap.stations)[0]])
        
        # view one example Line
        print(tubemap.lines[list(tubemap.lines)[0]])
        
        # view the first Connection
        print(tubemap.connections[0])
        
        # view stations for the first Connection
        print([station for station in tubemap.connections[0].stations])
        
    except IndexError:
        return


if __name__ == "__main__":
    test_import()
