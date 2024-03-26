import random 

#from convert import CellConverter
from battleship.convert import CellConverter

class Ship:
    """ Represent a ship that is placed on the board.
    """
    def __init__(self, start, end, should_validate=True):
        """ Creates a ship given its start and end coordinates on the board. 
        
        The order of the cells do not matter.
        
        Args:
            start (tuple[int, int]): tuple of 2 positive integers representing
                the starting cell coordinates of the Ship on the board
            end (tuple[int, int]): tuple of 2 positive integers representing
                the ending cell coordinates of the Ship on the board
            should_validate (bool): should the constructor check whether the 
                given coordinates result in a horizontal or vertical ship? 
                Defaults to True.

        Raises:
            ValueError: if should_validate==True and 
                if the ship is neither horizontal nor vertical
        """
        # Start and end (x, y) cell coordinates of the ship
        self.x_start, self.y_start = start
        self.x_end, self.y_end = end

        # make x_start on left and x_end on right
        self.x_start, self.x_end = (
            min(self.x_start, self.x_end), max(self.x_start, self.x_end)
        )
        
        # make y_start on top and y_end on bottom
        self.y_start, self.y_end = (
            min(self.y_start, self.y_end), max(self.y_start, self.y_end)
        )
        
        if should_validate:
            if not self.is_horizontal() and not self.is_vertical():
                raise ValueError("The given coordinates are invalid. "
                    "The ship needs to be either horizontal or vertical.")

        # Set of all (x,y) cell coordinates that the ship occupies
        self.cells = self.get_cells()
        
        # Set of (x,y) cell coordinates of the ship that have been damaged
        self.damaged_cells = set()
    

    def __len__(self):
        return self.length()
        

    def __repr__(self):
        return (f"Ship(start=({self.x_start},{self.y_start}), "
            f"end=({self.x_end},{self.y_end}))")
        

    def is_vertical(self):
        """ Check whether the ship is vertical.
        
        Returns:
            bool : True if the ship is vertical. False otherwise.
        """
        if self.x_start == self.x_end:
            return True 
        else:
            return False
   

    def is_horizontal(self):
        """ Check whether the ship is horizontal.
        
        Returns:
            bool : True if the ship is horizontal. False otherwise.
        """
        if self.y_start == self.y_end:
            return True 
        else:
            return False
    

    def get_cells(self):
        """ Get the set of all cell coordinates that the ship occupies.
        
        For example, if the start cell is (3, 3) and end cell is (5, 3),
        then the method should return {(3, 3), (4, 3), (5, 3)}.
        
        This method is used in __init__() to initialise self.cells
        
        Returns:
            set[tuple] : Set of (x ,y) coordinates of all cells a ship occupies
        """
        occupied = set()
        if self.is_horizontal():
            for x in range(self.x_start, self.x_end+1):
                occupied.add((x, self.y_start))
        elif self.is_vertical():
            for y in range(self.y_start, self.y_end+1):
                occupied.add((self.x_start, y))
        return occupied


    def length(self):
        """ Get length of ship (the number of cells the ship occupies).
        
        Returns:
            int : The number of cells the ship occupies
        """
        return len(self.get_cells())


    def is_occupying_cell(self, cell):
        """ Check whether the ship is occupying a given cell

        Args:
            cell (tuple[int, int]): tuple of 2 positive integers representing
                the (x, y) cell coordinates to check

        Returns:
            bool : return True if the given cell is one of the cells occupied 
                by the ship. Otherwise, return False
        """
        if cell in self.get_cells():
            return True
        else:
            return False
    

    def receive_damage(self, cell):
        """ Receive attack at given cell. 
        
        If ship occupies the cell, add the cell coordinates to the set of 
        damaged cells. Then return True. 
        
        Otherwise return False.

        Args:
            cell (tuple[int, int]): tuple of 2 positive integers representing
                the cell coordinates that is damaged

        Returns:
            bool : return True if the ship is occupying cell (ship is hit). 
                Return False otherwise.
        """
        if not self.is_occupying_cell(cell):
            return False
        else:
            self.damaged_cells.add(cell)
            return True
    

    def count_damaged_cells(self):
        """ Count the number of cells that have been damaged.
        
        Returns:
            int : the number of cells that are damaged.
        """
        damaged = len(self.damaged_cells)
        return damaged
        

    def has_sunk(self):
        """ Check whether the ship has sunk.
        
        Returns:
            bool : return True if the ship is damaged at all its positions. 
                Otherwise, return False
        """
        if self.cells == self.damaged_cells:
            return True
        else:
            return False
    

    def is_near_ship(self, other_ship):
        """ Check whether a ship is near another ship instance.
        
        Hint: Use the method is_near_cell(...) to complete this method.

        Args:
            other_ship (Ship): another Ship instance against which to compare

        Returns:
            bool : returns True if and only if the coordinate of other_ship is 
                near to this ship. Returns False otherwise.
        """
        cells = other_ship.get_cells()
        for cell in cells:
            if self.is_near_cell(cell):
                return True
        return False


    def is_near_cell(self, cell):
        """ Check whether the ship is near an (x,y) cell coordinate.

        In the example below:
        - There is a ship of length 3 represented by the letter S.
        - The positions 1, 2, 3 and 4 are near the ship
        - The positions 5 and 6 are NOT near the ship

        --------------------------
        |   |   |   |   | 3 |   |
        -------------------------
        |   | S | S | S | 4 | 5 |
        -------------------------
        | 1 |   | 2 |   |   |   |
        -------------------------
        |   |   | 6 |   |   |   |
        -------------------------

        Args:
            cell (tuple[int, int]): tuple of 2 positive integers representing
                the (x, y) cell coordinates to compare

        Returns:
            bool : returns True if and only if the (x, y) coordinate is at most
                one cell from any part of the ship OR is at the corner of the 
                ship. Returns False otherwise.
        """
        return (self.x_start-1 <= cell[0] <= self.x_end+1 
                and self.y_start-1 <= cell[1] <= self.y_end+1)


class ShipFactory:
    """ Class to create new ships in specific configurations."""
    def __init__(self, board_size=(10,10), ships_per_length=None):
        """ Initialises the ShipFactory class with necessary information.
        
        Args: 
            board_size (tuple[int,int]): the (width, height) of the board in 
                terms of number of cells. Defaults to (10, 10)
            ships_per_length (dict): A dict with the length of ship as keys and
                the count as values. Defaults to 1 ship each for lengths 1-5.
        """
        self.board_size = board_size
        
        if ships_per_length is None:
            # Default: lengths 1 to 5, one ship each
            self.ships_per_length = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}
        else:
            self.ships_per_length = ships_per_length

    @classmethod
    def create_ship_from_str(cls, start, end, board_size=(10,10)):
        """ A class method for creating a ship from string based coordinates.
        
        Example usage: ship = ShipFactory.create_ship_from_str("A3", "C3")
        
        Args:
            start (str): starting coordinate of the ship (example: 'A3')
            end (str): ending coordinate of the ship (example: 'C3')
            board_size (tuple[int,int]): the (width, height) of the board in 
                terms of number of cells. Defaults to (10, 10)

        Returns:
            Ship : a Ship instance created from start to end string coordinates
        """
        converter = CellConverter(board_size)
        return Ship(start=converter.from_str(start),
                    end=converter.from_str(end))
    

    def place_ship(self, length):
        """ Place a ship of given length on the board.
           
            Args:
                length (int): length of the ship to place
                start (tuple[int,int]): (x,y) coordinate of the starting cell
                ships (list[Ship]): list of ships already placed on the board
                horizontal (bool): whether the ship should be placed horizontally
                vertical (bool): whether the ship should be placed vertically
            
            Returns:
                list[Ship] : list of ships on the board, including the new ship
        """
        while True:
            # randomly choose horizontal or vertical
            direction = random.choice([0, 1])
            if direction == 0:
                horizontal = True
                vertical = False
            else:
                horizontal = False
                vertical = True

            # randomly choose start and end
            start = (random.randint(1, self.board_size[0] - (length-1)), random.randint(1, self.board_size[1] - (length-1)))
            end = (start[0] + (length-1) * horizontal, start[1] + (length-1) * vertical)
            new_ship = Ship(start, end)
            if len(self.ships) == 0:
                self.ships.append(new_ship)
                break
            else:
                valid = True
                for other_ship in self.ships:
                    # Check if new ship is near other ships
                    if new_ship.is_near_ship(other_ship):
                        valid = False
                if valid == True:
                    self.ships.append(new_ship)
                    break
        

    def generate_ships(self):
        """ Generate a list of ships in the appropriate configuration.
        
        The number and length of ships generated must obey the specifications 
        given in self.ships_per_length.
        
        The ships must also not overlap with each other, and must also not be 
        too close to one another (as defined earlier in Ship::is_near_ship())
        
        The coordinates should also be valid given self.board_size
        
        Returns:
            list[Ships] : A list of Ship instances, adhering to the rules above
        """
        self.ships = []
        # I place the ships biggest first, as it is easier to plaece the smaller ships around them instead of the other way around
        for length, freq in reversed(self.ships_per_length.items()):
            for i in range(freq):
                self.place_ship(length)
        return self.ships
        
        
if __name__ == '__main__':
    # SANDBOX for you to play and test your methods
    """
    ship = Ship(start=(3, 3), end=(5, 3))
    print(ship.get_cells())
    print(ship.length())
    print(ship.is_horizontal())
    print(ship.is_vertical())
    print(ship.is_near_cell((5, 3)))
    
    print(ship.receive_damage((3, 3)))
    print(ship.receive_damage((4, 3)))
    print(ship.receive_damage((5, 3)))
    print(ship.receive_damage((10, 3)))
    print(ship.damaged_cells)
    print(ship.count_damaged_cells())
    print(ship.has_sunk())
    
    ship2 = Ship(start=(4, 1), end=(4, 5))
    print(ship.is_near_ship(ship2))
    """
    # For Task 3
    ships_per_length = {1: 2, 2: 1, 3: 1, 4: 1, 5: 2}
    ships = ShipFactory((12, 12), ships_per_length).generate_ships()
    print(ships)
        
    