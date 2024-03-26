import random

from battleship.board import Board
from battleship.convert import CellConverter

class Player:
    """ Class representing the player
    """
    count = 0  # for keeping track of number of players
    
    def __init__(self, board=None, name=None):
        """ Initialises a new player with its board.

        Args:
            board (Board): The player's board. If not provided, then a board
                will be generated automatically
            name (str): Player's name
        """
        
        if board is None:
            self.board = Board()
        else:
            self.board = board
        
        Player.count += 1
        if name is None:
            self.name = f"Player {self.count}"
        else:
            self.name = name
    
    def __str__(self):
        return self.name
    
    def select_target(self):
        """ Select target coordinates to attack.
        
        Abstract method that should be implemented by any subclasses of Player.
        
        Returns:
            tuple[int, int] : (x, y) cell coordinates at which to launch the 
                next attack
        """
        raise NotImplementedError
    
    def receive_result(self, is_ship_hit, has_ship_sunk):
        """ Receive results of latest attack.
        
        Player receives notification on the outcome of the latest attack by the 
        player, on whether the opponent's ship is hit, and whether it has been 
        sunk. 
        
        This method does not do anything by default, but can be overridden by a 
        subclass to do something useful, for example to record a successful or 
        failed attack.
        
        Returns:
            None
        """
        return None
    
    def has_lost(self):
        """ Check whether player has lost the game.
        
        Returns:
            bool: True if and only if all the ships of the player have sunk.
        """
        return self.board.have_all_ships_sunk()


class ManualPlayer(Player):
    """ A player playing manually via the terminal
    """
    def __init__(self, board, name=None):
        """ Initialise the player with a board and other attributes.
        
        Args:
            board (Board): The player's board. If not provided, then a board
                will be generated automatically
            name (str): Player's name
        """
        super().__init__(board=board, name=name)
        self.converter = CellConverter((board.width, board.height))
        
    def select_target(self):
        """ Read coordinates from user prompt.
               
        Returns:
            tuple[int, int] : (x, y) cell coordinates at which to launch the 
                next attack
        """
        print(f"It is now {self}'s turn.")

        while True:
            try:
                coord_str = input('coordinates target = ')
                x, y = self.converter.from_str(coord_str)
                return x, y
            except ValueError as error:
                print(error)


class RandomPlayer(Player):
    """ A Player that plays at random positions.

    However, it does not play at the positions:
    - that it has previously attacked
    """
    def __init__(self, name=None):
        """ Initialise the player with an automatic board and other attributes.
        
        Args:
            name (str): Player's name
        """
        # Initialise with a board with ships automatically arranged.
        super().__init__(board=Board(), name=name)
        self.tracker = set()

    def select_target(self):
        """ Generate a random cell that has previously not been attacked.
        
        Also adds cell to the player's tracker.
        
        Returns:
            tuple[int, int] : (x, y) cell coordinates at which to launch the 
                next attack
        """
        target_cell = self.generate_random_target()
        self.tracker.add(target_cell)
        return target_cell

    def generate_random_target(self):
        """ Generate a random cell that has previously not been attacked.
               
        Returns:
            tuple[int, int] : (x, y) cell coordinates at which to launch the 
                next attack
        """
        has_been_attacked = True
        random_cell = None
        
        while has_been_attacked:
            random_cell = self.get_random_coordinates()
            has_been_attacked = random_cell in self.tracker

        return random_cell

    def get_random_coordinates(self):
        """ Generate random coordinates.
               
        Returns:
            tuple[int, int] : (x, y) cell coordinates at which to launch the 
                next attack
        """
        x = random.randint(1, self.board.width)
        y = random.randint(1, self.board.height)
        return (x, y)


class AutomaticPlayer(Player):
    """ Player playing automatically using a strategy."""
    def __init__(self, name=None):
        """ Initialise the player with an automatic board and other attributes.
        
        Args:
            name (str): Player's name
        """
        # Initialise with a board with ships automatically arranged.
        super().__init__(board=Board(), name=name)
        
        self.tracker = set()
        self.recent_hits = []
        self.target_cell = None


    def get_neighbours(self, cell):
        """ Generate the neighbouring (non diagonal) cells of a cell.
        
        Args:
            cell (tuple[int, int]): (x, y) cell coordinates
        
        Returns:
            list[tuple[int, int]]: list of neighbouring cells
        """
        neighbours = []
        x, y = cell
        for i in range(-1, 2):
            for j in range(-1, 2):
                if abs(i) == abs(j):
                    continue
                else:
                    neighbours.append(((x + i)%(1+self.board.width), (y + j)%(1+self.board.height)))
        return neighbours

    
    def adjacent_target(self):
        """ Generate a target cell based on the recent hits. I.e. we hit cells neighbouring the most recent hit until the 
        ship has sunk in a rule based fashion
        
        Returns:
            tuple[int, int] : (x, y) cell coordinates at which to launch the 
                next attack

        """
        # get the most recent hit
        last_hit = self.recent_hits[-1]
        # get the neighbouring cells
        neighbours = self.get_neighbours(last_hit)
        # remove the cells that have already been attacked
        neighbours = [cell for cell in neighbours if cell not in self.tracker]
        # if there are no neighbours left, then we have sunk the ship, so we clear the recent hits list
        if neighbours == []:
            first_hit = self.recent_hits[0]
            neighbours = self.get_neighbours(first_hit)
            # remove the cells that have already been attacked
            neighbours = [cell for cell in neighbours if cell not in self.tracker]
            return neighbours[0]
        else:
            # otherwise we select the first cell in the list
            return neighbours[0]


    def receive_result(self, is_ship_hit, has_ship_sunk):
        """ Receive results of latest attack.
        
        Player receives notification on the outcome of the latest attack by the 
        player, on whether the opponent's ship is hit, and whether it has been 
        sunk. 
        
        If the attack is succesful, then the target cell is added to the recent hits list. If the ship has sunk
        then the recent hits list is cleared.
        
        Returns:
            None
        """
        if is_ship_hit and not has_ship_sunk:
            self.recent_hits.append(self.target_cell)
        elif is_ship_hit and has_ship_sunk:
            for cell in self.recent_hits:
                neighbours = self.get_neighbours(cell)
                for cell in neighbours:
                    self.tracker.add(cell)  
            self.recent_hits.clear()
        else:
            pass
                      

    def get_random_coordinates(self):
        """ Generate random coordinates.
               
        Returns:
            tuple[int, int] : (x, y) cell coordinates at which to launch the 
                next attack
        """
        x = random.randint(1, self.board.width)
        y = random.randint(1, self.board.height)
        return (x, y)
        

    def select_target(self):
        """ Select target coordinates to attack.
        
        Returns:
            tuple[int, int] : (x, y) cell coordinates at which to launch the 
                next attack
        """
        has_been_attacked = True

        while has_been_attacked:
            if self.recent_hits == []:
                self.target_cell = self.get_random_coordinates()
                has_been_attacked = self.target_cell in self.tracker
            else:
                self.target_cell = self.adjacent_target()
                has_been_attacked = self.target_cell in self.tracker

        self.tracker.add(self.target_cell)

        return self.target_cell
       
