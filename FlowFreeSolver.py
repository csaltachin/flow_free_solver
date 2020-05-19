"""
Flow Free Solver v1.0
csaltachin@mit.edu
"""

import time

# Color code dictionary
COLORS = {1: "RED",
          2: "BLUE",
          3: "GREEN",
          4: "YELLOW",
          5: "ORANGE",
          6: "LIGHT BLUE",
          7: "PURPLE",
          8: "BROWN"}


""" Class definitions """

class Board:
    
    def __init__(self, matrix):
        self.matrix = matrix
        self.height = len(matrix)
        self.width = len(matrix[0])
        # Endpoints dictionary {(x,y): color}
        self.endpoints = self.set_endpoints()
        # Endpoints list [(x,y), ...] sorted by color codes
        self.endpoints_sorted = sorted(list(self.endpoints.keys()), key = lambda p: self.endpoints[p],
                                       reverse = False)
        
        # Colors set
        self.colors = set(self.endpoints.values())
        # Divide endpoints into starts and ends, to make starting/ending paths easier
        self.starts, self.ends = {}, {}
        for p in self.endpoints_sorted:
            if self.endpoints[p] in self.starts.values():
                self.ends[p] = self.endpoints[p]
            else:
                self.starts[p] = self.endpoints[p]
    
    # Set endpoints dictionary, fetching from self.matrix
    def set_endpoints(self):
        endpoints = dict()
        
        for x in range(self.width):
            for y in range(self.height):
                char = self.matrix[y][x]
                if char != "#" and char != "0":
                    endpoints[tuple([x, y])] = int(char)
        
        # Check that there are 2 endpoints of each color
        color_list = list(endpoints.values())
        for c in set(endpoints.values()):
            assert color_list.count(c) == 2, f"You must have exactly 2 endpoints for each color in the board, failed for color {c}"
        return endpoints
    
    def get_matrix(self):
        return self.matrix.copy()
    
    def get_endpoints_sorted(self):
        return self.endpoints_sorted.copy()
    
    def get_starts(self):
        return self.starts.copy()
    
    def get_ends(self):
        return self.ends.copy()
    
    # Get start with least color (useful for the initial state)
    def get_first_start(self):
        return min(list(self.starts.keys()), key = lambda s: self.starts[s])
    
    def are_coords_in_board(self, coords):
        try:
            x, y = coords[0], coords[1]
            return x in range(self.width) and y in range(self.height) and self.matrix[y][x] != "#"
        except IndexError:
            return False
    
    def is_endpoint(self, coords):
        return coords in self.endpoints
    
    def max_color(self):
        return max(self.colors)
    
    def get_start_of_color(self, color):
        if color in self.colors:
            for pos in self.starts:
                if self.starts[pos] == color:
                    return pos
        return None
    
    def get_end_of_color(self, color):
        for pos in self.ends:
            if self.ends[pos] == color:
                return pos
        raise KeyError
            


class State:
    
    def __init__(self, board, depth = 0, previous = None, new_move = None):
        self.board = board
        self.depth = depth
        # new_move is a tuple ((x,y), color) representing last move that led to this state
        
        # Set positions dictionary {(x,y): color} for this state
        # based on the previous one plus the new move
        if previous == None:
            self.positions = dict()
        else:
            self.positions = previous.get_positions()
            self.positions[new_move[0]] = new_move[1]
        
        # Set head position ((x,y) tuple) for this state based on new move
        if previous == None:
            # Start with the least color start
            self.head = self.board.get_first_start()
        else:
            self.head = new_move[0]
        
        # Set highest color connected so far (ie the color before head)
        if previous == None:
            self.best_color = 0
        else:
            self.best_color = self.positions[self.head] - 1
    
    def get_board(self):
        return self.board
    
    def get_positions(self):
        return self.positions.copy()
    
    def get_depth(self):
        return self.depth
    
    def get_head(self):
        return self.head
    
    def get_best_color(self):
        return self.best_color
    
    def is_in_positions(self, coords):
        return coords in self.positions
    
    
""" Main functions """

# Get board from inputted matrix
def get_board_from_input():
    matrix = []
    width = int(input("> Enter board width: "))
    height = int(input("> Enter board height: "))
    print("Enter the board row by row. Colors are positive integers, empty spaces are 0, non-board spaces are #.")
          
    for r in range(height):
        row = list(input())
        assert len(row) == width, f"All rows must be of the specified width ({width})"
        matrix.append(row)
    
    return Board(matrix)


# Get neighbors to position in board
def get_neighbors_in_board(board, coords):
    x, y = coords[0], coords[1]
    neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
    return [pos for pos in neighbors if board.are_coords_in_board(pos)]


# Get legal moves [list of (x,y) positions] given state (and its board) and some pos
def get_legal_moves(state, coords):
    board = state.get_board()
    neighbors = get_neighbors_in_board(board, coords)
    return [pos for pos in neighbors if not board.is_endpoint(pos) and not state.is_in_positions(pos)]


# Print state as a matrix
def print_state(state):
    matrix = state.get_board().get_matrix()
    positions = state.get_positions()
    for pos in positions:
        matrix[pos[1]][pos[0]] = str(positions[pos])
        
    for row in matrix:
        print("".join(row))


# Main algorithm
def solve_board(board):
    
    # State count, prune count
    COUNT = 0
    PRUNES = 0
    
    # Open states queue (we check one of these on each iteration of below loop)
    OPEN = []
    # Queue priority key (a dictionary {State: key})
    PRIORITY = {}
    # Also get board ends already
    ENDS = board.get_ends()
    
    # Create initial state, add to OPEN queue
    firstState = State(board)
    OPEN.append(firstState)
    PRIORITY[firstState] = 0
    
    # Main search loop
    while OPEN != []:
        
        # Update COUNT
        COUNT += 1
        # Print COUNT every few states
        if int(COUNT % 1000000) == 0:
            print(f"{int(COUNT // 1e6)} million states inspected.")
        
        # Choose state with the highest *depth* (TWEAK THIS) (will tweak with PRIORITY QUEUE)
        currentState = max(OPEN, key = lambda s: PRIORITY[s])
        # Remove it from OPEN, PRIORITY
        OPEN.remove(currentState)
        del PRIORITY[currentState]
        
        # Set up currentState stuff
        currentHead = currentState.get_head()
        currentColor = currentState.get_best_color() + 1
        currentDepth = currentState.get_depth()
        currentPositions = currentState.get_positions()
        
        # Get next legal moves (a dictionary)
        # First check if head is right next to its end (ie path done)
        neighbors = get_neighbors_in_board(board, currentHead)
        nextStart, nextMoves = "blank", dict() # Default values
        # goodMove = False # For priority heuristic
        
        pruneCount = 0 # For pruning
        for pos in neighbors: 
            # First, try to prune
            if pos in currentPositions and currentPositions[pos] == currentColor:
                pruneCount += 1
            if pruneCount > 1:
                # Prune
                PRUNES += 1
                # Print PRUNES every few prunes
                if PRUNES > 0 and int(PRUNES % 1000000) == 0:
                    print(f"({int(PRUNES // 1e6)} million states prunned.)")
                nextStart = "prune"
                break
                
            if pos in ENDS and ENDS[pos] == currentColor:
                # Try to get next start
                nextStart = board.get_start_of_color(currentColor + 1)
        
        # This will yield nextStart = None if currentColor was the last color, ie. if the flow is solved.
        # In this case, return the final state
        if nextStart == None:
            print("Flow solved!")
            print(f"{COUNT} total states inspected.")
            return currentState
        # Else, if nextStart is a position (tuple), we get the next legal moves from this next start
        # We pass the next color on each of these
        elif type(nextStart) == tuple:
            nextMoves = {pos: currentColor + 1 for pos in get_legal_moves(currentState, nextStart)}
            
        # Now suppose nextStart is "blank", ie path is not done yet.
        # Then simply get next legal moves from current head
        elif nextStart == "blank":
            nextMoves = {pos: currentColor for pos in get_legal_moves(currentState, currentHead)}
        # If nextStart is "prune", nextMoves remains empty
                
        # The if-else's are done and we have our nextMoves.
        # Now for each move in nextMoves, add a new state to OPEN
        # For each new state, add its PRIORITY (TWEAK THIS)
        for move in nextMoves:
            nextState = State(board, currentDepth + 1, currentState, tuple([move, nextMoves[move]]))
            OPEN.append(nextState)
            # PRIORITY[nextState] = nextMoves[move]
            PRIORITY[nextState] = currentDepth + 1
                
        # Delete currentState, as it is no longer needed
        del currentState
            
        
    # Suppose we exit the while loop as OPEN becomes empty. This means no solution was found.
    print("No solution found.")
    print(f"{COUNT} total states inspected.")
    return None
                
        
""" Test code """

if __name__ == "__main__":
     board1 = get_board_from_input()
     w, h = board1.width, board1.height
     print(f"Solving this {w}x{h} board:")
     print_state(State(board1))
     print("Thinking...")

     t0 = time.time()
     f1 = solve_board(board1)

     t1 = time.time() - t0
     h, m, s = t1//3600, (t1//60) % 60, t1%60
     print(f"Done after {int(h)} hours, {int(m)} minutes, {round(s, 3)} seconds.")
     if f1 != None:
          print_state(f1)
