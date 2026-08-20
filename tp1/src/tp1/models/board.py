from enum import Enum

from interfaces.coords import Coords

class BoardCellType(Enum):
    WALL = 0
    BLOCK = 1
    EMPTY = 2
    GOAL = 3

class BoardMovement(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

class BoardCell:
    def __init__(self, cell_type: BoardCellType):
        self.cell_type = cell_type

class Board:
    player_position: Coords
    cells: list[list[BoardCell]]

    def __init__(self, width: int, height: int, player_position: Coords, elements: list[tuple[int, int, BoardCellType]] = list()):
        self.width = width
        self.height = height
        self.player_position = player_position
        self.cells = [[BoardCell(BoardCellType.EMPTY) for _ in range(width)] for _ in range(height)]
        
        for x, y, cell_type in elements:
            self.set_cell(x, y, cell_type)

    def set_cell(self, x: int, y: int, cell_type: BoardCellType):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.cells[y][x] = BoardCell(cell_type)
        else:
            raise ValueError("Coordinates out of bounds")
    
    def get_cell(self, x: int, y: int) -> BoardCell:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.cells[y][x]
        else:
            raise ValueError("Coordinates out of bounds")

    def move_player(self, movement: BoardMovement) -> bool:
        new_x = self.player_position[0] + movement.value[0]
        new_y = self.player_position[1] + movement.value[1]
        if 0 <= new_x < self.width and 0 <= new_y < self.height:
            if self.get_cell(new_x, new_y).cell_type == BoardCellType.EMPTY:
                self.player_position = (new_x, new_y)

                return False
            elif self.get_cell(new_x, new_y).cell_type == BoardCellType.BLOCK:
                block_position = self.move_element((new_x, new_y), movement)
                if block_position:
                    self.player_position = (new_x, new_y)
                    self.set_cell(block_position[0], block_position[1], BoardCellType.BLOCK)
                    self.set_cell(new_x, new_y, BoardCellType.EMPTY)

                    return self.get_cell(block_position[0], block_position[1]).cell_type == BoardCellType.GOAL
                return False

        return False

    def move_element(self, position: Coords, movement: BoardMovement) -> Coords:
        new_x = position[0] + movement.value[0]
        new_y = position[1] + movement.value[1]
        if 0 <= new_x < self.width and 0 <= new_y < self.height:
            if self.get_cell(new_x, new_y).cell_type == BoardCellType.EMPTY:
                return (new_x, new_y)
            else:
                raise ValueError("Cannot move element to a non-empty cell")
        else:
            raise ValueError("Coordinates out of bounds")

    def get_player_peer_movements(self) -> list[BoardMovement]:
        peers = []
        for movement in BoardMovement:
            new_x = self.player_position[0] + movement.value[0]
            new_y = self.player_position[1] + movement.value[1]
            if 0 <= new_x < self.width and 0 <= new_y < self.height and self.get_cell(new_x, new_y).cell_type != BoardCellType.WALL:
                peers.append(movement)

        return peers

    def is_goal(self) -> bool:
        for y, row in enumerate(self.cells):
            for x, cell in enumerate(row):
                if cell.cell_type == BoardCellType.BLOCK:
                    if self.get_cell(x, y).cell_type != BoardCellType.GOAL:
                        return False

        return True

    def __eq__(self, value: object) -> bool:
        if isinstance(value, Board):
            return self.width == value.width and self.height == value.height and self.cells == value.cells and self.player_position == value.player_position
        return False

    def __hash__(self) -> int:
        return hash((self.width, self.height, tuple(tuple(cell.cell_type for cell in row) for row in self.cells), self.player_position))
