from typing import Sequence
from enum import Enum

from tp1.interfaces.board import Board
from tp1.types.sokoban_state import SokobanBoardState

class BoardCellType(Enum):
    WALL = 0
    GOAL = 1
    EMPTY = 2

class BoardMovement(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

class BoardCell:
    def __init__(self, cell_type: BoardCellType):
        self.cell_type = cell_type

class SokobanBoard(Board[SokobanBoardState]):
    width: int
    height: int
    cells: list[list[BoardCell]]

    def __init__(self, width: int, height: int, elements: list[tuple[int, int, BoardCellType]] = list()):
        """
        Initializes a Board instance.
        width: The width of the board.
        height: The height of the board.
        elements: A list of tuples where each tuple contains the x-coordinate, y-coordinate, and the type of the cell (BoardCellType).
        elements should be only the non-moving elements (walls, goals, etc.). The player position is not included in this list.
        """
        self.width = width
        self.height = height
        self.cells = [[BoardCell(BoardCellType.EMPTY) for _ in range(width)] for _ in range(height)]
        
        for x, y, cell_type in elements:
            self.__set_cell(x, y, cell_type)

    def __set_cell(self, x: int, y: int, cell_type: BoardCellType):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.cells[y][x] = BoardCell(cell_type)
        else:
            raise ValueError("Coordinates out of bounds")

    def possible_movements(self, state: SokobanBoardState) -> Sequence[SokobanBoardState]:
        movements: list[SokobanBoardState] = []
        player_coords, box_coords = state

        for movement in BoardMovement:
            new_x = player_coords[0] + movement.value[0]
            new_y = player_coords[1] + movement.value[1]
            if self.__is_out_of_bounds(new_x, new_y):
                continue
            if self.__lands_in_wall(new_x, new_y):
                continue
            if (new_x, new_y) in box_coords:
                # If the player is trying to move into a box, check if the box can be pushed
                box_new_x = new_x + movement.value[0]
                box_new_y = new_y + movement.value[1]
                if self.__is_out_of_bounds(box_new_x, box_new_y):
                    continue
                if self.__lands_in_wall(box_new_x, box_new_y):
                    continue
                if (box_new_x, box_new_y) in box_coords:
                    continue  # Can't push a box into another box
                # Move the player and the box
                new_box_coords = tuple((box_new_x, box_new_y) if (x, y) == (new_x, new_y) else (x, y) for (x, y) in box_coords)
                movements.append(((new_x, new_y), new_box_coords))
            else:
                # Move the player only
                movements.append(((new_x, new_y), box_coords))

        return movements

    def __is_out_of_bounds(self, x: int, y: int) -> bool:
        return x < 0 or x >= self.width or y < 0 or y >= self.height

    def __lands_in_wall(self, x: int, y: int) -> bool:
        return self.cells[y][x].cell_type == BoardCellType.WALL

    def is_goal(self, state: SokobanBoardState) -> bool:
        _, box_coords = state
        for x, y in box_coords:
            if self.cells[y][x].cell_type != BoardCellType.GOAL:
                return False
        return True
