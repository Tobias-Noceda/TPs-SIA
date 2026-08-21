from typing import Sequence, Tuple

from interfaces.state_manager import StateManager
from interfaces.board_state import BoardState
from interfaces.coords import Coords

from models.board import Board, BoardCellType

# Board Node has a type (an enum with block, empty and goal) and a position (a tuple with x and y coordinates)
class BoardManager(StateManager[BoardState]):
    board: Board

    def __init__(self, board: Board):
        self.board = board

    def next(self, state: BoardState) -> Sequence[BoardState]:
        curr = self.clone_with_state(state)
        next: Sequence[BoardState] = []

        peer_movements = curr.get_player_peer_movements()
        for movement in peer_movements:
            board = self.clone_with_player(movement.value)
            board.move_player(movement)

            boxes: Sequence[Coords] = []
            for x, row in enumerate(board.cells):
                for y, cell in enumerate(row):
                    if cell.cell_type == BoardCellType.BLOCK:
                        boxes.append((x, y))

            next.append((board.player_position, tuple(boxes)))

        return next

    def dist(self, start: BoardState, end: BoardState) -> int:
        return 1

    def is_goal(self, state: BoardState) -> bool:
        return self.board.is_goal()

    def clone_with_player(self, player: Coords):
        # TODO: NASSSSTYYY
        things: list[tuple[int, int, BoardCellType]] = []

        for y, row in enumerate(self.board.cells):
            for x, cell in enumerate(row):
                things.append((x, y, cell.cell_type))

        return Board(self.board.width, self.board.height, player, things)

    def clone_with_state(self, state: BoardState):
        player: Coords = state[0]
        boxes: Sequence[Coords] = state[1]

        # TODO: NASSSSTYYY
        things: list[tuple[int, int, BoardCellType]] = []

        for y, row in enumerate(self.board.cells):
            for x, cell in enumerate(row):
                if cell.cell_type != BoardCellType.BLOCK:
                    things.append((x, y, cell.cell_type))

        for (x, y) in boxes:
            things.append((x, y, BoardCellType.BLOCK))

        return Board(self.board.width, self.board.height, player, things)
