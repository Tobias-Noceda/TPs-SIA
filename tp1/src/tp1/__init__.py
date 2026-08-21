from algorithms.a_star import a_star

from interfaces.coords import Coords
from interfaces.board_state import BoardState

from models.board import Board, BoardCellType
from models.board_manager import BoardManager

import json
import sys
from pathlib import Path

manager: BoardManager

def h_trivial(state: BoardState) -> int:
    return 0 if manager.is_goal(state) else 1

def parse_board(path: str | Path) -> tuple[Board, BoardState]:
    with open(path) as f:
        data = json.load(f)

    elements = []
    for x, y in data.get("walls", []):
        elements.append((x, y, BoardCellType.WALL))

    cubes: list[Coords] = []
    for x, y in data.get("cubes", []):
        elements.append((x, y, BoardCellType.BLOCK))
        cubes.append((x, y))

    for x, y in data.get("wins", []):
        elements.append((x, y, BoardCellType.GOAL))

    board = Board(data["x"], data["y"], tuple(data.get("player", (0, 0))), elements)

    return board, (board.player_position, tuple(cubes))

def main() -> None:
    global manager

    if len(sys.argv) < 2:
        print("Usage: tp1 <board.json>")
        sys.exit(1)

    board, start = parse_board(sys.argv[1])
    print(f"Board {board.width}x{board.height}")
    print(f"Player at {board.player_position}")
    print(f"Walls at {[ (x, y) for y, row in enumerate(board.cells) for x, cell in enumerate(row) if cell.cell_type == BoardCellType.WALL ]}")
    print(f"Cubes at {[ (x, y) for y, row in enumerate(board.cells) for x, cell in enumerate(row) if cell.cell_type == BoardCellType.BLOCK ]}")
    print(f"Goals at {[ (x, y) for y, row in enumerate(board.cells) for x, cell in enumerate(row) if cell.cell_type == BoardCellType.GOAL ]}")

    manager = BoardManager(board)
    solution = a_star(manager, start, h_trivial)

    print(solution)
    print("Solution found!" if solution is not None else ":/")

if __name__ == "__main__":
    main()
