from tp1.algorithms.a_star import a_star

from tp1.types.coords import Coords

import json
import sys
from pathlib import Path

from tp1.models.board_manager import BoardManager
from tp1.models.sokoban_board import BoardCellType, SokobanBoard
from tp1.types.sokoban_state import SokobanBoardState

def parse_board(path: str | Path) -> tuple[SokobanBoard, SokobanBoardState]:
    with open(path) as f:
        data = json.load(f)

    elements = []
    for x, y in data.get("walls", []):
        elements.append((x, y, BoardCellType.WALL))

    cubes: list[Coords] = []
    for x, y in data.get("cubes", []):
        cubes.append((x, y))

    for x, y in data.get("wins", []):
        elements.append((x, y, BoardCellType.GOAL))

    board = SokobanBoard(data["x"], data["y"], elements)

    return board, (tuple(data.get("player", (0, 0))), tuple(cubes))

def main() -> None:
    global manager

    if len(sys.argv) < 2:
        print("Usage: tp1 <board.json>")
        sys.exit(1)

    board, start = parse_board(sys.argv[1])
    print(f"Board {board.width}x{board.height}")
    print(f"Player at {start[0]}")
    print(f"Walls at {[ (x, y) for y, row in enumerate(board.cells) for x, cell in enumerate(row) if cell.cell_type == BoardCellType.WALL ]}")
    print(f"Cubes at {start[1]}")
    print(f"Goals at {[ (x, y) for y, row in enumerate(board.cells) for x, cell in enumerate(row) if cell.cell_type == BoardCellType.GOAL ]}")

    manager = BoardManager(board)
    solution = a_star(manager, start, board.h_trivial)

    print(solution)
    print("Solution found!" if solution is not None else ":/")

if __name__ == "__main__":
    main()
