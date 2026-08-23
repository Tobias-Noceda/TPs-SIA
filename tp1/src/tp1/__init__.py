from tp1.algorithms.a_star import a_star
from tp1.algorithms.iddfs import iddfs
from tp1.algorithms.dfs import dfs

from tp1.models.sokoban_parser import parse_board
from tp1.models.board_manager import BoardManager
from tp1.models.sokoban_board import BoardCellType

import sys

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

    # solution = a_star(manager, start, board.h_trivial)
    # solution = iddfs(manager, start)
    solution = dfs(manager, start)

    print(solution)
    print("Solution found!" if solution is not None else ":/")

if __name__ == "__main__":
    main()
