import json
import sys
from pathlib import Path

from tp1.models.board import Board, BoardCellType
from tp1.models.board_node import BoardNode
from tp1.algorithms.a_star import a_star

def h_trivial(state: BoardNode) -> int:
    return 0 if state.is_goal() else 1

def goal_func(state: BoardNode) -> bool:
    return state.is_goal()

def parse_board(path: str | Path) -> Board:
    with open(path) as f:
        data = json.load(f)

    elements = []
    for x, y in data.get("walls", []):
        elements.append((x, y, BoardCellType.WALL))
    for x, y in data.get("cubes", []):
        elements.append((x, y, BoardCellType.BLOCK))
    for x, y in data.get("wins", []):
        elements.append((x, y, BoardCellType.GOAL))

    board = Board(data["x"], data["y"], tuple(data.get("player", (0, 0))), elements)

    return board

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: tp1 <board.json>")
        sys.exit(1)

    board = parse_board(sys.argv[1])
    print(f"Board {board.width}x{board.height}")
    print(f"Player at {board.player_position}")
    print(f"Walls at {[ (x, y) for y, row in enumerate(board.cells) for x, cell in enumerate(row) if cell.cell_type == BoardCellType.WALL ]}")
    print(f"Cubes at {[ (x, y) for y, row in enumerate(board.cells) for x, cell in enumerate(row) if cell.cell_type == BoardCellType.BLOCK ]}")
    print(f"Goals at {[ (x, y) for y, row in enumerate(board.cells) for x, cell in enumerate(row) if cell.cell_type == BoardCellType.GOAL ]}")

    a_star(BoardNode(board), goal_func, h_trivial)

    print("Solution found!")
