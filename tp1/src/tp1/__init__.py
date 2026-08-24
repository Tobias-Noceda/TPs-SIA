from tp1.algorithms.a_star import a_star
from tp1.algorithms.greedy import greedy
from tp1.algorithms.iddfs import iddfs
from tp1.algorithms.dfs import dfs
from tp1.algorithms.bfs import bfs

from tp1.models.sokoban_parser import parse_board
from tp1.models.board_manager import BoardManager
from tp1.models.sokoban_board import BoardCellType, SokobanBoard, path_to_movements
from tp1.types.sokoban_state import SokobanBoardState

import json
import sys
import time
from pathlib import Path
from typing import Sequence

def main() -> None:
    global manager

    if len(sys.argv) < 2:
        print("Usage: tp1 <board.json>")
        sys.exit(1)

    board_path = Path(sys.argv[1])
    board, start = parse_board(board_path)

    print(f"Board {board.width}x{board.height}")
    print(f"Player at {start[0]}")
    print(f"Walls at {[ (x, y) for y, row in enumerate(board.cells) for x, cell in enumerate(row) if cell.cell_type == BoardCellType.WALL ]}")
    print(f"Cubes at {start[1]}")
    print(f"Goals at {[ (x, y) for y, row in enumerate(board.cells) for x, cell in enumerate(row) if cell.cell_type == BoardCellType.GOAL ]}")

    manager = BoardManager(board)

    start_time = time.perf_counter()

    # Swap the active algorithm by commenting/uncommenting the corresponding block.
    # algorithm_name = "a_star"
    # path = a_star(manager, start, board.h_trivial)
    # algorithm_name = "greedy"
    # path = greedy(manager, start, board.h_trivial)
    # algorithm_name = "iddfs"
    # path = iddfs(manager, start)
    # algorithm_name = "dfs"
    # path, expanded, frontier = dfs(manager, start)
    algorithm_name = "bfs"
    path, expanded, frontier = bfs(manager, start)

    elapsed_time = time.perf_counter() - start_time

    print(path)
    print("Solution found!" if path is not None else ":/")

    save_result(board_path, board, start, algorithm_name, path,expanded, frontier, elapsed_time)

def save_result(
    board_path: Path,
    board: SokobanBoard,
    start: SokobanBoardState,
    algorithm_name: str,
    path: Sequence[SokobanBoardState] | None,
    expanded: int,
    frontier: int,
    elapsed_time: float,
) -> None:
    board_data = {
        "x": board.width,
        "y": board.height,
        "player": list(start[0]),
        "walls": [list(coord) for coord in board.walls],
        "cubes": [list(coord) for coord in start[1]],
        "wins": [list(coord) for coord in board.wins],
    }

    output = {
        "algorithm": algorithm_name,
        "board": board_data,
        "steps": [movement.name for movement in path_to_movements(path)] if path is not None else [],
        "expanded_nodes": expanded,
        "frontier_nodes": frontier,
        "execution_time_seconds": elapsed_time,
    }

    output_dir = Path(__file__).resolve().parents[2] / "res" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{board_path.stem}_{algorithm_name}.json"

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved result to {output_path}")

if __name__ == "__main__":
    main()
