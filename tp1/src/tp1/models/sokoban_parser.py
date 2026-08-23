from tp1.models.sokoban_board import BoardCellType, SokobanBoard

from tp1.types.coords import Coords
from tp1.types.sokoban_state import SokobanBoardState

import json
from pathlib import Path

def parse_board(path: str | Path) -> tuple[SokobanBoard, SokobanBoardState]:
    with open(path) as f:
        data = json.load(f)

    elements = []
    for x, y in data.get("walls", []):
        elements.append((x, y, BoardCellType.WALL))
    for x, y in data.get("wins", []):
        elements.append((x, y, BoardCellType.GOAL))

    board = SokobanBoard(data["x"], data["y"], elements)

    cubes: list[Coords] = []
    for x, y in data.get("cubes", []):
        cubes.append((x, y))

    player = data.get("player", (0, 0))

    return board, (tuple(player), tuple(cubes))
