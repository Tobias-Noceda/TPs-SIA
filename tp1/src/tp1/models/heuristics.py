import math
from typing import Sequence

from tp1.models.sokoban_board import BoardCellType, BoardMovement, SokobanBoard
from tp1.types.coords import Coords
from tp1.types.heuristic import Heuristic
from tp1.types.sokoban_state import SokobanBoardState

def is_blocked(board: SokobanBoard, x: int, y: int) -> bool:
    """A box can never be pushed here. The edge of the board acts as a wall."""
    if x < 0 or x >= board.width or y < 0 or y >= board.height:
        return True

    return board.cells[y][x].cell_type == BoardCellType.WALL

def is_goal_cell(board: SokobanBoard, x: int, y: int) -> bool:
    return board.cells[y][x].cell_type == BoardCellType.GOAL

def dead_cells_of(board: SokobanBoard) -> frozenset[Coords]:
    """
    Cells where a box can never be pushed off again, so putting one there makes
    the level unsolvable. A state holding such a box has no solution, and every
    heuristic below reports infinity for it.

    Depends only on walls and goals, which are static, so it is computed once
    when the heuristic is built and then checked in O(1) per state.

    A goal is never dead: a box sitting on it is already where it belongs.

    The detection is deliberately asymmetric. A cell is only reported as dead
    when it is provably stuck, because a false positive would make the search
    report an unsolvable level, while a false negative only costs speed.
    """
    return frozenset(
        (x, y)
        for y in range(board.height)
        for x in range(board.width)
        if not is_blocked(board, x, y)
        and not is_goal_cell(board, x, y)
        and (is_corner(board, x, y) or is_wall_trapped(board, x, y))
    )

def is_corner(board: SokobanBoard, x: int, y: int) -> bool:
    """A box against a wall on both axes at once can never be moved again."""
    vertical = is_blocked(board, x, y - 1) or is_blocked(board, x, y + 1)
    horizontal = is_blocked(board, x - 1, y) or is_blocked(board, x + 1, y)

    return vertical and horizontal

def is_wall_trapped(board: SokobanBoard, x: int, y: int) -> bool:
    return any(
        is_trapped_against(board, x, y, dx, dy)
        for dx, dy in (movement.value for movement in BoardMovement)
    )

def is_trapped_against(board: SokobanBoard, x: int, y: int, dx: int, dy: int) -> bool:
    """
    Checks the box at (x, y) against the wall in the (dx, dy) direction.

    A box with a wall above it can only slide sideways: pushing it down needs
    the player standing above, inside the wall, and pushing it up drives it
    into the wall. So it is stuck in its row, and it is dead if the stretch it
    can reach holds no goal.

    The catch is that the stretch ends where the wall ends: if the box can
    slide to a cell where the wall is gone, it recovers its vertical movement
    and escapes the row, so it is alive. Checking the whole row instead of the
    reachable stretch would report solvable states as dead.
    """
    if not is_blocked(board, x + dx, y + dy):
        return False

    # The box can only slide along the axis perpendicular to the wall
    slide_x, slide_y = dy, dx

    for step in (1, -1):
        reached_x, reached_y = x, y

        while True:
            if is_goal_cell(board, reached_x, reached_y):
                return False  # Reachable goal, the box is still useful here
            if not is_blocked(board, reached_x + dx, reached_y + dy):
                return False  # The wall ends, the box can escape the stretch

            next_x, next_y = reached_x + slide_x * step, reached_y + slide_y * step

            if is_blocked(board, next_x, next_y):
                break  # Closed end, try the other direction

            reached_x, reached_y = next_x, next_y

    return True

def manhattan(a: Coords, b: Coords) -> int:
    # @link https://en.wikipedia.org/wiki/Taxicab_geometry
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def closest_goal_manhattan(board: SokobanBoard) -> Heuristic[SokobanBoardState]:
    """
    Sum, over every box, of the Manhattan distance to its closest goal.

    Admissible: pushing a box one cell costs at least one player movement, so
    moving a box to a goal costs at least the Manhattan distance between them
    (walls can only make the real path longer). Pushes of different boxes are
    always different player movements, so adding up the per box estimates never
    overestimates the real cost.

    Weak when several boxes pick the same closest goal: the estimate counts all
    of them against it and underestimates by a lot. Fixing that needs a minimum
    cost matching between boxes and goals, see matching_manhattan.

    Returns infinity on a deadlocked state, which is still admissible: no
    solution exists from there, so its real cost is infinite too.
    """
    goals: Sequence[Coords] = board.wins
    dead: frozenset[Coords] = dead_cells_of(board)

    if not goals:
        raise ValueError("The board has no goals")

    def h(state: SokobanBoardState) -> float:
        _, boxes = state

        if any(box in dead for box in boxes):
            return math.inf

        return sum(min(manhattan(box, goal) for goal in goals) for box in boxes)

    return h

def matching_manhattan(board: SokobanBoard) -> Heuristic[SokobanBoardState]:
    """
    Minimum cost assignment of boxes to goals, using Manhattan as the cost of
    moving a box to a goal. Every box gets a different goal.

    Admissible: the real solution ends with every box on a different goal, which
    is one particular assignment, and its real cost is at least the sum of the
    Manhattan distances it implies. Taking the minimum over every assignment can
    only lower that bound, so the estimate never overestimates.

    Dominates closest_goal_manhattan (which is this without the "different goal"
    restriction), so it never expands more nodes than it does.

    Solved with a bitmask DP over the subsets of goals:

        dp[mask] = min cost of assigning the first popcount(mask) boxes to
                   exactly the goals in mask

    The box index is the popcount of the mask, so it does not need its own
    dimension: a mask always describes a prefix of the boxes. Runs in
    O(2^goals * goals), fine for the board sizes of this TP but exponential in
    the number of goals.

    The estimate only depends on the boxes, not on the player, and the search
    visits the same box layout with many different player positions, so results
    are memoized by box layout.
    """
    goals: Sequence[Coords] = board.wins
    dead: frozenset[Coords] = dead_cells_of(board)

    if not goals:
        raise ValueError("The board has no goals")

    cache: dict[Sequence[Coords], float] = {}

    def h(state: SokobanBoardState) -> float:
        _, boxes = state

        if boxes in cache:
            return cache[boxes]

        if any(box in dead for box in boxes):
            cache[boxes] = math.inf
            return math.inf

        cost = [[manhattan(box, goal) for goal in goals] for box in boxes]

        dp: list[float] = [math.inf] * (1 << len(goals))
        dp[0] = 0

        best: float = math.inf

        for mask in range(1 << len(goals)):
            if dp[mask] == math.inf:
                continue

            box = mask.bit_count()

            if box == len(boxes):
                best = min(best, dp[mask])
                continue

            for goal in range(len(goals)):
                if mask & (1 << goal):
                    continue

                dp[mask | (1 << goal)] = min(
                    dp[mask | (1 << goal)], dp[mask] + cost[box][goal]
                )

        cache[boxes] = best

        return best

    return h
