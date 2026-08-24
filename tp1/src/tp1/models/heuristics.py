import math
from typing import Sequence

from tp1.models.sokoban_board import BoardMovement, SokobanBoard
from tp1.types.coords import Coords
from tp1.types.heuristic import Heuristic
from tp1.types.sokoban_state import SokobanBoardState

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
    dead: frozenset[Coords] = board.dead_cells()

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
    dead: frozenset[Coords] = board.dead_cells()

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
