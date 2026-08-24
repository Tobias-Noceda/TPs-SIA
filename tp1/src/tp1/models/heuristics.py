import math
# from itertools import permutations  # only used by matching_manhattan, below
from typing import Sequence

from tp1.models.sokoban_board import SokobanBoard

from tp1.types.coords import Coords
from tp1.types.heuristic import Heuristic
from tp1.types.sokoban_state import SokobanBoardState

def trivial(board: SokobanBoard) -> Heuristic[SokobanBoardState]:
    def h(state: SokobanBoardState) -> int:
        return 0 if board.is_goal(state) else 1

    return h

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

def reduction_manhattan(board: SokobanBoard) -> Heuristic[SokobanBoardState]:
    """
    Row plus column reduction of the box/goal Manhattan cost matrix: the first
    step of the Hungarian algorithm, used on its own as a lower bound on the
    minimum cost assignment of boxes to goals.

    Subtract from every row its minimum, then from every column of the result
    its minimum, and add up everything subtracted.

    Admissible: with as many boxes as goals every assignment uses exactly one
    cell of each row and of each column, so subtracting a constant from a row or
    a column lowers every assignment by the same amount. After both reductions
    the matrix is non negative, hence for every assignment

        real cost = reduced cost + row mins + col mins >= row mins + col mins

    Dominates closest_goal_manhattan, which is the sum of the row minimums
    alone: the column reduction is what charges for the goals that no box picked
    as its closest one, the case where closest_goal_manhattan underestimates the
    most.

    Not exact: what it leaves out is the cost of the optimal assignment in the
    twice reduced matrix, which the Hungarian algorithm goes on to close.
    A* only needs a lower bound, and this one is a few lines and O(boxes*goals),
    against O(goals!) for the exact brute force matching kept below.

    Requires as many boxes as goals. With fewer boxes an assignment no longer
    uses every column, so charging for the unused ones overestimates and breaks
    admissibility.

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

        if len(boxes) != len(goals):
            raise ValueError(
                f"The board has {len(boxes)} boxes and {len(goals)} goals; "
                "this heuristic is only admissible when they match"
            )

        if any(box in dead for box in boxes):
            cache[boxes] = math.inf
            return math.inf

        cost = [[manhattan(box, goal) for goal in goals] for box in boxes]

        rows = [min(row) for row in cost]
        reduced = [[c - m for c in row] for row, m in zip(cost, rows)]
        cols = [min(row[goal] for row in reduced) for goal in range(len(goals))]

        best: float = sum(rows) + sum(cols)

        cache[boxes] = best

        return best

    return h

# def matching_manhattan(board: SokobanBoard) -> Heuristic[SokobanBoardState]:
#     """
#     Minimum cost assignment of boxes to goals, using Manhattan as the cost of
#     moving a box to a goal. Every box gets a different goal.
#
#     Admissible: the real solution ends with every box on a different goal, which
#     is one particular assignment, and its real cost is at least the sum of the
#     Manhattan distances it implies. Taking the minimum over every assignment can
#     only lower that bound, so the estimate never overestimates.
#
#     Dominates closest_goal_manhattan (which is this without the "different goal"
#     restriction), so it never expands more nodes than it does.
#
#     Solved by brute force over every injective assignment of boxes to goals:
#     O(goals! / (goals - boxes)!) per box layout. Fine for the handful of boxes
#     of this TP; a bitmask DP over the subsets of goals would bring it down to
#     O(2^goals * goals) if the boards ever grow.
#
#     The estimate only depends on the boxes, not on the player, and the search
#     visits the same box layout with many different player positions, so results
#     are memoized by box layout.
#     """
#     goals: Sequence[Coords] = board.wins
#     dead: frozenset[Coords] = board.dead_cells()
#
#     if not goals:
#         raise ValueError("The board has no goals")
#
#     cache: dict[Sequence[Coords], float] = {}
#
#     def h(state: SokobanBoardState) -> float:
#         _, boxes = state
#
#         if boxes in cache:
#             return cache[boxes]
#
#         if any(box in dead for box in boxes):
#             cache[boxes] = math.inf
#             return math.inf
#
#         # Every way of giving each box a different goal; the cheapest one is the
#         # bound. Empty when there are more boxes than goals: unsolvable, so inf.
#         best: float = min(
#             (
#                 sum(manhattan(box, goal) for box, goal in zip(boxes, assignment))
#                 for assignment in permutations(goals, len(boxes))
#             ),
#             default=math.inf,
#         )
#
#         cache[boxes] = best
#
#         return best
#
#     return h
