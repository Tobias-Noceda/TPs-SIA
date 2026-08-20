from tp1.models.node import Node
from tp1.models.board import Board

# Board Node has a type (an enum with block, empty and goal) and a position (a tuple with x and y coordinates)
class BoardNode(Node):  
    board: Board
  
    def __init__(self, board: Board):
        self.board = board

    def peers(self) -> list[Node]:
        peer_movements = self.board.get_player_peer_movements()
        peers = []

        for movement in peer_movements:
            new_board = Board(self.board.width, self.board.height, self.board.player_position, [(x, y, cell.cell_type) for y, row in enumerate(self.board.cells) for x, cell in enumerate(row)])
            new_board.move_player(movement)
            peers.append(BoardNode(new_board))

        return peers

    def dist(self, other: Node) -> int:
        return 1

    def is_goal(self) -> bool:
        return self.board.is_goal()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BoardNode):
            return self.board == other.board
        return False

    def __hash__(self) -> int:
        return hash(self.board)
