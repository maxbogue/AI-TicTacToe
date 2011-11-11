import random

from mlp import MLP

# The size of a side of the game board.
N = 3

# Constants for player 1, player 2, and empty space.
P1 = 1
P2 = -1
ES = 0

def print_board(board):
    """Prints a very simple board representation.
    
    X is used for player 1, O for player 2, and a - for an empty space.
    
    """
    d = {
        P1: 'X',
        P2: 'O',
        ES: '-'
    }
    for i in range(0, N*N, N):
        print("".join(d[board[j]] for j in range(i, i+N)))

def next_states(board, p):
    """Generates every possible next state for a board at player p's turn."""
    ls = []
    for i in range(N*N):
        if board[i] == 0:
            b = list(board)
            b[i] = p
            ls.append(b)
    return ls

def final_state(board, p):
    """Computes whether a board is in a final state.
    
    Returns the score for player p if it is:
    
    0 for loss
    1 for tie
    2 for win
    
    """
    lines = [[], []]
    for i in range(N):
        lines[0].append(i * (N + 1))
        lines[1].append((i + 1) * (N - 1))
        lines.append(list(filter(lambda x: x % N == i, range(N*N))))
        lines.append(list(filter(lambda x: x // N == i, range(N*N))))
    for line in lines:
        if all(map(lambda i: board[i] == p, line)):
            return 2
        if all(map(lambda i: board[i] == -p, line)):
            return 0
    if not any(map(lambda x: x == ES, board)):
        return 1 # Tie
    return None

def minimax(board, p, f=None, i=None, a=None, b=None):
    result = final_state(board, p)
    if result != None:
        return result, board
    else:
        v = None
        best_board = None
        for s in next_states(board, p):
            if f and i == 0:
                u = 2 - f(s, -p)
            else:
                u = 2 - minimax(s, -p, f, i - 1 if i else None, b, a)[0]
            if not v or u > v:
                v = u
                best_board = s
                a = max(a, v) if a else v
            if b and v >= b:
                return v, s
        return v, best_board

class Agent(object):
    """An interface that defines what a game agent should be able to do."""
    
    def set_player(self, p):
        """Sets the player of this agent to 1 or -1."""
        self.p = p
    
    def move(self, board):
        """Performs an action on the board and returns the new board."""
        raise NotImplementedError
    
    def game_over(self, score):
        """Function called when the game has ended."""
        pass
    

class MinimaxAgent(Agent):
    """A perfect agent that plays using the full minimax algorithm."""
    
    def move(self, board):
        return minimax(board, self.p)[1]
    

class MLPAgent(Agent):
    """An imperfect agent that uses MLP learning to play."""
    
    def __init__(self, mlp):
        self.mlp = mlp
        self.examples = []
    
    def eval(self, board, p):
        assert(p == self.p)
        x = [p] + board
        self.examples.append(x)
        return self.mlp.run(x)[0] * 2
    
    def move(self, board):
        return minimax(board, self.p, self.eval, 3)[1]
    
    def game_over(self, score):
        truths = [[score] for _ in range(len(self.examples))]
        self.mlp.train(self.examples, truths)
    

class RandomAgent(Agent):
    
    def move(self, board):
        return random.choice(next_states(board, self.p))
    

class SemiRandomAgent(Agent):
    
    def eval(self, board, p):
        assert(p == self.p)
        return random.random() * 2
    
    def move(self, board):
        return minimax(board, self.p, self.eval, 3)[1]
    

def game(p1, p2):
    p1.set_player(P1)
    p2.set_player(P2)
    board = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    p1_turn = True
    while final_state(board, 1) == None:
        if p1_turn:
            board = p1.move(board)
        else:
            board = p2.move(board)
        p1_turn = not p1_turn
    result = final_state(board, P1)
    p1.game_over(result)
    p2.game_over(2 - result)
    return result - 1, board

def main():
    mlp = MLP.create(10, 10, 1)
    lr = MLP.create(10, 1)
    for i in range(1, 101):
        winner, board = game(MLPAgent(mlp), SemiRandomAgent())
        if winner == P1:
            print("Game %s was won by player 1:" % i)
        elif winner == P2:
            print("Game %s was won by player 2:" % i)
        else:
            print("Game %s was a tie." % i)
        print_board(board)
    print("Final:")
    print(mlp.weights)

def test():
    assert minimax([
        1, -1, 1,
        -1, 0, 0,
        0, 0, 0
    ], 1)[0] == 2
    assert minimax([
        -1, 1, -1,
        1, -1, 1,
        0, 0, 0
    ], 1)[0] == 0
    assert minimax([
        -1, 1, -1,
        1, 0, 1,
        0, 0, 0
    ], -1)[0] == 2
    assert minimax([
        1, -1, 1,
        -1, 1, 0,
        0, 0, 0
    ], -1)[0] == 0
    assert minimax([
        0, 0, -1,
        0, 1, 0,
        0, 0, 0
    ], 1)[0] == 1
    assert minimax([
        0, 0, 0,
        0, 0, 0,
        0, 0, 0
    ], 1)[0] == 1

if __name__ == "__main__":
    # test()
    main()
