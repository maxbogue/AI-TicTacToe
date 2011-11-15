import json
import random
import sys

from nn import NeuralNet

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
    
    def __init__(self, name):
        self.name = name
        self.score = 0
    
    def set_player(self, p):
        """Sets the player of this agent to 1 or -1."""
        self.p = p
    
    def move(self, board):
        """Performs an action on the board and returns the new board."""
        raise NotImplementedError
    
    def game_over(self, score):
        """Function called when the game has ended."""
        self.score += score
    
    def __str__(self):
        return "%s as P%s" % (self.name, 1 if self.p == 1 else 2)
    

class MinimaxAgent(Agent):
    """A perfect agent that plays using the full minimax algorithm."""
    
    def move(self, board):
        return minimax(board, self.p)[1]
    

class NeuralNetAgent(Agent):
    """An imperfect agent that uses a neural network to learn and play."""
    
    def __init__(self, name, nn):
        Agent.__init__(self, name)
        self.nn = nn
        self.examples = []
    
    def eval(self, board, p):
        assert(p == self.p)
        x = [p] + board
        self.examples.append(x)
        return self.nn.run(x)[0] * 2
    
    def move(self, board):
        return minimax(board, self.p, self.eval, 3)[1]
    
    def game_over(self, score):
        Agent.game_over(self, score)
        truths = [[score] for _ in range(len(self.examples))]
        self.nn.train(self.examples, truths))
        self.examples = []
    

class RandomAgent(Agent):
    
    def move(self, board):
        return random.choice(next_states(board, self.p))
    

class SemiRandomAgent(Agent):
    
    def eval(self, board, p):
        assert(p == self.p)
        return random.random() * 2
    
    def move(self, board):
        return minimax(board, self.p, self.eval, 3)[1]
    

def play_game(p1, p2):
    p1.set_player(P1)
    p2.set_player(P2)
    board = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    p1_turn = True
    while final_state(board, 1) == None:
        if p1_turn:
            board = p1.move(board)
        else:
            board = p2.move(board)
        print()
        print_board(board)
        p1_turn = not p1_turn
    result = final_state(board, P1)
    p1.game_over(result)
    p2.game_over(2 - result)
    if result == 1:
        print("Tie game. (%s)\n" % p1)
        return None, board
    else:
        winner = p1 if result == 2 else p2
        loser = p1 if result == 0 else p2
        print("%s has defeated %s.\n" % (winner, loser))
        return winner, board

def tournament(p1, p2):
    if random.random() < 0.5:
        p1, p2 = p2, p1
    results = []
    for i in range(1, 6):
        print("Game %s:" % i)
        winner, board = play_game(p1, p2)
        results.append((i, p1, winner, board))
        if winner != p1:
            p1, p2 = p2, p1
    print("Tournament summary: %s versus %s\n" % (p1.name, p2.name))
    for i, pp1, winner, board in results:
        if winner:
            print("Game %s was won by %s as P%s:" %
                (i, winner.name, 1 if winner == pp1 else 2))
        else:
            print("Game %s was a tie. (%s as P1)" % (i, pp1.name))
        print_board(board)
        print()
    print("Final scores:")
    print("%s: %s" %(p1.name, p1.score))
    print("%s: %s" %(p2.name, p2.score))

def main(weight_file=None):
    if weight_file:
        with open(weight_file, "r") as f:
            weights = json.load(f)
        mlp = NeuralNet(weights["mlp"])
        lr = NeuralNet(weights["lr"])
    else:
        mlp = NeuralNet.create(10, 10, 1)
        lr = NeuralNet.create(10, 1)
        train(mlp)
        train(lr)
        with open("weights.json", "w") as f:
            json.dump({
                "mlp": mlp.weights,
                "lr": lr.weights,
            }, f)
    tournament(NeuralNetAgent("MLP", mlp), NeuralNetAgent("LR", lr))

def train(nn):
    last = None
    count = 0
    a1 = NeuralNetAgent("Agent 1", nn)
    a2 = NeuralNetAgent("Agent 2", nn)
    for i in range(1, 101):
        print("Training game %s:" % i)
        winner, board = play_game(a1, a2)
        if last and board == last:
            count += 1
        else:
            last = board
            count = 0
        # if count > 10:
            # break
    print(nn.weights)

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
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
