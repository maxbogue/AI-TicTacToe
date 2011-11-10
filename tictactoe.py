N = 3

P1 = 1
P2 = -1
ES = 0

def print_board(board):
    d = {
        P1: 'X',
        P2: 'O',
        ES: '-'
    }
    for i in range(0, N*N, N):
        print("".join(d[board[j]] for j in range(i, i+N)))
    print()

def next_states(board, p):
    ls = []
    for i in range(N*N):
        if board[i] == 0:
            b = list(board)
            b[i] = p
            ls.append(b)
    return ls

def final_state(board, p):
    if not any(map(lambda x: x == ES, board)):
        return 1 # Tie
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
    return None

def minimax(board, p, a=None, b=None):
    result = final_state(board, p)
    if result != None:
        return result
    else:
        v = 0
        for s in next_states(board, p):
            v = max(v, 2 - minimax(s, -p, b, a))
            if b and v >= b:
                return v
            a = max(a, v) if a else v
        return v

def game(p1, p2):
    p1.set_player(P1)
    p2.set_player(P2)
    board = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    p1_turn = True
    while final_state(board, 1) == None:
        if p1_turn:
            board = p1.move(board)
        else:
            board = p1.move(board)
        p1_turn = not p1_turn
    result = final_state(board, 1)
    p1.game_over(result)
    p2.game_over(2 - result)
    print_board(board)

class Agent(object):
    """An interface that defines what a game agent should be able to do."""
    
    def set_player(p):
        """Sets the player of this agent to 1 or -1."""
        self.p == p
    
    def move(self, board):
        """Performs an action on the board and returns the new board."""
        raise NotImplementedError
    
    def game_over(score):
        """Function called when the game has ended."""
        pass
    

# print(minimax([0 for _ in range(n*n)], 1))
assert minimax([
    1, -1, 1,
    -1, 0, 0,
    0, 0, 0
], 1) == 2
assert minimax([
    -1, 1, -1,
    1, -1, 1,
    0, 0, 0
], 1) == 0
assert minimax([
    -1, 1, -1,
    1, 0, 1,
    0, 0, 0
], -1) == 2
assert minimax([
    1, -1, 1,
    -1, 1, 0,
    0, 0, 0
], -1) == 0
assert minimax([
    0, 0, -1,
    0, 1, 0,
    0, 0, 0
], 1) == 1
assert minimax([
    0, 0, 0,
    0, 0, 0,
    0, 0, 0
], 1) == 1