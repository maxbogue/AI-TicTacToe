n = 3

def print_board(board):
    d = {
        1: 'X',
        -1: 'O',
        0: '-'
    }
    for i in range(0, n*n, n):
        print("".join(d[board[j]] for j in range(i, i+n)))
    print()

def next_states(board, p):
    ls = []
    for i in range(n*n):
        if board[i] == 0:
            b = list(board)
            b[i] = p
            ls.append(b)
    return ls

def final_state(board, p):
    if not any(map(lambda x: x == 0, board)):
        return 1 # Tie
    lines = [[], []]
    for i in range(n):
        lines[0].append(i * (n + 1))
        lines[1].append((i + 1) * (n - 1))
        lines.append(list(filter(lambda x: x % n == i, range(n*n))))
        lines.append(list(filter(lambda x: x // n == i, range(n*n))))
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