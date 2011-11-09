n = 3

def print_board(board):
    d = {
        True: 'X',
        False: 'O',
        None: '-'
    }
    for i in range(0, n*n, n):
        print("".join(d[board[j]] for j in range(i, i+n)))
    print()

def next_states(board, p):
    ls = []
    for i in range(n*n):
        if board[i] == None:
            b = list(board)
            b[i] = p
            ls.append(b)
    return ls

def final_state(board, p):
    # print_board(board)
    if not any(map(lambda l: l == None, board)):
        return 1
    lines = [[], []]
    r = range(n*n)
    for i in range(n):
        lines[0].append(i * (n + 1))
        lines[1].append((i + 1) * (n - 1))
        lines.append(list(filter(lambda x: x % n == i, r)))
        lines.append(list(filter(lambda x: x // n == i, r)))
    for line in lines:
        if all(map(lambda i: board[i] == p, line)):
            return 2
        if all(map(lambda i: board[i] == (not p), line)):
            return 0
    return None

def minimax(board, p, a=None, b=None):
    result = final_state(board, p)
    if result != None:
        return result
    else:
        v = 0
        for s in next_states(board, p):
            v = max(v, 2 - minimax(s, not p, b, a))
            if b and v >= b:
                return v
            a = max(a, v) if a else v
        return v

# print(minimax([None for _ in range(n*n)], True))
assert minimax([
    True, False, True,
    False, None, None,
    None, None, None
], True) == 2
assert minimax([
    False, True, False,
    True, False, True,
    None, None, None
], True) == 0
assert minimax([
    False, True, False,
    True, None, True,
    None, None, None
], False) == 2
assert minimax([
    True, False, True,
    False, True, None,
    None, None, None
], False) == 0
assert minimax([
    None, None, False,
    None, True, None,
    None, None, None
], True) == 1
assert minimax([
    None, None, None,
    None, None, None,
    None, None, None
], True) == 1