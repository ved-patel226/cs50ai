"""
Tic Tac Toe Player
"""

import math
from copy import deepcopy
import textColor as ttt

x
X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """

    xcount = 0
    ocount = 0

    for row in board:
        for cell in row:
            if cell == X:
                xcount += 1
            if cell == O:
                ocount += 1
            else:
                continue

    if xcount <= ocount:
        return X
    else:
        return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """

    all_actions = set()

    for row_index, row in enumerate(board):
        for col_index, col in enumerate(row):
            if col == EMPTY:
                all_actions.add((row_index, col_index))

    return all_actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """

    cur_player = player(board)

    new_board = deepcopy(board)

    i, j = action

    if new_board[i][j] == EMPTY:
        new_board[i][j] = cur_player
        return new_board
    else:
        raise Exception


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """

    for player in (X, O):
        for row in board:
            if row == [player, player, player]:
                return player

        for col in range(3):
            if [board[0][col], board[1][col], board[2][col]] == [
                player,
                player,
                player,
            ]:
                return player

        if [board[0][0], board[1][1], board[2][2]] == [player, player, player]:
            return player
        elif [board[0][2], board[1][1], board[2][0]] == [player, player, player]:
            return player
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """

    if winner(board) is not None:
        return True

    for row in board:
        if EMPTY in row:
            return False

    return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """

    player = winner(board)

    if player == X:
        return 1
    elif player == O:
        return -1
    else:  # Tie
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """

    def max_value(board):

        if board == initial_state():
            return 0, (0, 1)

        counter = 0
        optimal = ()
        if terminal(board):
            return utility(board), optimal
        else:
            v = -5
            for action in actions(board):
                counter += 1
                minval = min_value(result(board, action))[0]
                if minval > v:
                    v = minval
                    optimal = action

            print(ttt.red(f"Max Value: {v}"))
            print(ttt.blue(f"Optimal: {optimal}"))
            print(ttt.yellow(f"Counter: {counter}"))

            return v, optimal

    def min_value(board):

        counter = 0
        optimal = ()
        if terminal(board):
            return utility(board), optimal
        else:
            v = 5
            for action in actions(board):
                counter += 1
                maxval = max_value(result(board, action))[0]
                if maxval < v:
                    v = maxval
                    optimal = action
            print(ttt.red(f"Max Value: {v}"))
            print(ttt.blue(f"Optimal: {optimal}"))
            print(ttt.yellow(f"Counter: {counter}"))

            return v, optimal

    cur_player = player(board)

    if terminal(board):
        return None

    if cur_player == X:
        return max_value(board)[1]

    else:
        return min_value(board)[1]

    raise NotImplementedError
