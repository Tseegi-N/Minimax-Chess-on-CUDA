import chess
import time
import random
from datetime import datetime

# Piece values
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}

# Evaluate board at the end for how many pieces are left
def evalBoard(board, color):
    opponent_color = not color
    score = 0
    # compute score
    for piece_type in PIECE_VALUES:
        score += len(board.pieces(piece_type, color)) * PIECE_VALUES[piece_type]
        score -= len(board.pieces(piece_type, opponent_color)) * PIECE_VALUES[piece_type]
    return score


# minimax function for maximizing and minimizing
def minimax(board, depth, is_maximizing, bot_color):
    # at depth 0, evaluate board
    if depth == 0 or board.is_game_over():
        return evalBoard(board, bot_color), None

    best_move = None

    # at bots turn, maximize score recursively
    if is_maximizing:
        max_eval = float('-inf')
        for move in board.legal_moves:
            # do move
            board.push(move)
            eval, _ = minimax(board, depth - 1, False, bot_color)
            # undo move
            board.pop()

            # get max evaluation score
            if eval >= max_eval:
                max_eval = eval
                best_move = move
        return max_eval, best_move

    # at users turn, minimize score recursively
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            # do move
            board.push(move)
            eval, _ = minimax(board, depth - 1, True, bot_color)
            # undo move
            board.pop()

            # get min evaluation score
            if eval <= min_eval:
                min_eval = eval
                best_move = move
        return min_eval, best_move

# bot move with minimax
def bot_move(board, bot_color, depth=5):
    tic = time.perf_counter()
    _, best_move = minimax(board, depth, board.turn == bot_color, bot_color)
    toc = time.perf_counter()
    print(f"Minimax calculation for {depth} depth: {toc - tic:0.4f} seconds")
    return best_move


# def play_chess():
#     fen_start_pos = input("Starting FEN position? (hit ENTER for standard starting position): ")
#     if fen_start_pos == "":
#         board = chess.Board()
#     else:
#         board = chess.Board(fen_start_pos)

#     depth = 2  # minimax depth
#     move_count = 0

#     while not board.is_game_over():
#         move_count += 1
#         print(f"Move {move_count}:")
#         print(board)

#         move = bot_move(board, depth)
#         if move is None:
#             print("No legal moves available. Game over!")
#             break

#         print(f"Bot ({'White' if board.turn == chess.WHITE else 'Black'}) plays: {board.san(move)}")
#         board.push(move)
#         print(f"New FEN position: {board.fen()}\n")

#         if move_count > 100:
#             print("Game stopped: Too many moves (likely a draw).")
#             break

#     print("Game Over!")
#     outcome = board.outcome()
#     if outcome:
#         print(f"Result: {winner(outcome)}")
#     else:
#         print("Game Over! Stalemate or insufficient material.")

def play_chess():
    print(datetime.now())

    # obtain user color
    while True:
        bot_color_input = input("Computer Player? (w=white/b=black): ").lower()
        if bot_color_input in ['w', 'b']:
            bot_color = chess.WHITE if bot_color_input == 'w' else chess.BLACK
            break
        print("Invalid choice. Please choose 'w' for white or 'b' for black.")

    user_color = not bot_color

    fen_start_pos = input("Starting FEN position? (hit ENTER for standard starting position): ")

    if fen_start_pos == "":
        board = chess.Board()
    else:
        board = chess.Board(fen_start_pos)

    # Game loop
    while not board.is_game_over():
        if board.turn == bot_color:
            move = bot_move(board, bot_color)
            print(f"Bot move as {'White' if bot_color == chess.WHITE else 'Black'}: {move}")
            board.push(move)
            print(f"New FEN position: {board.fen()}")
        else:
            user_move = input(f"Your move as {'White' if user_color == chess.WHITE else 'Black'}: ")
            try:
                move = chess.Move.from_uci(user_move)
                if move in board.legal_moves:
                    board.push(move)
                    print(f"New FEN position: {board.fen()}")
                else:
                    print("Invalid move. Try again.")
            except:
                print("Invalid move format. Please try again.")

    # Game over
    outcome = board.outcome()
    if board.is_checkmate():
        print("Checkmate!")
    elif board.is_stalemate():
        print("Stalemate!")
    elif board.is_insufficient_material():
        print("Insufficient Material!")
    print(winner(outcome))


def winner(outcome):
    if outcome.winner is chess.WHITE:
        return "White wins!"
    elif outcome.winner is chess.BLACK:
        return "Black wins!"
    else:
        return "It's a draw."

def main():
    play_chess()

main()
