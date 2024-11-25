#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <limits>
#include "include/chess.hpp"

// namespace
using namespace chess;
// fix for hash error (specialized)
namespace std {
    template <>
    struct hash<chess::PieceType> {
        std::size_t operator()(const chess::PieceType& piece) const noexcept {
            return static_cast<std::size_t>(piece);
        }
    };
}
using namespace std;

// piece values
std::unordered_map<chess::PieceType, int> pieceMap = {
        {chess::PieceType::KING, 0},
        {chess::PieceType::QUEEN, 9},
        {chess::PieceType::PAWN, 1},
        {chess::PieceType::KNIGHT, 3},
        {chess::PieceType::BISHOP, 3},
        {chess::PieceType::ROOK, 5}
};

// evaluate board
int evalBoard(const Board &board, Color botColor) {
    Color opponentColor = botColor == Color::WHITE ? Color::BLACK : Color::WHITE;
    int score = 0;

    // iterate over all squares on the board
    for (int sq = 0; sq < 64; ++sq) { // 64 squares on a chessboard
        Square square = static_cast<Square>(sq); // convert index to Square
        Piece piece = board.at(square);          // get the piece at this square

        if (piece.internal() != Piece::NONE) {   // if there's a piece
            PieceType type = piece.type();       // get the piece type
            Color color = piece.color();         // get the piece color

            int value = pieceMap[type];      // piece value
            score += (color == botColor) ? value : -value; // add/subtract based on color
        }
    }

    return score;
}

// minimax algorithm
pair<int, Move> minimax(Board &board, int depth, bool isMaximizing, Color botColor, int alpha, int beta) {
    auto [resultReason, result] = board.isGameOver();
    // base case
    if (depth == 0 || resultReason != GameResultReason::NONE) {
        return {evalBoard(board, botColor), Move()};
    }

    // generate move
    Move bestMove;
    Movelist moves;
    movegen::legalmoves(moves, board);

    if (isMaximizing) {
        int maxEval = numeric_limits<int>::min();

        // for all the moves in legal moves
        for (const auto &move : moves) {
            board.makeMove(move);
            auto [eval, _] = minimax(board, depth - 1, false, botColor, alpha, beta);
            board.unmakeMove(move);

            if (eval > maxEval) {
                maxEval = eval;
                bestMove = move;
            }

            // alpha beta implementation
            alpha = max(alpha, eval);
            if (beta <= alpha) break; // beta cut-off
        }

        return {maxEval, bestMove};
    } else {
        int minEval = numeric_limits<int>::max();

        for (const auto &move : moves) {
            board.makeMove(move);
            auto [eval, _] = minimax(board, depth - 1, true, botColor, alpha, beta);
            board.unmakeMove(move);

            if (eval < minEval) {
                minEval = eval;
                bestMove = move;
            }

            beta = min(beta, eval);
            if (beta <= alpha) break; // alpha cut-off
        }

        return {minEval, bestMove};
    }
}

// bot move using minimax
Move botMove(Board &board, Color botColor, int depth = 3) {
    int alpha = std::numeric_limits<int>::min(); // initial alpha
    int beta = std::numeric_limits<int>::max(); // initial beta

    auto [_, bestMove] = minimax(board, depth, true, botColor, alpha, beta);
    return bestMove;
}

// switch color turn
Color switchTurn(Color currentTurn) {
    return (currentTurn == Color::WHITE) ? Color::BLACK : Color::WHITE;
}


// play chess
void playChess() {
    // user input
    std::string fen;
    cout << "Enter computer player color (w=white, b=black): ";
    char botColorInput;
    cin >> botColorInput;

    // color
    Color botColor = (botColorInput == 'w') ? Color::WHITE : Color::BLACK;
    Color userColor = (botColor == Color::WHITE) ? Color::BLACK : Color::WHITE;
    Color currentTurn = Color::WHITE;

    cout << "Starting FEN position? (hit ENTER for default): ";
    cin.ignore();
    getline(cin, fen);

    Board board = (fen.empty()) ? Board() : Board(fen);

    auto [resultReason, result] = board.isGameOver();

    // game loop
    while (resultReason == GameResultReason::NONE) {
        cout << board << endl;

        // bot turn
        if (currentTurn == botColor) {
            Move botMoveChoice = botMove(board, botColor);
            board.makeMove(botMoveChoice);
            std::cout << "Bot move: " << uci::moveToUci(botMoveChoice) << std::endl;
        } else {
            cout << "Your move: ";
            string userMoveStr;
            cin >> userMoveStr;

            try {
                // convert str to uci move
                Move userMove = uci::uciToMove(board, userMoveStr);
                board.makeMove(userMove);
            } catch (const std::invalid_argument &e) {
                cout << "Invalid move. Try again.\n";
            }
        }
        // switch color if game is finished (haven't tested) to display correct color at end
        if(resultReason == GameResultReason::NONE){
            currentTurn = switchTurn(currentTurn);
        }
        // check if game is finished
        auto [resultReason, result] = board.isGameOver();
    }

    // if game is over
    if (resultReason != GameResultReason::NONE) {
        std::cout << "Game Over: ";

        // print the reason for game over
        switch (resultReason) {
            case GameResultReason::CHECKMATE:
                cout << (currentTurn == botColor ? "You win!" : "Bot wins!") << endl;
                break;
            case GameResultReason::STALEMATE:
                std::cout << "Stalemate! ";
                break;
            case GameResultReason::INSUFFICIENT_MATERIAL:
                std::cout << "Draw due to insufficient material. ";
                break;
            case GameResultReason::FIFTY_MOVE_RULE:
                std::cout << "Draw due to fifty-move rule. ";
                break;
            case GameResultReason::THREEFOLD_REPETITION:
                std::cout << "Draw due to threefold repetition. ";
                break;
            default:
                std::cout << "Unknown reason.";
                break;
        }
    }
}

int main() {
    playChess();
    return 0;
}
