#include "include/chess.hpp"

using namespace chess;

int main () {
    Board board = Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    Movelist moves;
    movegen::legalmoves(moves, board);

    for (const auto &move : moves) {
        std::cout << uci::moveToUci(move) << std::endl;
    }

    // Check if the game is over
    auto [resultReason, result] = board.isGameOver();

    if (resultReason != GameResultReason::NONE) {
        std::cout << "Game Over: ";

        // Print the reason for game over
        switch (resultReason) {
            case GameResultReason::CHECKMATE:
                std::cout << "Checkmate! ";
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

    return 0;
}
