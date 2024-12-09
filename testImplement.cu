#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <limits>
#include <cstdlib> //for random number gen
#include <sys/time.h>
#include "include/chess.hpp"
#include <cuda_runtime.h>

#define DEPTH 3

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

int numEvals = 0;

// piece values
std::unordered_map<chess::PieceType, int> pieceMap = {
        {chess::PieceType::KING, 9},
        {chess::PieceType::QUEEN, 9},
        {chess::PieceType::PAWN, 1},
        {chess::PieceType::KNIGHT, 3},
        {chess::PieceType::BISHOP, 3},
        {chess::PieceType::ROOK, 5}
};

double get_clock(){
	struct timeval tv; int ok;
	ok = gettimeofday(&tv, (void *) 0);
	if (ok<0) { printf("gettimeofday error"); }
	return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

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
    numEvals +=1;
    return score;
}

// eval board on the gpu
__global__ void evalBoardKernel(int* scores, const Board &board, int numMoves, Color botColor) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numMoves) {
        int score = 0;
        // evaluate each piece
        for (int sq = 0; sq < 64; ++sq) { // 64 squares on a chessboard
            Square square = static_cast<Square>(sq); // convert index to Square
            Piece piece = board.at(square);          // get the piece at this square

            if (piece.internal() != Piece::NONE) {   // if there's a piece
                PieceType type = piece.type();       // get the piece type
                Color color = piece.color();         // get the piece color

                int value;      // piece value

                switch(type) {
                    case static_cast<int>(chess::PieceType::KING):
                        value = 0;
                        break;
                    case static_cast<int>(chess::PieceType::QUEEN):
                        value = 9;
                        break;
                    case static_cast<int>(chess::PieceType::PAWN):
                        value = 1;
                        break;
                    case static_cast<int>(chess::PieceType::KNIGHT):
                        value = 3;
                        break;
                    case static_cast<int>(chess::PieceType::BISHOP):
                        value = 3;
                        break;
                    case static_cast<int>(chess::PieceType::ROOK):
                        value = 5;
                        break;
                }
                score += (color == botColor) ? value : -value; // add/subtract based on color
            }
        }
        scores[idx] = score;
    }
}

// prepare evalboard for the gpu
int evalBoardParallel(const Board &boards, Color botColor) {
    int numMoves = 32;
    int* d_scores;
    int* d_boards;

    // allocate device memory
    cudaMalloc(&d_scores, numMoves * sizeof(int));
    cudaMalloc(&d_boards, numMoves * 64 * sizeof(int));

    // copy boards to device
    //cudaMemcpy(d_boards, boards.data(), numMoves * 64 * sizeof(int), cudaMemcpyHostToDevice);

    // launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numMoves + threadsPerBlock - 1) / threadsPerBlock;
    evalBoardKernel<<<blocksPerGrid, threadsPerBlock>>>(d_scores, boards, numMoves, botColor);
    cudaDeviceSynchronize();

    // copy results back
    std::vector<int> scores(numMoves);
    cudaMemcpy(scores.data(), d_scores, numMoves * sizeof(int), cudaMemcpyDeviceToHost);

    // free up space
    cudaFree(d_scores);
    cudaFree(d_boards);

    // return results
    return *std::max_element(scores.begin(), scores.end());
}


//randomly chooses capture moves
Move random(Board &board){
	Movelist moves;
	Move move;
	movegen::legalmoves<movegen::MoveGenType::CAPTURE>(moves, board); //generate capture moves
	if (moves.empty())
		movegen::legalmoves<movegen::MoveGenType::QUIET>(moves, board);
	move = moves[rand() % moves.size()]; //get rand move
	return move;
}

// minimax algorithm
pair<int, Move> minimax(Board &board, int depth, bool isMaximizing, Color botColor, int alpha, int beta) {
    auto [resultReason, result] = board.isGameOver();
    // base case
    if (depth == 0 || resultReason != GameResultReason::NONE) {
        int eval = evalBoardParallel(board, botColor);
        return {eval, Move()};
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
Move botMove(Board &board, Color botColor, int depth = DEPTH) {
    int alpha = std::numeric_limits<int>::min(); // initial alpha
    int beta = std::numeric_limits<int>::max(); // initial beta
	numEvals = 0;

	double t0 = get_clock();
    auto [_, bestMove] = minimax(board, depth, true, botColor, alpha, beta);
	double t1 = get_clock();
    printf("time: %f s, numEvals: %d, evals/s: %f\n", t1-t0, numEvals, numEvals/(t1-t0) );
    return bestMove;
}

// switch color turn
Color switchTurn(Color currentTurn) {
    return (currentTurn == Color::WHITE) ? Color::BLACK : Color::WHITE;
}


// play chess
void playChess() {
    // user input
     char botColorInput;
     std::string fen;
     cout << "Enter computer player color (w=white, b=black, o=bots play): ";
     cin >> botColorInput;

    // color
    Color botColor = (botColorInput == 'w') ? Color::WHITE : Color::BLACK;	//if o/bots play, minimax bot is black, random is white
    Color userColor = (botColor == Color::WHITE) ? Color::BLACK : Color::WHITE;
    Color currentTurn = Color::WHITE;

    cout << "Starting FEN position? (hit ENTER for default): ";
    cin.ignore();
    getline(cin, fen);

    Board board = (fen.empty()) ? Board() : Board(fen);

    pair<GameResultReason, GameResult> results_pair = board.isGameOver();

    // game loop
    while (results_pair.first == GameResultReason::NONE){
        cout << board << endl;

        // bot turn
        if (currentTurn == botColor) {
            Move botMoveChoice = botMove(board, botColor);
            board.makeMove(botMoveChoice);
            std::cout << "Minimax bot move: " << uci::moveToUci(botMoveChoice) << std::endl;
        } else if (botColorInput == 'o') {
        	Move randomBotMove = random(board);
        	board.makeMove(randomBotMove);
        	std::cout << "Random bot move: " << uci::moveToUci(randomBotMove) << std::endl;
        } else {
        	bool tryAgain = true;
        	while(tryAgain == true){
	            cout << "Your move: ";
	            string userMoveStr;
	            cin >> userMoveStr;
	            Move userMove = uci::uciToMove(board, userMoveStr);

	            Movelist legal_moves;
	            movegen::legalmoves(legal_moves,board);

	            for (int i = 0; i < legal_moves.size(); i++){
	            	if (legal_moves[i] == userMove){
	            		tryAgain = false;
	            		break;
	            	}
	            }
		       	if (tryAgain == true)	cout << "Invalid move. Try again.\n";
	        }
        }
        // switch color if game is finished (haven't tested) to display correct color at end
        if(results_pair.first == GameResultReason::NONE){
            currentTurn = switchTurn(currentTurn);
        }
        // check if game is finished
        results_pair = board.isGameOver();
        std::cout << "GameResultReason: " << results_pair.first << "\n";
        std::cout << "GameResult: " << results_pair.second << "\n";
        std::cout << "Current Turn: " << currentTurn << "\n";
    }

    // if game is over
    if (results_pair.first != GameResultReason::NONE) {
        std::cout << "Game Over: ";

        // print the reason for game over
        switch (results_pair.first) {
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
