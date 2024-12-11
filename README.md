# Implementing Minimax Chessbot on CUDA
## This is a final team project for CSC 220: Advanced Programming

The project implements a Minimax chessbot using C++ and CUDA, leveraging the SHL chess library for efficient chess functionality. The aim is to optimize the Minimax calculations, originally performed in Python, and parallelize them for enhanced performance. The C++ implementation features recursive Minimax calls and incorporates alpha-beta pruning to improve computational efficiency.

To compile and see the results on CPP, run noCuda.cpp file and call output file.
```
g++ noCuda.cpp -o chessCPP -std=c++17
./chessCPP
```
Similarly, run Python file to see minimax chess bot on Python. 
```
python chessBot.py
```
As for direct CUDA implementation, run the following. 
```
nvcc testImplement.cu -o chessTree -std=c++17 --expt-relaxed-constexpr
./testChess
```

## What is minimax?
Minimax is a chess algorithm that we learned about and implemented in Python in CSC 290. It recursively searches through all the possible moves both players could choose to a certain specified depth, and then evaluates all the resulting leaf node game states. The evaluation simply adds up all the pieces on the board with different weights depending on how important the piece is. The values were obtained from https://www.chess.com/terms/chess-piece-value: pawns are 1 point, knights and bishops are 3, rooks are 5, and the queen is 9. Kings are valued at 0 because they cannot be captured. In minimax, both sides' pieces are added up, and opposition color pieces (black or white) have negative points while minimax pieces (e.g. white if opposition is black) have positive points. Once we have the evaluation, we work our way back up the game states tree, where the opposition bot always chooses the optimal move for it: the minimizing move (the move where the evaluation score is lowest) and the minimax bot always chooses the maximizing move (the move where the evaluation score is highest). So this algorithm assumes that both players will choose their best moves available. It doesn't have strategies for getting checkmate except to capture many/valuable pieces. 

## Our setup
We used the Disservin C++ chess library (https://disservin.github.io/chess-library/) for representing the chess board, getting legal moves, implementing a gameplay function, etc. First we implemented minimax on the CPU with alpha-beta pruning, which decreases the amount of searching the algorithm has to do by not searching branches that the algorithm has already determined to be fruitless. 

A paper has already been published on using the GPU to implement Reversi (Othello) game minimax using CUDA, "Parallel Minimax Tree Searching on GPU" by Kamil Rocki and Reiji Suda. They first ran minimax on the CPU, for a certain depth, saved the leaf node game states, passed those to a kernel where each thread takes a game state and runs a modified version of minimax to several more depths, then returns the best moves and evaluations to the CPU. Then the CPU recursively determines the best move from the evaluations from the GPU. 

We decided to attempt a simplified version of Rocki and Suda's implementation: we would run minimax on the CPU, save the leaf node game states, send those to the GPU, evaluate them all at once in parallel, send the evaluations back to the CPU, and have the CPU perform the minimizing/maximizing part of the algorithm to get the optimal next move. Unfortunately, we ran out of time working on the simplified version and were not able to finish this implementation. Had we been able to successfully implement this simplified version, we would have also tried to implement the more complicated version.

## Runtime analysis
 We conducted runtime analysis using profilers like, nvidia prof on CUDA direct implementation, gprof on C++, and Python timer functions. Notice that at depth 5, Python exceeds 2 minutes while at depth 7, C++ and CUDA direct implementations are under 10 seconds. See the table below for detailed timestamps.
 
 ![Table of Runtime](https://github.com/Tseegi-N/Minimax-Chess-on-CUDA/blob/master/graphs/Screenshot 2024-12-11 110235.png)

For better analysis, we made two graphs with the timestamp data from runtime analysis. Y-axis represents runtime in seconds. The figure on the right represents chess bot runtime without any alterations. Python skyrockets after depth 3 due to the overwhelming amount of nodes to evaluate at deeper depths. On the other hand, C++ and CUDA implementations remain low even at depths like 7. The log-scale graph on the left shows that the rates of change between different depths of C++ and CUDA implementations are noticeably lower than Python chess bot. It’s important to note that CUDA implementation runtime remains lower than C++ at all levels of depth, probably due to better keroppi GPU performance compared to our local CPU despite added duration when copying and allocating data into the GPU. 

![Graph Normal](https://github.com/Tseegi-N/Minimax-Chess-on-CUDA/blob/master/graphs/Minimax Chessbot Runtime.png)
![Graph Log-scale](https://github.com/Tseegi-N/Minimax-Chess-on-CUDA/blob/master/graphs/Minimax Chessbot Runtime (Log-scale).png)

## Discussion
We also considered trying to parallelize our minimax on the CPU, but we decided to do it on the GPU because that is what we were more familiar with. Perhaps it would have been easier after all to parallelize on the CPU after (OpenMP and OpenCL were mentioned in class), this way we would have been able to do some recursion in parallel and not have to worry about passing chess-library functions to the GPU or making sure to use CUDA-compatible types on the CPU. But we decided that learning to use C++ and the chess library was enough of a challenge for this project, which brings us to our next point of discussion: we gained experience coding in C++. The namespace syntax took some getting used to, and our learning was accelerated by having to look directly into the chess-library header file to find and use chess-library functions. The chess-library didn't have very good documentation, despite advertising otherwise. 

Further improvements include passing the game states tree to a function that finds the maximum/minimum from the bottom up, so that instead of calling minimax twice, we utilize the tree structure. Finally, this program could also be improved by implementing minimax on the GPU as Rocki and Suda did. This would require us to alter minimax so that instead of recursively calling a minimax function implementing a modified iterative version of the algorithm, which can be parallelized on the GPU without worrying about recursion, which is further discussed in “An analysis of alpha-beta pruning” by Donald E. Knuth,  Ronald W. Moore. 

Overall, despite being unable to finish implementing minimax using the GPU, this project was a good learning experience. We planned our CPU and GPU code, considered different ways of parallelizing minimax, did lots of coding in C++, and got to combine what we learned from CSC 290 and 220. Even the switch to C++ from Python drastically improved runtime, and so did just evaluating the leaf nodes on the GPU as we recursively worked our way through the game states tree on the CPU.

## Citations
Knuth, D. E., & Moore, R. W. (1975). An analysis of alpha-beta pruning. Artificial intelligence, 6(4), 293-326.

Rocki, K., & Suda, R. (2009, September). Parallel minimax tree searching on GPU. In International Conference on Parallel Processing and Applied Mathematics (pp. 449-456). Berlin, Heidelberg: Springer Berlin Heidelberg.

# An extensive SHL Chess Library for C++

[![Chess Library](https://github.com/Disservin/chess-library/actions/workflows/chess-library.yml/badge.svg)](https://github.com/Disservin/chess-library/actions/workflows/chess-library.yml)

## [Documentation](https://disservin.github.io/chess-library)

**chess-library** is a multi-purpose library for chess in C++17.

It can be used for any type of chess program, be it a chess engine, a chess GUI, or a chess data anaylsis tool.

### Why this library?

- **Fast**: This library is fast enough for pretty much any purpose in C++ and it is faster than most other chess libraries in C++.
- **Documentation**: Easy to browse **documentation** at https://disservin.github.io/chess-library
- **Robust**: Unit Tests & it has been tested on millions of chess positions, while developing the C++ part of [Stockfish's Winrate Model](https://github.com/official-stockfish/WDL_model).
- **PGN Support**: Parse basic PGN files.
- **Namespace**: Everything is in the `chess::` namespace, so it won't pollute your namespace.
- **Compact Board Representation in 24bytes**: The board state can be compressed into 24 bytes, using `PackedBoard` and `Board::Compact::encode`/`Board::Compact::decode`.

### Usage

This is a single header library.

You only need to include `chess.hpp` header!
Aftewards you can access the chess logic over the `chess::` namespace.

### Comparison to other chess libraries

The 3 other big chess libraries that I know of in C++ are:

- [surge](https://github.com/nkarve/surge)

  Pros:

  - relatively fast, see [PERFT Comparison](#perft-comparison)

  Cons:

  - lacks documentation and utility functionality, also no support for Chess960.
  - not very active anymore

- [THC](https://github.com/billforsternz/thc-chess-library)

  Pros:

  - Header and source file split, leading to faster compilation times.
  - Position compression

  Cons:

  - Rather slow, see [PERFT Comparison](#perft-comparison)
  - Lacks documentation
  - No support for Chess960

- [chessgen](https://github.com/markhc/chessgen)

  Pros:

  - Modern C++ (and relatively active)
  - Header Only

  Cons:

  - No documentation
  - Early Version (after 4 years)
  - No support for Chess960 (I think)

### Repositories using this library:

- Stockfish Winrate Model
  https://github.com/official-stockfish/WDL_model
- CLI Tool for running chess engine matches
  https://github.com/Disservin/fast-chess
- GUI-based Chess Player as well as a Chess Engine
  https://github.com/Orbital-Web/Raphael
- UCI Chess Engine (\~3.3k elo)
  https://github.com/rafid-dev/rice (old version)
- Texel tuner for HCE engines
  https://github.com/GediminasMasaitis/texel-tuner

### Benchmarks

Tested on Ryzen 9 5950X:

With movelist preallocation:

#### Standard Chess

```
depth 7  time 8988  nodes 3195901860   nps 355534749 fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
depth 5  time 430   nodes 193690690    nps 449398352 fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1
depth 7  time 661   nodes 178633661    nps 269839367 fen 8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1
depth 6  time 1683  nodes 706045033    nps 419266646 fen r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1
depth 5  time 210   nodes 89941194     nps 426261582 fen rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8
depth 5  time 377   nodes 164075551    nps 434062304 fen r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 1
```

#### Chess960

```
depth 6  time 358   nodes 119060324    nps 331644356 fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w HAha - 0 1
depth 6  time 710   nodes 191762235    nps 269707784 fen 1rqbkrbn/1ppppp1p/1n6/p1N3p1/8/2P4P/PP1PPPP1/1RQBKRBN w FBfb - 0 9
depth 6  time 2434  nodes 924181432    nps 379540629 fen rbbqn1kr/pp2p1pp/6n1/2pp1p2/2P4P/P7/BP1PPPP1/R1BQNNKR w HAha - 0 9
depth 6  time 927   nodes 308553169    nps 332492639 fen rqbbknr1/1ppp2pp/p5n1/4pp2/P7/1PP5/1Q1PPPPP/R1BBKNRN w GAga - 0 9
depth 6  time 2165  nodes 872323796    nps 402734901 fen 4rrb1/1kp3b1/1p1p4/pP1Pn2p/5p2/1PR2P2/2P1NB1P/2KR1B2 w D - 0 21
depth 6  time 6382  nodes 2678022813   nps 419555508 fen 1rkr3b/1ppn3p/3pB1n1/6q1/R2P4/4N1P1/1P5P/2KRQ1B1 b Ddb - 0 14
```

#### Exceptions

This library might throw exceptions in some cases, for example when the input is invalid or things are not as expected.
To disable exceptions, define `CHESS_NO_EXCEPTIONS` before including the header.

#### PERFT Comparison

[Benchmark implementation](./comparison/benchmark.cpp) for more information.

`chess-library` had no movelist preallocation, same as the other libraries.

```
chess-library:
depth 6  time 539   nodes 119060324    nps 220482081 fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
depth 5  time 538   nodes 193690690    nps 359351929 fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1
depth 6  time 64    nodes 11030083     nps 169693584 fen 8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1
depth 5  time 53    nodes 15833292     nps 293209111 fen r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1
depth 5  time 267   nodes 89941194     nps 335601470 fen rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8
depth 5  time 468   nodes 164075551    nps 349841260 fen r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 11

Surge:
depth 6  time 713   nodes 119060324    nps 166751154 fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
depth 5  time 841   nodes 193690690    nps 230036448 fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1
depth 6  time 75    nodes 11030083     nps 145132671 fen 8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1
depth 5  time 85    nodes 15833292     nps 184108046 fen r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1
depth 5  time 419   nodes 89941194     nps 214145700 fen rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8
depth 5  time 770   nodes 164075551    nps 212808756 fen r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 11

THC:
depth 6  time 3294  nodes 119060324    nps 36133633  fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
depth 5  time 5043  nodes 193690690    nps 38400216  fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1
depth 6  time 404   nodes 11030083     nps 27234772  fen 8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1
depth 5  time 388   nodes 15833292     nps 40702550  fen r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1
depth 5  time 2909  nodes 89941194     nps 30907626  fen rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8
depth 5  time 3403  nodes 164075551    nps 48200808  fen r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 11
```

### Development Setup

This project is using the meson build system. https://mesonbuild.com/

#### Setup

```bash
meson setup build
```

#### Compilation

```bash
meson compile -C build
```

#### Tests

```bash
meson test -C build
```

#### Example

Download the [Lichess March 2017 database](https://database.lichess.org/standard/lichess_db_standard_rated_2017-03.pgn.zst) and place it in the parent directory where you've checked out this repository.
You can decompress this with the following command: `unzstd -d lichess_db_standard_rated_2017-03.pgn.zst`

```bash
cd example
meson setup build
meson compile -C build

./build/example
```
