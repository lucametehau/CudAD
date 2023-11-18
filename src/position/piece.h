/**
    CudAD is a CUDA neural network trainer, specific for chess engines.
    Copyright (C) 2022 Finn Eggers

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef BINARYPOSITIONWRAPPER__PIECE_H_
#define BINARYPOSITIONWRAPPER__PIECE_H_

#include "defs.h"

enum Colors { BLACK = false, WHITE = true, N_COLORS = 2 };

enum Sides { KING_SIDE, QUEEN_SIDE, N_SIDES = 2 };

enum PieceTypes {
    PAWN          = 0,
    KNIGHT        = 1,
    BISHOP        = 2,
    ROOK          = 3,
    QUEEN         = 4,
    KING          = 5,
    N_PIECE_TYPES = 6
};

enum Pieces {
    NO_PIECE     = -1,
    BLACK_PAWN   = 0,
    BLACK_KNIGHT = 1,
    BLACK_BISHOP = 2,
    BLACK_ROOK   = 3,
    BLACK_QUEEN  = 4,
    BLACK_KING   = 5,
    WHITE_PAWN   = 8,
    WHITE_KNIGHT = 9,
    WHITE_BISHOP = 10,
    WHITE_ROOK   = 11,
    WHITE_QUEEN  = 12,
    WHITE_KING   = 13,
    N_PIECES     = 14
};

constexpr char
    piece_identifier[] {'p', 'n', 'b', 'r', 'q', 'k', ' ', ' ', 'P', 'N', 'B', 'R', 'Q', 'K'};

inline Color     getPieceColor(Piece p) { return p & 0x8; }

inline PieceType getPieceType(Piece p) { return p & 0x7; }

inline Piece     getPiece(Color c, PieceType pt) { return c * 8 + pt; }

#endif    // BINARYPOSITIONWRAPPER__PIECE_H_
