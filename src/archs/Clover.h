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

#ifndef CUDAD_SRC_MAPPINGS_CLOVER_H_
#define CUDAD_SRC_MAPPINGS_CLOVER_H_

#include "../activations/ReLU.h"
#include "../activations/Sigmoid.h"
#include "../data/SArray.h"
#include "../data/SparseInput.h"
#include "../dataset/dataset.h"
#include "../layer/DenseLayer.h"
#include "../loss/Loss.h"
#include "../loss/MPE.h"
#include "../optimizer/Adam.h"
#include "../optimizer/AdamW.h"
#include "../optimizer/Optimiser.h"
#include "../position/fenparsing.h"
#include "../layer/Input.h"
#include "../layer/ActivationLayer.h"
#include "../layer/MergeLayer.h"
#include "../network/Network.h"
#include "../layer/PairwiseMultiplyLayer.h"
#include <iostream>
#include <tuple>
#include <cassert>

class Clover {
    public:
    static constexpr int   Inputs        = 8 * 12 * 64;
    static constexpr int   L2            = 1024;
    static constexpr int   Outputs       = 1;
    static constexpr float SigmoidScalar = 0.00447111749925f;

    using LayerList = std::vector<LayerInterface*>;

    static Optimiser*      get_optimiser() {
        Adam* optim  = new Adam();
        optim->lr    = 0.01;
        optim->beta1 = 0.95;
        optim->beta2 = 0.999;

        optim->schedule.gamma = 0.1;
        optim->schedule.step  = 250;

        return optim;
    }

    static Loss* get_loss_function() {
        MPE* loss_f = new MPE(2.5, false);

        return loss_f;
    }

    static std::tuple<LayerList, LayerList> get_layers() {

        // positional inputs
        auto i1 = new Input(true, Inputs, 32);
        auto i2 = new Input(true, Inputs, 32);

        // both accumulators merged + relu
        auto h1 = new DenseLayer<L2>(i1);
        //h1->lasso_regularization = 1.0 / 8388608.0;
        auto h2 = new DenseLayer<L2>(i2, h1);
        auto m1 = new MergeLayer(h1,h2);

        // hidden of main network
        auto a1 = new ActivationLayer<ReLU>(m1);
        auto h3 = new DenseLayer<Outputs>(a1);

        auto a2 = new ActivationLayer<Sigmoid>(h3);

        a2->f.scalar = SigmoidScalar;

        return {
            {i1,i2},
            {h1,h2,m1,a1,h3,a2}
        };
    }

    static void assign_inputs_batch(DataSet&       positions,
                                    Network&       network,
                                    SArray<float>& output,
                                    SArray<bool>&  output_mask) {

//        ASSERT(positions.positions.size() == in1.n);
//        ASSERT(positions.positions.size() == in2.n);

        SparseInput& in1 = network.getInputs()[0]->sparse_data;
        SparseInput& in2 = network.getInputs()[1]->sparse_data;
//        DenseMatrix& in3 = network.getInputs()[2]->dense_data.values;

        in1.clear();
        in2.clear();
//        in3.clear();
//        output_mask.clear<HOST>();

#pragma omp parallel for schedule(static) num_threads(8)
        for (int i = 0; i < positions.positions.size(); i++)
            assign_input(positions.positions[i], in1, in2, output, output_mask, i);


    }

    static int king_square_index(Square relative_king_square) {
        // clang-format off
        //return 0;
        constexpr int indices[N_SQUARES] {
            3, 2, 1, 0, 0, 1, 2, 3,
            3, 2, 1, 0, 0, 1, 2, 3,
            5, 5, 4, 4, 4, 4, 5, 5,
            5, 5, 4, 4, 4, 4, 5, 5,
            6, 6, 6, 6, 6, 6, 6, 6,
            6, 6, 6, 6, 6, 6, 6, 6,
            7, 7, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 7, 7, 7,
        };
        /*constexpr int indices[N_SQUARES] {
            1, 1, 0, 0, 0, 0, 1, 1,
            1, 1, 0, 0, 0, 0, 1, 1,
            2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2,
            3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 3, 3,
            4, 4, 4, 4, 4, 4, 4, 4,
            4, 4, 4, 4, 4, 4, 4, 4,
        };*/
        // clang-format on

        return indices[relative_king_square];
    }

    static int index(Square psq, Piece piece, Square kingSquare, Color view) {
        if(view == BLACK) {
            kingSquare ^= 56;
            psq ^= 56;
            piece = (piece >= 8 ? piece - 8 : piece + 8);
        }

        if (fileIndex(kingSquare) > 3) {
            kingSquare ^= 7;
            psq ^= 7;
        }
        return 8 * 64 * (6 * getPieceColor(piece) + getPieceType(piece)) + 64 * king_square_index(kingSquare) + psq;
    }

    static void assign_input(Position&      p,
                             SparseInput&   in1,
                             SparseInput&   in2,
                             SArray<float>& output,
                             SArray<bool>&  output_mask,
                             int            id) {

        // track king squares
        Square wKingSq = p.getKingSquare<WHITE>();
        Square bKingSq = p.getKingSquare<BLACK>();

        BB     bb {p.m_occupancy};
        int    idx = 0;

        while (bb) {
            Square sq                    = bitscanForward(bb);
            Piece  pc                    = p.m_pieces.getPiece(idx);

            auto   piece_index_white_pov = index(sq, pc, wKingSq, WHITE);
            auto   piece_index_black_pov = index(sq, pc, bKingSq, BLACK);

            //std::cout << "Piece is " << int(pc) << ", square is " << int(sq) << " indexes are " << piece_index_white_pov << " " << piece_index_black_pov << "\n";

            if (p.m_meta.getActivePlayer() == WHITE) {
                in1.set(id, piece_index_white_pov);
                in2.set(id, piece_index_black_pov);
            } else {
                in2.set(id, piece_index_white_pov);
                in1.set(id, piece_index_black_pov);
            }

            bb = lsbReset(bb);
            idx++;
        }

        float p_value = p.m_result.score;
        float w_value = p.m_result.wdl;

        // flip if black is to move -> relative network style
        if (p.m_meta.getActivePlayer() == BLACK) {
            //p_value = -p_value;
            w_value = -w_value;
        }

        float p_target = 1 / (1 + expf(-p_value * SigmoidScalar));
        float w_target = (w_value + 1) / 2.0f;

        const int bucket = 0;

        output     (id * Outputs + bucket) = p_target * 0.7 + w_target * 0.3;
        output_mask(id * Outputs + bucket) = true;
    }
};

#endif
