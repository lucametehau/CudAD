/****************************************************************************************************
 *                                                                                                  *
 *                                                FFES                                              *
 *                                          by. Finn Eggers                                         *
 *                                                                                                  *
 *                    FFES is free software: you can redistribute it and/or modify                  *
 *                it under the terms of the GNU General Public License as published by              *
 *                 the Free Software Foundation, either version 3 of the License, or                *
 *                                (at your option) any later version.                               *
 *                       FFES is distributed in the hope that it will be useful,                    *
 *                   but WITHOUT ANY WARRANTY; without even the implied warranty of                 *
 *                   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                  *
 *                            GNU General Public License for more details.                          *
 *                 You should have received a copy of the GNU General Public License                *
 *                   along with FFES.  If not, see <http://www.gnu.org/licenses/>.                  *
 *                                                                                                  *
 ****************************************************************************************************/

//
// Created by Luecx on 10.11.2021.
//

#ifndef DIFFERENTIATION_SRC_ACITVATIONS_RELU_H_
#define DIFFERENTIATION_SRC_ACITVATIONS_RELU_H_

#include "../data/DenseMatrix.h"
#include "../data/SArray.h"
#include "Activation.h"

struct ReLU : Activation {

    // clang-format off
    void apply      (const SArray<float> &in,
                           SArray<float> &out, Mode mode) ;
    void backprop   (const SArray<float> &in,
                           SArray<float> &in_grd,
                     const SArray<float> &out,
                     const SArray<float> &out_grd, Mode mode) ;
    // clang-format on

    void logOverview() override;
};

#endif    // DIFFERENTIATION_SRC_ACITVATIONS_RELU_H_
