
//
// Created by Luecx on 14.01.2022.
//

#ifndef CUDAD_SRC_OPERATIONS_MM_BP_MM_BP_H_
#define CUDAD_SRC_OPERATIONS_MM_BP_MM_BP_H_

#include "../mm/mm_cublas.h"

// clang-format off
template<Mode mode>
inline void mm_bp(DenseMatrix &mat1,
        DenseMatrix &mat1_grd,
        DenseMatrix &mat2,
        DenseMatrix &mat2_grd,
        DenseMatrix &res_grd){

    ASSERT(mat1_grd.n == mat2_grd.m);
    ASSERT(mat1_grd.m == res_grd .m);
    ASSERT(mat2_grd.n == res_grd .n);

    if(mode == DEVICE){

        mm_cublas(
            res_grd,
            mat2,
            mat1_grd,
            1,          // alpha = 1
            0,          // beta = 0
            false,      // transpose weights
            true);      //

        mm_cublas(
            mat1,
            res_grd,
            mat2_grd,
            1,          // alpha = 1
            0,          // beta = 0
            true,       // transpose res_grd
            false);     //

    }else{
        ASSERT(false);
    }
}
// clang-format on
#endif    // CUDAD_SRC_OPERATIONS_MM_BP_MM_BP_H_
