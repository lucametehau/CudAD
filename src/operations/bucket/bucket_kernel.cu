
//
// Created by Luecx on 30.01.2022.
//
#include "bucket.h"

__global__ void bucket_kernel(
    const float* __restrict__ inp,
          float* __restrict__ out,
          float max_lower_bucket,
          float min_upper_bucket,
          int buckets,
          int input_size){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= input_size) return;

    float bucket_size = (min_upper_bucket - max_lower_bucket) / (buckets - 2);

    int inner_bucket_idx = ceil((inp[idx] - max_lower_bucket) / bucket_size);

    if(inner_bucket_idx < 0) inner_bucket_idx = 0;
    if(inner_bucket_idx >= buckets) inner_bucket_idx = buckets-1;

    // clear the output
    int outp_offset = buckets * idx;
    for(int i = 0; i < buckets; i++){
        out[outp_offset + i] = 0;
    }

    // set the correct value
    out[outp_offset + inner_bucket_idx] = inp[idx];
}