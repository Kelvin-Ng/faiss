#pragma once

#include "../utils/LoadStoreOperators.cuh"
#include "../utils/StaticUtils.h"

namespace faiss { namespace gpu {

template <typename LookupT, typename LookupVecT>
struct LoadCodeDistances {
  static inline __device__ void load(LookupT* smem,
                                     LookupT* codes,
                                     int numCodes) {
    constexpr int kWordSize = sizeof(LookupVecT) / sizeof(LookupT);

    // We can only use the vector type if the data is guaranteed to be
    // aligned. The codes are innermost, so if it is evenly divisible,
    // then any slice will be aligned.
    if (numCodes % kWordSize == 0) {
      // Load the data by float4 for efficiency, and then handle any remainder
      // limitVec is the number of whole vec words we can load, in terms
      // of whole blocks performing the load
      constexpr int kUnroll = 2;
      int limitVec = numCodes / (kUnroll * kWordSize * blockDim.x);
      limitVec *= kUnroll * blockDim.x;

      LookupVecT* smemV = (LookupVecT*) smem;
      LookupVecT* codesV = (LookupVecT*) codes;

      for (int i = threadIdx.x; i < limitVec; i += kUnroll * blockDim.x) {
        LookupVecT vals[kUnroll];

#pragma unroll
        for (int j = 0; j < kUnroll; ++j) {
          vals[j] =
            LoadStore<LookupVecT>::load(&codesV[i + j * blockDim.x]);
        }

#pragma unroll
        for (int j = 0; j < kUnroll; ++j) {
          LoadStore<LookupVecT>::store(&smemV[i + j * blockDim.x], vals[j]);
        }
      }

      // This is where we start loading the remainder that does not evenly
      // fit into kUnroll x blockDim.x
      int remainder = limitVec * kWordSize;

      for (int i = remainder + threadIdx.x; i < numCodes; i += blockDim.x) {
        smem[i] = codes[i];
      }
    } else {
      // Potential unaligned load
      constexpr int kUnroll = 4;

      int limit = utils::roundDown(numCodes, kUnroll * blockDim.x);

      int i = threadIdx.x;
      for (; i < limit; i += kUnroll * blockDim.x) {
        LookupT vals[kUnroll];

#pragma unroll
        for (int j = 0; j < kUnroll; ++j) {
          vals[j] = codes[i + j * blockDim.x];
        }

#pragma unroll
        for (int j = 0; j < kUnroll; ++j) {
          smem[i + j * blockDim.x] = vals[j];
        }
      }

      for (; i < numCodes; i += blockDim.x) {
        smem[i] = codes[i];
      }
    }
  }
};

} } // namespace
