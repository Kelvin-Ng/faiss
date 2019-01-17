/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 * 
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */


#include "SimpleIMICut.cuh"
#include "../../FaissAssert.h"
#include "../utils/ConversionOperators.cuh"
#include "../utils/DeviceDefs.cuh"
#include "../utils/DeviceUtils.h"
#include "../utils/Float16.cuh"
#include "../utils/MathOperators.cuh"
#include "../utils/PtxUtils.cuh"
#include "../utils/StaticUtils.h"
#include "../utils/Reductions.cuh"
#include "math.h"
#include <thrust/tuple.h>

namespace faiss { namespace gpu {

// Input: (batch x dim), # repeats
// Output: (# repeats, norm of batch vector)
// Done under the presumption that the dimension size is not too large
// (<10k or so), since there wouldn't be enough parallelism applying a
// single block to the problem. Also that each vector is large enough
// (>64), since a single block works on multiple rows' norms at the
// same time.
// T: the type we are doing the math in (e.g., float, half)
// TVec: the potentially vectorized type we are loading in (e.g.,
// float4, half2)
template <typename T, typename TVec, typename int64_t,
          int RowTileSize, bool NormLoop>
__global__ void SimpleIMICut(Tensor<TVec, 3, true, int64_t> input,
                             Tensor<int, 2, true, int64_t> output
                             int squareLen,
                             int totalLen) {
  extern __shared__ char smemByte1[]; // #warps * RowTileSize elements
  extern __shared__ char smemByte2[];
  T* smemMin = (T*) smemByte1;
  T* smemId  = (T*) smemByte2;
  int64_t numWarps = utils::divUp(blockDim.x, kWarpSize);
  int64_t laneId = getLaneId();
  int64_t warpId = threadIdx.x / kWarpSize;

  bool lastRowTile = (blockIdx.x == (gridDim.x - 1));
  int64_t rowStart = RowTileSize * blockIdx.x;
  T rowMin[RowTileSize];
  int rowId[RowTileSize];
  T argMin;
  int argId;

  constexpr int stp = sizeof(TVec) / sizeof(T);
  int totalCol = (totalLen - 2 * squareLen + 3) / stp;
  int upperCol = totalLen / stp;
  squareLen = (squareLen - 1) / stp;
  if (lastRowTile) {
    // We are handling the very end of the input matrix rows
    for (int64_t row = 0; row < input.getSize(1) - rowStart; ++row) {
      rowMin[0] = 1e10;
      if (NormLoop) {
        for (int64_t col = threadIdx.x;
             col < totalCol; col += blockDim.x) {
          TVec val = Math<TVec>::abs(Math<TVec>::revSub(input[0][rowStart + row][col + squareLen],  input[1][rowStart + row][upperCol - (col + squareLen)]));
          argId = Math<TVec>::argMin(val);
          argMin = Math<TVec>::getVal(val,argId);
          rowId[0] = ( argMin < rowMin[0] ? (col + squareLen) * stp +argId : rowId[0]);
          rowMin[0] = min(argMin,rowMin[0]);
        }
      } else {
        TVec val = Math<TVec>::abs(Math<TVec>::revSub(input[0][rowStart + row][threadIdx.x + squareLen], input[1][rowStart + row][upperCol - (threadIdx.x + squareLen)]));
        argId = Math<TVec>::argMin(val);
        argMin = Math<TVec>::getVal(val,argId);
        rowId[0] = ( argMin < rowMin[0] ? (threadIdx.x + squareLen) * stp + argId : rowId[0]);
        rowMin[0] = min(argMin,rowMin[0]);
      }

      thrust::tie(rowMin[0], rowId[0]) = warpReduceAllMin(rowMin[0], rowId[0]);
      if (laneId == 0) {
        smemMin[row * numWarps + warpId] = rowMin[0];
        smemId[row * numWarps + warpId] = rowId[0];
      }
    }
  } else {
    // We are guaranteed that all RowTileSize rows are available in
    // [rowStart, rowStart + RowTileSize)

    if (NormLoop) {
      // A single block of threads is not big enough to span each
      // vector
      TVec tmp[RowTileSize];

#pragma unroll
      for (int row = 0; row < RowTileSize; ++row) {
        rowMin[row] = 1e10;
      }

      for (int64_t col = threadIdx.x;
           col < totalCol; col += blockDim.x) {

#pragma unroll
        for (int row = 0; row < RowTileSize; ++row) {
          tmp[row] = Math<TVec>::abs(Math<TVec>::revSub(input[0][rowStart + row][col + squareLen], input[1][rowStart + row][upperCol -(col + squareLen)]));
        }

#pragma unroll
        for (int row = 0; row < RowTileSize; ++row) {
          argId = Math<TVec>::argMin(tmp[row]);
          argMin = Math<TVec>::getVal(tmp[row], argId);
          rowId[row] = ( argMin < rowMin[row] ? (col + squareLen) * stp +argId : rowId[row]);
          rowMin[row] = min(argMin, rowMin[row]);
        }
      }
    } else {
      TVec tmp[RowTileSize];

      // A block of threads is the exact size of the vector
#pragma unroll
      for (int row = 0; row < RowTileSize; ++row) {
        tmp[row] = Math<TVec>::abs(Math<TVec>::revSub(input[0][rowStart + row][threadIdx.x + squareLen], input[1][rowStart + row][upperCol - (threadIdx.x + squareLen)]));
      }

#pragma unroll
     for (int row = 0; row < RowTileSize; ++row) {
       argId = Math<TVec>::argMin(tmp[row]);
       argMin = Math<TVec>::getVal(tmp[row], argId);
       rowId[row] = (argMin < rowMin[row] ? (threadIdx.x + squareLen) * stp + argId : rowId[row]);
       rowMin[row] = min(argMin, rowMin[row]);
     }
   }

    // Sum up all parts in each warp
#pragma unroll
    for (int row = 0; row < RowTileSize; ++row) {
      thrust::tie(rowMin[row], rowId[row]) = warpReduceAllMin(rowMin[row], rowId[row]);
    }

    if (laneId == 0) {
#pragma unroll
      for (int row = 0; row < RowTileSize; ++row) {
        smemMin[row * numWarps + warpId] = rowMin[row];
        smemId[row * numWarps + warpId] = rowId[row];
      }
    }
  }

  __syncthreads();

  // Sum across warps
  if (warpId == 0) {
#pragma unroll
    for (int row = 0; row < RowTileSize; ++row) {
      rowMin[row] = laneId < numWarps ? smemMin[row * numWarps + laneId] : 1e10;
      rowId[row] = laneId < numWarps ? smemId[row * numWarps + laneId] : Math<T>::zero();
    }

#pragma unroll
    for (int row = 0; row < RowTileSize; ++row) {
      thrust::tie(rowMin[row], rowId[row]) = warpReduceAllMin(rowMin[row], rowId[row]);
    }

    // Write out answer
    if (laneId == 0) {
#pragma unroll
      for (int row = 0; row < RowTileSize; ++row) {
        int outCol = rowStart + row;
        if (lastRowTile) {
          if(outCol < output.getSize(1)) {
            output[0][outcol] = rowId[row];
            output[1][outcol] = totalLen - rowId[row];
          }
        } else {
	      output[0][outcol] = rowId[row];
	      output[1][outcol] = totalLen - rowId[row];
	    } 
      }
    }
  }
}

template <typename T, typename TVec, typename int64_t>
void runSimpleIMICut(Tensor<T, 3, true, int64_t>& input,
                     Tensor<int, 2, true, int64_t>& output,
                     int squareLen,
                     int totalLen,
                     cudaStream_t stream) {
  FAISS_ASSERT(input.getSize(0) == output.getSize(0));

  int64_t maxThreads = (int64_t) getMaxThreadsCurrentDevice();
  constexpr int rowTileSize = 8;

#define RUN_L2(TYPE_T, TYPE_TVEC, INPUT)                                            \
  do {                                                                              \
    if (normLoop) {                                                                 \
      SimpleIMICut<TYPE_T, TYPE_TVEC, int64_t, rowTileSize, true>                   \
         <<<grid, block, smem, stream>>>(INPUT, output, squareLen, totalLen);       \
    } else {                                                                        \
      SimpleIMICut<TYPE_T, TYPE_TVEC, int64_t, rowTileSize, false>                  \
        <<<grid, block, smem, stream>>>(INPUT, output, squareLen, totalLen);        \
    }                                                                               \
  }while (0)                                                                        
  //Make Sure that the considered segment [squareLen - 1,totalLen - squareLen+1] located in some complete pieces of float4,
  //which requested that (squareLen - 1) % 4 == 0 and (totalLen - 2 * (squareLen - 1) + 1) % 4 == 0 
  //equals to squareLen % 4 == 1 and totalLen % 4 == 3
  if ( (squareLen % 4 == 1) && (totalLen % 4 == 3) ){
    // Can load using the vectorized type
    auto inputV = input.template castResize<TVec>();

    auto dim = (totalLen - 2 * squareLen + 3) / 4;
    bool normLoop = dim > maxThreads;
    auto numThreads = min(dim, maxThreads);

    auto grid = dim3(utils::divUp(inputV.getSize(1), rowTileSize));
    auto block = dim3(numThreads);

    auto smem = sizeof(T) * rowTileSize * utils::divUp(numThreads, kWarpSize);

    RUN_L2(T, TVec, inputV);
  } else {
    // Can't load using the vectorized type

    auto dim = totalLen - 2 * squareLen + 3;
    bool normLoop = dim > maxThreads;
    auto numThreads = min(dim, maxThreads);

    auto grid = dim3(utils::divUp(input.getSize(1), rowTileSize));
    auto block = dim3(numThreads);

    auto smem = sizeof(T) * rowTileSize * utils::divUp(numThreads, kWarpSize);

    RUN_L2(T, T, input);
  }
#undef RUN_L2

  CUDA_TEST_ERROR();
}

void runSimpleIMICut(Tensor<float, 3, true>& input,
                     Tensor<int, 2, true>& output,
                     int squareLen;
                     int totalLen;
                     cudaStream_t stream) {
  input = input.narrow(2, 0, totalLen);
  if (input.canUseIndexType<int>()) {
    runSimpleIMICut<float, int, int>(input, output, squareLen, totalLen, stream);
  } else {
    auto inputCast = input.castIndexType<long>();
    auto outputCast = output.castIndexType<long>();
    runSimpleIMICut<float, int, long>(inputCast, outputCast, squareLen, totalLen, stream);
  }
}

#ifdef FAISS_USE_FLOAT16
void runSimpleIMICut(Tensor<half, 3, true>& input,
                     Tensor<int, 2, true>& output,
                     int squareLen;
                     int totalLen;
                     cudaStream_t stream) {
  input = input.narrow(2, 0, totalLen);
  if (input.canUseIndexType<int>()) {
    runSimpleIMICut<half, int, int>(input, output, squareLen, totalLen, stream);
  } else {
    auto inputCast = input.castIndexType<long>();
    auto outputCast = output.castIndexType<long>();
    runSimpleIMICut<half, int, long>(inputCast, outputCast, squareLen, totalLen, stream);
  }
}
#endif

} } // namespace
