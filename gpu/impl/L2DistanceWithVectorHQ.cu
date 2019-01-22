/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */


#include "L2DistanceWithVectorHQ.cuh"
#include "../../FaissAssert.h"
#include "../utils/ConversionOperators.cuh"
#include "../utils/DeviceDefs.cuh"
#include "../utils/DeviceUtils.h"
#include "../utils/Float16.cuh"
#include "../utils/MathOperators.cuh"
#include "../utils/PtxUtils.cuh"
#include "../utils/StaticUtils.h"
#include "../utils/Reductions.cuh"

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
          int RowTileSize, bool NormLoop, bool NormSquared, typename InputIndexT>
__global__ void l2DistanceWithVectorHQ(const Tensor<TVec, 3, true, int64_t> input,
                                       const Tensor<InputIndexT, 1, true, int64_t> inputIndices,
                                       const Tensor<TVec, 1, true, int64_t> vec,
                                       Tensor<T, 2, true, int64_t> output) {
  extern __shared__ char smemByte[]; // #warps * RowTileSize elements
  T* smem = (T*) smemByte;

  int64_t numWarps = utils::divUp(blockDim.x, kWarpSize);
  int64_t laneId = getLaneId();
  int64_t warpId = threadIdx.x / kWarpSize;

  bool lastRowTile = (blockIdx.y == (gridDim.y - 1));
  int64_t coarseRow = blockIdx.x;
  int64_t fineRowStart = RowTileSize * blockIdx.y;
  T rowNorm[RowTileSize];

  if (lastRowTile) {
    // We are handling the very end of the input matrix rows
    for (int64_t row = 0; row < input.getSize(1) - fineRowStart; ++row) {
      if (NormLoop) {
        rowNorm[0] = Math<T>::zero();

        for (int64_t col = threadIdx.x;
             col < input.getSize(2); col += blockDim.x) {
          TVec val = Math<TVec>::sub(input[inputIndices[coarseRow]][fineRowStart + row][col], vec[col]);
          val = Math<TVec>::mul(val, val);
          rowNorm[0] = Math<T>::add(rowNorm[0], Math<TVec>::reduceAdd(val));
        }
      } else {
        TVec val = Math<TVec>::sub(input[inputIndices[coarseRow]][fineRowStart + row][threadIdx.x], vec[threadIdx.x]);
        val = Math<TVec>::mul(val, val);
        rowNorm[0] = Math<TVec>::reduceAdd(val);
      }

      rowNorm[0] = warpReduceAllSum(rowNorm[0]);
      if (laneId == 0) {
        smem[row * numWarps + warpId] = rowNorm[0];
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
        rowNorm[row] = Math<T>::zero();
      }

      for (int64_t col = threadIdx.x;
           col < input.getSize(2); col += blockDim.x) {
#pragma unroll
        for (int row = 0; row < RowTileSize; ++row) {
          tmp[row] = Math<TVec>::sub(input[inputIndices[coarseRow]][fineRowStart + row][col] - vec[col]);
        }

#pragma unroll
        for (int row = 0; row < RowTileSize; ++row) {
          tmp[row] = Math<TVec>::mul(tmp[row], tmp[row]);
        }

#pragma unroll
        for (int row = 0; row < RowTileSize; ++row) {
          rowNorm[row] = Math<T>::add(rowNorm[row],
                                      Math<TVec>::reduceAdd(tmp[row]));
        }
      }
    } else {
      TVec tmp[RowTileSize];

      // A block of threads is the exact size of the vector
#pragma unroll
      for (int row = 0; row < RowTileSize; ++row) {
        tmp[row] = Math<TVec>::sub(input[inputIndices[coarseRow]][fineRowStart + row][threadIdx.x], vec[threadIdx.x]);
      }

#pragma unroll
      for (int row = 0; row < RowTileSize; ++row) {
        tmp[row] = Math<TVec>::mul(tmp[row], tmp[row]);
      }

#pragma unroll
      for (int row = 0; row < RowTileSize; ++row) {
        rowNorm[row] = Math<TVec>::reduceAdd(tmp[row]);
      }
    }

    // Sum up all parts in each warp
#pragma unroll
    for (int row = 0; row < RowTileSize; ++row) {
      rowNorm[row] = warpReduceAllSum(rowNorm[row]);
    }

    if (laneId == 0) {
#pragma unroll
      for (int row = 0; row < RowTileSize; ++row) {
        smem[row * numWarps + warpId] = rowNorm[row];
      }
    }
  }

  __syncthreads();

  // Sum across warps
  if (warpId == 0) {
#pragma unroll
    for (int row = 0; row < RowTileSize; ++row) {
      rowNorm[row] = laneId < numWarps ?
                              smem[row * numWarps + laneId] : Math<T>::zero();
    }

#pragma unroll
    for (int row = 0; row < RowTileSize; ++row) {
      rowNorm[row] = warpReduceAllSum(rowNorm[row]);
    }

    // Write out answer
    if (laneId == 0) {
#pragma unroll
      for (int row = 0; row < RowTileSize; ++row) {
        int outCol = fineRowStart + row;

        if (lastRowTile) {
          if (outCol < output.getSize(1)) {
            output[coarseRow][outCol] =
              NormSquared ? rowNorm[row] :
              ConvertTo<T>::to(
                sqrtf(ConvertTo<float>::to(rowNorm[row])));
          }
        } else {
          output[coarseRow][outCol] =
            NormSquared ? rowNorm[row] :
            ConvertTo<T>::to(
              sqrtf(ConvertTo<float>::to(rowNorm[row])));
        }
      }
    }
  }
}

template <typename T, typename TVec, typename int64_t, typename InputIndexT>
__host__ __device__
void runL2DistanceWithVectorHQ(const Tensor<T, 3, true, int64_t>& input, // (coarseIdx, fineIdx, dim) -> val
                               const Tensor<InputIndexT, 1, true, int64_t>& inputIndices, // coarseRank -> coarseIdx
                               const Tensor<T, 1, true, int64_t>& vec,
                               Tensor<T, 2, true, int64_t> output, // (coarseRank, fineIdx) -> val
                               bool normSquared,
                               cudaStream_t stream) {
  FAISS_ASSERT(inputIndices.getSize(0) == output.getSize(0));

#ifndef __CUDA_ARCH__
  int64_t maxThreads = (int64_t) getMaxThreadsCurrentDevice();
#else
  int64_t maxThreads = 256; // FIXME: query this number in kernel?
#endif
  constexpr int rowTileSize = 8;

#define RUN_L2(TYPE_T, TYPE_TVEC, INPUT, INPUT_INDICES, VEC)                                \
  do {                                                                  \
    if (normLoop) {                                                     \
      if (normSquared) {                                                \
        l2DistanceWithVectorHQ<TYPE_T, TYPE_TVEC, int64_t, rowTileSize, true, true>      \
          <<<grid, block, smem, stream>>>(INPUT, INPUT_INDICES, VEC, output);               \
      } else {                                                          \
        l2DistanceWithVectorHQ<TYPE_T, TYPE_TVEC, int64_t, rowTileSize, true, false>     \
          <<<grid, block, smem, stream>>>(INPUT, INPUT_INDICES, VEC, output);               \
      }                                                                 \
    } else {                                                            \
      if (normSquared) {                                                \
        l2DistanceWithVectorHQ<TYPE_T, TYPE_TVEC, int64_t, rowTileSize, false, true>     \
          <<<grid, block, smem, stream>>>(INPUT, INPUT_INDICES, VEC, output);               \
      } else {                                                          \
        l2DistanceWithVectorHQ<TYPE_T, TYPE_TVEC, int64_t, rowTileSize, false, false>    \
          <<<grid, block, smem, stream>>>(INPUT, INPUT_INDICES, VEC, output);               \
      }                                                                 \
    }                                                                   \
  } while (0)

  if (input.template canCastResize<TVec>()) {
    // Can load using the vectorized type
    auto inputV = input.template castResize<TVec>();
    auto vecV = vec.template castResize<TVec>();

    auto dim = inputV.getSize(2);
    bool normLoop = dim > maxThreads;
    auto numThreads = min(dim, maxThreads);

    auto grid = dim3(inputIndices.getSize(0), utils::divUp(inputV.getSize(1), rowTileSize)); // TODO: inefficient when inputV.size(1) < rowTileSize
    auto block = dim3(numThreads);

    auto smem = sizeof(T) * rowTileSize * utils::divUp(numThreads, kWarpSize);

    RUN_L2(T, TVec, inputV, inputIndices, vecV);
  } else {
    // Can't load using the vectorized type

    auto dim = input.getSize(2);
    bool normLoop = dim > maxThreads;
    auto numThreads = min(dim, maxThreads);

    auto grid = dim3(inputIndices.getSize(0), utils::divUp(input.getSize(1), rowTileSize));
    auto block = dim3(numThreads);

    auto smem = sizeof(T) * rowTileSize * utils::divUp(numThreads, kWarpSize);

    RUN_L2(T, T, input, inputIndices, vec);
  }

#undef RUN_L2

  CUDA_TEST_ERROR();
}

template <typename InputIndexT>
__host__ __device__
void runL2DistanceWithVectorHQ(const Tensor<float, 3, true>& input, // (coarseIdx, fineIdx, dim) -> val
                               const Tensor<InputIndexT, 1, true>& inputIndices, // coarseRank -> coarseIdx
                               const Tensor<float, 1, true>& vec,
                               Tensor<float, 2, true> output, // (coarseRank, fineIdx) -> val
                               bool normSquared,
                               cudaStream_t stream) {
#ifdef __CUDA_ARCH__ // we can call canUseIndexType only on CPU
  runL2DistanceWithVectorHQ<float, float4>(input, inputIndices, vec, output, normSquared, stream);
#else
  if (input.canUseIndexType<int>()) {
    runL2DistanceWithVectorHQ<float, float4, int>(input, inputIndices, vec, output, normSquared, stream);
  } else {
    auto inputCast = input.castIndexType<long>();
    auto inputIndicesCast = inputIndices.template castIndexType<long>();
    auto vecCast = vec.castIndexType<long>();
    auto outputCast = output.castIndexType<long>();
    runL2DistanceWithVectorHQ<float, float4, long>(inputCast, inputIndicesCast, vecCast, outputCast, normSquared, stream);
  }
#endif
}

#ifdef FAISS_USE_FLOAT16
template <typename InputIndexT>
__host__ __device__
void runL2DistanceWithVectorHQ(const Tensor<half, 3, true>& input, // (coarseIdx, fineIdx, dim) -> val
                               const Tensor<InputIndexT, 1, true>& inputIndices, // coarseRank -> coarseIdx
                               const Tensor<half, 1, true>& vec,
                               Tensor<half, 2, true> output, // (coarseRank, fineIdx) -> val
                               bool normSquared,
                               cudaStream_t stream) {
#ifdef __CUDA_ARCH__
  runL2DistanceWithVectorHQ<half, half2>(input, inputIndices, vec, output, normSquared, stream);
#else
  if (input.canUseIndexType<int>()) {
    runL2DistanceWithVectorHQ<half, half2, int>(input, inputIndices, vec, output, normSquared, stream);
  } else {
    auto inputCast = input.castIndexType<long>();
    auto inputIndicesCast = inputIndices.template castIndexType<long>();
    auto vecCast = vec.castIndexType<long>();
    auto outputCast = output.castIndexType<long>();
    runL2DistanceWithVectorHQ<half, half2, long>(inputCast, inputIndicesCast, vecCast, outputCast, normSquared, stream);
  }
#endif
}
#endif

} } // namespace
