/**
 * Copyright (c) 2019-present, Husky Data Lab.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */


#include "HQThirdStageL2Distances.cuh"
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

__device__ __forceinline__ void
printCodewords2(int a, int b, int c, int d, float4 val) {
    printf("codewords2[%d][%d][%d][%d]: (%f %f %f %f)\n", a, b, c, d, val.x, val.y, val.z, val.w);
}

__device__ __forceinline__ void
printCodewords2(int a, int b, int c, int d, float val) {
    printf("codewords2[%d][%d][%d][%d]: %f\n", a, b, c, d, val);
}

__device__ __forceinline__ void
printCodewords1(int a, int b, int c, int d, float4 val) {
    printf("codewords1[%d][%d][%d][%d]: (%f %f %f %f)\n", a, b, c, d, val.x, val.y, val.z, val.w);
}

__device__ __forceinline__ void
printCodewords1(int a, int b, int c, int d, float val) {
    printf("codewords1[%d][%d][%d][%d]: %f\n", a, b, c, d, val);
}

template <int numCodes2, typename T, typename TVec>
__device__ __forceinline__ T
HQThirdStageL2DistancesOneCol(const Tensor<TVec, 2, true>& queries,
                              const void** listCodes1,
                              const void** listCodes2,
                              const Tensor<TVec, 4, true>& codewords1,
                              const Tensor<TVec, 4, true>& codewords2,
                              int imiId[2],
                              int fineId,
                              int listId,
                              int col) {
  int halfDim = queries.getSize(1) / 2;
  bool isSecondHalf = (col >= halfDim);

  const unsigned char* myListCodes1 = (const unsigned char*)listCodes1[listId];
  const unsigned char* myListCodes2 = (const unsigned char*)listCodes2[listId];

  unsigned char code1 = myListCodes1[fineId * 2 + isSecondHalf];
  unsigned char codes2[numCodes2];

#pragma unroll
  for (int i = 0; i < numCodes2; ++i) {
    codes2[i] = myListCodes2[fineId * numCodes2 + i];
  }

  TVec val = queries[blockIdx.y][col];

  TVec my_codewords[1 + numCodes2];

  my_codewords[0] = codewords1[isSecondHalf][imiId[isSecondHalf]][code1][col - isSecondHalf * halfDim];
  if (imiId[0] == 3353 && imiId[1] == 1055) {
      printCodewords1(isSecondHalf, imiId[isSecondHalf], code1, col - isSecondHalf * halfDim, my_codewords[0]);
  }
#pragma unroll
  for (int i = 0; i < numCodes2 / 2; ++i) {
    my_codewords[1 + i * 2 + 0] = codewords2[i][0                    ][codes2[i * 2 + 0]][col];
    my_codewords[1 + i * 2 + 1] = codewords2[i][1 + codes2[i * 2 + 0]][codes2[i * 2 + 1]][col];
    //printCodewords2(i, 0, codes2[i * 2 + 0], col, my_codewords[1 + i * 2 + 0]);
    //printCodewords2(i, 1 + codes2[i * 2 + 0], codes2[i * 2 + 1], col, my_codewords[1 + i * 2 + 1]);
  }

#pragma unroll
  for (int i = 0; i < 1 + numCodes2; ++i) {
    val = Math<TVec>::sub(val, my_codewords[i]);
  }
  val = Math<TVec>::mul(val, val);

  return Math<TVec>::reduceAdd(val);
}

// TODO:
// 1. When looping along rows, use grid-stride loops
// 2. Load the codes for all rows before computation to allow more memory transactions in flight
template <typename T, typename TVec,
          int RowTileSize, bool NormLoop, bool NormSquared, int numCodes2>
__global__ void HQThirdStageL2Distances(Tensor<TVec, 2, true> queries,
                                        Tensor<int, 3, true> indices,
                                        const void** listCodes1,
                                        const void** listCodes2,
                                        Tensor<TVec, 4, true> codewords1,
                                        Tensor<TVec, 4, true> codewords2,
                                        int imiSize,
                                        Tensor<T, 2, true> distances) {
  extern __shared__ char smemByte[]; // #warps * RowTileSize elements
  T* smem = (T*) smemByte;

  int numWarps = utils::divUp(blockDim.x, kWarpSize);
  int laneId = getLaneId();
  int warpId = threadIdx.x / kWarpSize;

  int qid = blockIdx.y;

  bool lastRowTile = (blockIdx.x == (gridDim.x - 1));
  int rowStart = RowTileSize * blockIdx.x;
  T rowNorm[RowTileSize];

  if (lastRowTile) {
    // We are handling the very end of the input matrix rows
    for (int row = 0; row < indices.getSize(2) - rowStart; ++row) {
      //printf("indices[0][%d][%d]: %d\n", qid, rowStart + row, (int)indices[0][qid][rowStart + row]);
      int imiId[2] = {indices[0][qid][rowStart + row], indices[1][qid][rowStart + row]};
      int fineId = indices[2][qid][rowStart + row];
      int listId = imiId[0] * imiSize + imiId[1];

      if (NormLoop) {
        rowNorm[0] = 0;

        for (int col = threadIdx.x;
             col < queries.getSize(1); col += blockDim.x) {
          T val = HQThirdStageL2DistancesOneCol<numCodes2, T>(queries, listCodes1, listCodes2, codewords1, codewords2, imiId, fineId, listId, col);
          rowNorm[0] += val;
        }
      } else {
        T val = HQThirdStageL2DistancesOneCol<numCodes2, T>(queries, listCodes1, listCodes2, codewords1, codewords2, imiId, fineId, listId, threadIdx.x);
        rowNorm[0] = val;
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

#pragma unroll
      for (int row = 0; row < RowTileSize; ++row) {
        //printf("indices[0][%d][%d]: %d\n", qid, rowStart + row, (int)indices[0][qid][rowStart + row]);
        int imiId[2] = {indices[0][qid][rowStart + row], indices[1][qid][rowStart + row]};
        int fineId = indices[2][qid][rowStart + row];
        int listId = imiId[0] * imiSize + imiId[1];

        rowNorm[row] = 0;

        for (int col = threadIdx.x;
             col < queries.getSize(1); col += blockDim.x) {
          T val = HQThirdStageL2DistancesOneCol<numCodes2, T>(queries, listCodes1, listCodes2, codewords1, codewords2, imiId, fineId, listId, col);
          rowNorm[row] += val;
        }
      }
    } else {
      // A block of threads is the exact size of the vector

#pragma unroll
      for (int row = 0; row < RowTileSize; ++row) {
        //printf("indices[0][%d][%d]: %d\n", qid, rowStart + row, (int)indices[0][qid][rowStart + row]);
        int imiId[2] = {indices[0][qid][rowStart + row], indices[1][qid][rowStart + row]};
        int fineId = indices[2][qid][rowStart + row];
        int listId = imiId[0] * imiSize + imiId[1];

        T val = HQThirdStageL2DistancesOneCol<numCodes2, T>(queries, listCodes1, listCodes2, codewords1, codewords2, imiId, fineId, listId, threadIdx.x);
        rowNorm[row] = val;
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
        int outCol = rowStart + row;

        if (lastRowTile) {
          if (outCol < distances.getSize(1)) {
            distances[qid][outCol] =
              NormSquared ? rowNorm[row] :
              ConvertTo<T>::to(
                sqrtf(ConvertTo<float>::to(rowNorm[row])));
          }
        } else {
          distances[qid][outCol] =
            NormSquared ? rowNorm[row] :
            ConvertTo<T>::to(
              sqrtf(ConvertTo<float>::to(rowNorm[row])));
        }
      }
    }
  }
}

void runHQThirdStageL2Distances(const Tensor<float, 2, true>& queries,
                                const Tensor<int, 3, true>& indices,
                                const void** listCodes1,
                                const void** listCodes2,
                                const Tensor<float, 4, true>& codewords1,
                                const Tensor<float, 4, true>& codewords2,
                                int imiSize,
                                int numCodes2,
                                Tensor<float, 2, true>& distances,
                                bool normSquared,
                                cudaStream_t stream) {
  int64_t maxThreads = (int64_t) getMaxThreadsCurrentDevice();
  constexpr int rowTileSize = 8;

#define RUN_L2(TYPE_T, TYPE_TVEC, QUERIES, CODEWORDS1, CODEWORDS2, NUMCODES2)                                \
  do {                                                                  \
    if (normLoop) {                                                     \
      if (normSquared) {                                                \
        HQThirdStageL2Distances<TYPE_T, TYPE_TVEC, rowTileSize, true, true, NUMCODES2>      \
          <<<grid, block, smem, stream>>>(QUERIES, indices, listCodes1, listCodes2, CODEWORDS1, CODEWORDS2, imiSize, distances);               \
      } else {                                                          \
        HQThirdStageL2Distances<TYPE_T, TYPE_TVEC, rowTileSize, true, false, NUMCODES2>     \
          <<<grid, block, smem, stream>>>(QUERIES, indices, listCodes1, listCodes2, CODEWORDS1, CODEWORDS2, imiSize, distances);               \
      }                                                                 \
    } else {                                                            \
      if (normSquared) {                                                \
        HQThirdStageL2Distances<TYPE_T, TYPE_TVEC, rowTileSize, false, true, NUMCODES2>     \
          <<<grid, block, smem, stream>>>(QUERIES, indices, listCodes1, listCodes2, CODEWORDS1, CODEWORDS2, imiSize, distances);               \
      } else {                                                          \
        HQThirdStageL2Distances<TYPE_T, TYPE_TVEC, rowTileSize, false, false, NUMCODES2>    \
          <<<grid, block, smem, stream>>>(QUERIES, indices, listCodes1, listCodes2, CODEWORDS1, CODEWORDS2, imiSize, distances);               \
      }                                                                 \
    }                                                                   \
  } while (0)

  if (queries.canCastResize<float4>() && codewords1.canCastResize<float4>() && codewords2.canCastResize<float4>()) {
    // Can load using the vectorized type
    auto queriesV = queries.castResize<float4>();
    auto codewords1V = codewords1.castResize<float4>();
    auto codewords2V = codewords2.castResize<float4>();

    auto dim = queriesV.getSize(1);
    bool normLoop = dim > maxThreads;
    auto numThreads = min(dim, (int)maxThreads);

    auto grid = dim3(utils::divUp(indices.getSize(2), rowTileSize), queriesV.getSize(0));
    auto block = dim3(numThreads);

    auto smem = sizeof(float) * rowTileSize * utils::divUp(numThreads, kWarpSize);

    switch (numCodes2) {
      case 2: {
        RUN_L2(float, float4, queriesV, codewords1V, codewords2V, 2);
        break;
      }
      case 4: {
        RUN_L2(float, float4, queriesV, codewords1V, codewords2V, 4);
        break;
      }
      case 6: {
        RUN_L2(float, float4, queriesV, codewords1V, codewords2V, 6);
        break;
      }
      case 8: {
        RUN_L2(float, float4, queriesV, codewords1V, codewords2V, 8);
        break;
      }
      default: {
        FAISS_ASSERT_MSG(false, "This number of code 2 is not supported");
      }
    }
  } else {
    // Can't load using the vectorized type

    auto dim = queries.getSize(1);
    bool normLoop = dim > maxThreads;
    auto numThreads = min((int64_t)dim, maxThreads);

    auto grid = dim3(utils::divUp(indices.getSize(2), rowTileSize), queries.getSize(0));
    auto block = dim3(numThreads);

    auto smem = sizeof(float) * rowTileSize * utils::divUp(numThreads, kWarpSize);

    switch (numCodes2) {
      case 2: {
        RUN_L2(float, float, queries, codewords1, codewords2, 2);
        break;
      }
      case 4: {
        RUN_L2(float, float, queries, codewords1, codewords2, 4);
        break;
      }
      case 6: {
        RUN_L2(float, float, queries, codewords1, codewords2, 6);
        break;
      }
      case 8: {
        RUN_L2(float, float, queries, codewords1, codewords2, 8);
        break;
      }
      default: {
        FAISS_ASSERT_MSG(false, "This number of code 2 is not supported");
      }
    }
  }

#undef RUN_L2

  CUDA_TEST_ERROR();
}

} } // namespace
