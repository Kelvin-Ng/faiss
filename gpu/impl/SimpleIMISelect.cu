/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */


#include "SimpleIMISelect.cuh"
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
__global__ void SimpleIMISelect(Tensor<TVec, 3, true, int64_t> input,
                       Tensor<int, 2, true, int64_t> output
					   int S,
					   int T) {
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

  if (lastRowTile) {
    // We are handling the very end of the input matrix rows
    for (int64_t row = 0; row < input.getSize(1) - rowStart; ++row) {
      if (NormLoop) {
        rowMin[0] = 1e10;

        for (int64_t col = threadIdx.x;
             col < T - 2 * S ; col += blockDim.x) {
          TVec val = abs(input[0][rowStart + row][col + S] - input[1][rowStart + row][col + S]);
          //rowNorm[0] = Math<T>::add(rowNorm[0], Math<TVec>::reduceAdd(val));
		  rowId[0] = ( val < rowMin[0] ? col + S : rowId[0]);
		  rowMin[0] = ( val < rowMin[0] ? val : rowMin[0]);
        }
      } else {
        TVec val = abs(input[0][rowStart + row][threadIdx.x + S] - input[1][rowStart + row][threadIdx.x + S]);
        //val = Math<TVec>::mul(val, val);
        //rowNorm[0] = Math<TVec>::reduceAdd(val);
		rowId[0] = ( val < rowMin[0] ? col + S : rowId[0]);
		rowMin[0] = ( val < rowMin[0] ? val : rowMin[0]);
      }

      //rowNorm[0] = warpReduceAllMin(rowNorm[0]);
	  pair<T,int> res = warpReduceAllMin(rowMin[0],rowId[0]);
	  rowMin[0] = res.first;
	  rowId[0] = res.second;
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
           col < T - 2 * S; col += blockDim.x) {

#pragma unroll
        for (int row = 0; row < RowTileSize; ++row) {
          tmp[row] = abs(input[0][rowStart + row][col + S]-input[1][rowStart+row][col + S]);
        }

#pragma unroll
        for (int row = 0; row < RowTileSize; ++row) {
		  rowId[row] = (tmp[row] < rowMin[row] ? col + S : rowId[row]);
		  rowMin[row] = (tmp[row] < rowMin[row] ? tmp[row] : rowMin[row]);
        }
      }
    } else {
      TVec tmp[RowTileSize];

      // A block of threads is the exact size of the vector
#pragma unroll
      for (int row = 0; row < RowTileSize; ++row) {
        tmp[row] = abs(input[0][rowStart + row][threadIdx.x + S] - input[1][rowStart + row][threadIdx.x + S]);
      }

#pragma unroll
      for (int row = 0; row < RowTileSize; ++row) {
        rowId[row] = (tmp[row] < rowMin[row] ? threadIdx.x + S : rowId[row]);
		rowMin[row] = (tmp[row] < rowMin[row] ? tmp[row] : rowMin[row]);
      }
    }

    // Sum up all parts in each warp
#pragma unroll
    for (int row = 0; row < RowTileSize; ++row) {
     // rowNorm[row] = warpReduceAllSum(rowNorm[row]);
	 pair<T, int> res = warpReduceAllMin(rowMin[row], rowId[row]);
	 rowMin[row] = res.first;
	 rowId[row] = res.second;
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
      pair<T, int> res = warpReduceAllMin(rowMin[row], rowId[row]);
	  rowMin[row] = res.first;
	  rowId[row] = res.second;
    }

    // Write out answer
    if (laneId == 0) {
#pragma unroll
      for (int row = 0; row < RowTileSize; ++row) {
        int outCol = rowStart + row;
        if (lastRowTile) {
		  if(outCol < output.getSize(1)) {
		    output[0][outcol] = rowId[row];
			output[1][outcol] = T - rowId[row];
        }
      }
	  else {
	  output[0][outcol] = rowId[row];
	  output[1][outcol] = T - rowId[row];
	  	}
	  }
    }
  }
}

template <typename T, typename TVec, typename int64_t>
void runSimpleIMISelct(Tensor<T, 3, true, int64_t>& input,
               Tensor<int, 2, true, int64_t>& output,
               int S,
			   int T,
               cudaStream_t stream) {
  FAISS_ASSERT(input.getSize(0) == output.getSize(0));

  int64_t maxThreads = (int64_t) getMaxThreadsCurrentDevice();
  constexpr int rowTileSize = 8;

#define RUN_L2(TYPE_T, TYPE_TVEC, INPUT)                                \
  do {                                                                  \
    if (normLoop) {                                                     \
        SimpleIMISelect<TYPE_T, TYPE_TVEC, int64_t, rowTileSize, true>  \
          <<<grid, block, smem, stream>>>(INPUT, output, S, T);         \
    } else {                                                            \
        SimpleIMISelect<TYPE_T, TYPE_TVEC, int64_t, rowTileSize, false> \
          <<<grid, block, smem, stream>>>(INPUT, output, S, T);         \
	  }                                                                 \
  } while (0)                                                           \

  if (input.template canCastResize<TVec>()) {
    // Can load using the vectorized type
    auto inputV = input.template castResize<TVec>();

    auto dim = T - 2 * S;//inputV.getSize(1);
    bool normLoop = dim > maxThreads;
    auto numThreads = min(dim, maxThreads);

    auto grid = dim3(utils::divUp(inputV.getSize(0), rowTileSize));
    auto block = dim3(numThreads);

    auto smem = sizeof(T) * rowTileSize * utils::divUp(numThreads, kWarpSize);

    RUN_L2(T, TVec, inputV);
  } else {
    // Can't load using the vectorized type

    auto dim = T - 2 * S;//input.getSize(1);
    bool normLoop = dim > maxThreads;
    auto numThreads = min(dim, maxThreads);

    auto grid = dim3(utils::divUp(input.getSize(0), rowTileSize));
    auto block = dim3(numThreads);

    auto smem = sizeof(T) * rowTileSize * utils::divUp(numThreads, kWarpSize);

    RUN_L2(T, T, input);
  }

#undef RUN_L2

  CUDA_TEST_ERROR();
}

void runSimpleIMISelect(Tensor<float, 3, true>& input,
               Tensor<int, 2, true>& output,
               int S;
			   int T;
               cudaStream_t stream) {
  if (input.canUseIndexType<int>()) {
    runSimpleIMISelect<float, int, int>(input, output, S, T, stream);
  } else {
    auto inputCast = input.castIndexType<long>();
    auto outputCast = output.castIndexType<long>();
    runSimpleIMISelect<float, int, long>(inputCast, outputCast, S, T, stream);
  }
}

#ifdef FAISS_USE_FLOAT16
void runSimpleIMISelect(Tensor<half, 3, true>& input,
               Tensor<int, 2, true>& output,
               int S;
			   int T;
               cudaStream_t stream) {
  if (input.canUseIndexType<int>()) {
    runSimpleIMISelect<half, int, int>(input, output, S, T, stream);
  } else {
    auto inputCast = input.castIndexType<long>();
    auto outputCast = output.castIndexType<long>();
    runSimpleIMISelect<half, int, long>(inputCast, outputCast, S, T, stream);
  }
}
#endif

} } // namespace
