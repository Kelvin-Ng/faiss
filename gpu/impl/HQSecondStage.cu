/**
 * Copyright (c) 2019-present, Husky Data Lab.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "HQSecondStage.cuh"

#include "../GpuResources.h"
#include "PQCodeDistances.cuh"
#include "PQCodeLoad.cuh"
#include "../utils/ConversionOperators.cuh"
#include "../utils/DeviceTensor.cuh"
#include "../utils/DeviceUtils.h"
#include "../utils/Float16.cuh"
#include "../utils/LoadStoreOperators.cuh"
#include "../utils/StaticUtils.h"
#include "../utils/ThrustAllocator.cuh"
#include "IVFUtils.cuh"
#include "LoadCodeDistances.cuh"

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_scan.h>

namespace faiss { namespace gpu {

/*template <typename ListIdT = unsigned long long>
__global__ void HQCalcListIds(const Tensor<int, 3, true> imiIndices, const Tensor<int, 2, true> imiUpperBounds, int nprobeSquareLen, int imiSize, Tensor<ListIdT, 2, true> listIds) {
    int qid = blockIdx.x;
    int overallRank = threadIdx.x;

    int imiUpperBoundCol = imiUpperBounds[1][qid];

    int upperBlockSize = imiUpperBoundCol * nprobeSquareLen;

    int coarseRank0, coarseRank1;
    if (overallRank > upperBlockSize) {
        coarseRank0 = nprobeSquareLen + (overallRank - upperBlockSize) / nprobeSquareLen;
        coarseRank1 = (overallRank - upperBlockSize) % nprobeSquareLen;
    } else {
        coarseRank0 = overallRank / imiUpperBoundCol;
        coarseRank1 = overallRank % imiUpperBoundCol;
    }

    listIds[qid][overallRank] = (ListIdT)imiIndices[0][qid][coarseRank0] * imiSize + (ListIdT)imiIndices[1][qid][coarseRank1];
}*/

template <typename ListIdT = unsigned long long>
void runHQCalcListIds(const Tensor<int, 3, true>& deviceIMIIndices, const Tensor<int, 2, true>& deviceIMIUpperBounds, int numQueries, int numListsPerQuery, int nprobeSquareLen, int imiSize, Tensor<ListIdT, 2, true>& deviceListIds, cudaStream_t stream) {
    thrust::counting_iterator<int> first(0);
    thrust::counting_iterator<int> last = first + numQueries * numListsPerQuery;

    // TODO: should I make a class instead of a lambda? I can control how to store the captured variables if I use a class.
    auto pos2ListId = [=] __device__ (int pos) {
        int qid = pos / numListsPerQuery;
        int overallRank = pos % numListsPerQuery;

        int imiUpperBoundCol = deviceIMIUpperBounds[1][qid];

        int upperBlockSize = imiUpperBoundCol * nprobeSquareLen;

        int coarseRank0, coarseRank1;
        if (overallRank > upperBlockSize) {
            coarseRank0 = nprobeSquareLen + (overallRank - upperBlockSize) / nprobeSquareLen;
            coarseRank1 = (overallRank - upperBlockSize) % nprobeSquareLen;
        } else {
            coarseRank0 = overallRank / imiUpperBoundCol;
            coarseRank1 = overallRank % imiUpperBoundCol;
        }

        return (ListIdT)deviceIMIIndices[0][qid][coarseRank0] * imiSize + (ListIdT)deviceIMIIndices[1][qid][coarseRank1];
    };

    thrust::transform(thrust::cuda::par.on(stream), first, last, deviceListIds.data(), pos2ListId);

    //HQCalcListIds<<<numQueries, numListsPerQuery, 0, stream>>>(deviceIMIIndices, deviceIMIUpperBounds, nprobeSquareLen, imiSize, deviceListIds);
}

template <typename ListIdT = unsigned long long>
void runHQCalcListOffsets(const int* deviceListLengths, const Tensor<ListIdT, 2, true>& deviceListIds, int* devicePrefixSumOffsets, GpuResources* resources, cudaStream_t stream) {
    constexpr int kThrustMemSize = 16384; // TODO: set a reasonable size

    auto& mem = resources->getMemoryManagerCurrentDevice();

    DeviceTensor<char, 1, true> thrustMem(
      mem, {kThrustMemSize}, stream);

    GpuResourcesThrustAllocator thrustAlloc(thrustMem.data(),
                                            thrustMem.getSizeInBytes());

    // TODO: should I make a class instead of a lambda? I can control how to store the captured variables if I use a class.
    auto listId2ListLength = [=] __device__ (ListIdT listId) {
        return deviceListLengths[listId];
    };

    thrust::transform_inclusive_scan(thrust::cuda::par(thrustAlloc).on(stream), deviceListIds.data(), deviceListIds.end(), devicePrefixSumOffsets, listId2ListLength, thrust::plus<int>());
}

template <typename LookupVecT, typename ListIdT, typename LookupT>
__global__ void
HQSecondStageDistances(// (qid, overallRank) -> listId
                       const Tensor<ListIdT, 2, true> listIds,
                       // (imiId, qid) -> upper_bound
                       const Tensor<int, 2, true> imiUpperBounds,
                       // (imiId, qid, coarseRank, fineIdx) -> val
                       const Tensor<LookupT, 4, true> distanceTable,
                       // (listId, i) -> item
                       // Should contains only the codes related to this function. Other codes (those for the third stage) should be stored separately
                       const void** listCodes,
                       // listId -> len
                       const int* listLengths,
                       // (qid, overallRank) -> offset
                       const Tensor<int, 2, true> prefixSumOffsets,
                       int nprobeSquareLen,
                       int imiSize,
                       // offset -> distance
                       Tensor<float, 1, true> distances) {
  constexpr int NumSubQuantizers = 2; // TODO: extend to possibly more sub quantizers

  // Where the pq code -> residual distance is stored
  extern __shared__ char smemCodeDistances[];
  LookupT* codeDist0 = (LookupT*) smemCodeDistances;
  LookupT* codeDist1 = (LookupT*) smemCodeDistances + distanceTable.getSize(3);

  // Each block handles a single bucket
  int qid = blockIdx.y;
  int overallRank = blockIdx.x;

  int imiUpperBoundCol = imiUpperBounds[1][qid];

  int upperBlockSize = imiUpperBoundCol * nprobeSquareLen;
  int coarseRank0, coarseRank1;
  if (overallRank > upperBlockSize) {
    coarseRank0 = nprobeSquareLen + (overallRank - upperBlockSize) / nprobeSquareLen;
    coarseRank1 = (overallRank - upperBlockSize) % nprobeSquareLen;
  } else {
    coarseRank0 = overallRank / imiUpperBoundCol;
    coarseRank1 = overallRank % imiUpperBoundCol;
  }

  // This is where we start writing out data
  // We ensure that before the array (at offset -1), there is a 0 value
  int outBase = *(prefixSumOffsets[qid][overallRank].data() - 1);
  float* distanceOut = distances[outBase].data();

  ListIdT listId = listIds[qid][overallRank];
  // Safety guard in case NaNs in input cause no list ID to be generated
  if (listId == (ListIdT)-1) {
    return;
  }

  unsigned char* codeList = (unsigned char*) listCodes[listId];
  int limit = listLengths[listId];

  constexpr int kNumCode32 = NumSubQuantizers <= 4 ? 1 :
    (NumSubQuantizers / 4);
  unsigned int code32[kNumCode32];
  unsigned int nextCode32[kNumCode32];

  // We double-buffer the code loading, which improves memory utilization
  if (threadIdx.x < limit) {
    LoadCode32<NumSubQuantizers>::load(code32, codeList, threadIdx.x);
  }

  LoadCodeDistances<LookupT, LookupVecT>::load(
    codeDist0,
    distanceTable[0][qid][coarseRank0].data(),
    distanceTable.getSize(3));
  LoadCodeDistances<LookupT, LookupVecT>::load(
    codeDist1,
    distanceTable[1][qid][coarseRank1].data(),
    distanceTable.getSize(3));

  // Prevent WAR dependencies
  __syncthreads();

  // Each thread handles one code element in the list, with a
  // block-wide stride
  for (int codeIndex = threadIdx.x;
       codeIndex < limit;
       codeIndex += blockDim.x) {
    // Prefetch next codes
    if (codeIndex + blockDim.x < limit) {
      LoadCode32<NumSubQuantizers>::load(
        nextCode32, codeList, codeIndex + blockDim.x);
    }

    float dist = 0.0f;

#pragma unroll
    for (int word = 0; word < kNumCode32; ++word) {
      auto code = getByte(code32[word], 0, 8);
      dist += ConvertTo<float>::to(codeDist0[code]);

      code = getByte(code32[word], 8, 8);
      dist += ConvertTo<float>::to(codeDist1[code]);
    }

    // Write out intermediate distance result
    // We do not maintain indices here, in order to reduce global
    // memory traffic. Those are recovered in the final selection step.
    distanceOut[codeIndex] = dist;

    // Rotate buffers
#pragma unroll
    for (int word = 0; word < kNumCode32; ++word) {
      code32[word] = nextCode32[word];
    }
  }
}

template <typename ListIdT, typename LookupT>
void runHQSecondStageDistances(// (qid, overallRank) -> listId
                               const Tensor<ListIdT, 2, true>& deviceListIds,
                               // (imiId, qid) -> upper_bound
                               const Tensor<int, 2, true>& deviceIMIUpperBounds,
                               // (imiId, qid, coarseRank, fineIdx) -> val
                               const Tensor<LookupT, 4, true>& deviceDistanceTable,
                               // (listId, i) -> item
                               // Should contains only the codes related to this function. Other codes (those for the third stage) should be stored separately
                               const void** deviceListCodes,
                               // listId -> len
                               const int* deviceListLengths,
                               // (qid, overallRank) -> offset
                               const Tensor<int, 2, true>& devicePrefixSumOffsets,
                               int numQueries,
                               int nprobeSquareLen,
                               int imiSize,
                               // offset -> distance
                               Tensor<float, 1, true>& deviceDistances,
                               cudaStream_t stream) {
  auto kThreadsPerBlock = 64;

  auto grid = dim3(devicePrefixSumOffsets.getSize(1), numQueries);
  auto block = dim3(kThreadsPerBlock);

  // pq centroid distances
  auto smem = sizeof(LookupT);
  smem *= deviceDistanceTable.getSize(3) * deviceDistanceTable.getSize(0);
  FAISS_ASSERT(smem <= getMaxSharedMemPerBlockCurrentDevice());

#define RUN_PQ_OPT(LOOKUP_VEC_T)                            \
  do {                                                      \
    HQSecondStageDistances<LOOKUP_VEC_T>                    \
      <<<grid, block, smem, stream>>>(                      \
        deviceListIds,                                      \
        deviceIMIUpperBounds,                               \
        deviceDistanceTable,                                \
        deviceListCodes,                                    \
        deviceListLengths,                                  \
        devicePrefixSumOffsets,                             \
        nprobeSquareLen,                                    \
        imiSize,                                            \
        deviceDistances);                                   \
  } while (0)

#ifdef FAISS_USE_FLOAT16
#define RUN_PQ()                       \
  do {                                 \
    if (sizeof(LookupT) == 2) {        \
      RUN_PQ_OPT(Half8);               \
    } else {                           \
      RUN_PQ_OPT(float4);              \
    }                                  \
  } while (0)
#else
#define RUN_PQ()                       \
    do {                               \
      RUN_PQ_OPT(float4);              \
    } while (0)
#endif // FAISS_USE_FLOAT16

  RUN_PQ();

#undef RUN_PQ
#undef RUN_PQ_OPT
}

void runHQSecondStage(const Tensor<int, 3, true>& deviceIMIIndices,
                      const Tensor<int, 2, true>& deviceIMIUpperBounds,
                      const Tensor<float, 4, true>& deviceDistanceTable,
                      const void** deviceListCodes,
                      const int* deviceListLengths,
                      int numQueries,
                      int k,
                      int numListsPerQuery,
                      int nprobeSquareLen,
                      int imiSize,
                      bool chooseLargest,
                      // (field, qid, rank) -> val
                      // field 0: imiId0
                      // field 1: imiId1
                      // field 2: fineId
                      Tensor<int, 3, true>& deviceOutIndices,
                      GpuResources* resources,
                      cudaStream_t stream) {
    auto& mem = resources->getMemoryManagerCurrentDevice();

    using ListIdT = unsigned long long; // TODO: make it configurable

    DeviceTensor<ListIdT, 2, true> deviceListIds(mem,
            {numQueries, numListsPerQuery}, stream);
    runHQCalcListIds(deviceIMIIndices,
                     deviceIMIUpperBounds,
                     numQueries,
                     numListsPerQuery,
                     nprobeSquareLen,
                     imiSize,
                     deviceListIds,
                     stream);

    DeviceTensor<int, 2, true> devicePrefixSumOffsets(mem,
            {numQueries, numListsPerQuery}, stream);
    runHQCalcListOffsets(deviceListLengths,
                         deviceListIds,
                         devicePrefixSumOffsets.data(),
                         resources,
                         stream);

    constexpr int maxListLen = 128; // FIXME: set a correct value
    DeviceTensor<float, 2, true> deviceDistances(mem, {numQueries, maxListLen * numListsPerQuery}, stream);
    Tensor<float, 1, true> deviceDistancesFlat = deviceDistances.downcastInner<1>();

    runHQSecondStageDistances(deviceListIds,
                              deviceIMIUpperBounds,
                              deviceDistanceTable,
                              deviceListCodes,
                              deviceListLengths,
                              devicePrefixSumOffsets,
                              numQueries,
                              nprobeSquareLen,
                              imiSize,
                              deviceDistancesFlat,
                              stream);

    constexpr int kNProbeSplit = 8;
    int pass2Chunks = std::min(numListsPerQuery, kNProbeSplit);

    DeviceTensor<float, 3, true> deviceHeapDistances(mem, {numQueries, pass2Chunks, k}, stream);
    DeviceTensor<int, 3, true> deviceHeapIndices(mem, {numQueries, pass2Chunks, k}, stream);

    runPass1SelectLists(devicePrefixSumOffsets,
                        deviceDistancesFlat,
                        numListsPerQuery,
                        k,
                        chooseLargest,
                        deviceHeapDistances,
                        deviceHeapIndices,
                        stream);

    Tensor<float, 2, true> deviceHeapDistancesFlat = deviceHeapDistances.downcastInner<2>();
    Tensor<int, 2, true> deviceHeapIndicesFlat = deviceHeapIndices.downcastInner<2>();
    runPass2SelectIMILists(deviceHeapDistancesFlat,
                           deviceHeapIndicesFlat,
                           devicePrefixSumOffsets,
                           deviceListIds,
                           k,
                           imiSize,
                           chooseLargest,
                           deviceOutIndices,
                           stream);
}

} } // namespace
