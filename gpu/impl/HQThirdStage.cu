/**
 * Copyright (c) 2019-present, Husky Data Lab.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "HQThirdStage.cuh"

#include "HQThirdStageL2Distances.cuh"

#include "../utils/DeviceTensor.cuh"
#include "../utils/BlockSelectKernel.cuh"

namespace faiss { namespace gpu {

__global__ void HQThirdStageConvertIndices(const Tensor<int, 3, true> indices, const Tensor<int, 2, true> toConvert, Tensor<int, 3, true> converted) {
    int qid = blockIdx.x;
    int rank = threadIdx.x;

    int idx = toConvert[qid][rank];
    converted[0][qid][rank] = (int)indices[0][qid][idx];
    converted[1][qid][rank] = (int)indices[1][qid][idx];
    converted[2][qid][rank] = (int)indices[2][qid][idx];
}

void runHQThirdStageConvertIndices(const Tensor<int, 3, true>& deviceIndices, const Tensor<int, 2, true>& deviceToConvert, Tensor<int, 3, true>& deviceConverted) {
    HQThirdStageConvertIndices<<<deviceToConvert.getSize(0), deviceToConvert.getSize(1)>>>(deviceIndices, deviceToConvert, deviceConverted);
}

void runHQThirdStage(const Tensor<float, 2, true>& deviceQueries,
                     const Tensor<int, 3, true>& deviceIndices,
                     const void** deviceListCodes1,
                     const void** deviceListCodes2,
                     const Tensor<float, 3, true>& deviceCodewordsIMI,
                     const Tensor<float, 4, true>& deviceCodewords1,
                     const Tensor<float, 4, true>& deviceCodewords2,
                     int imiSize,
                     int numCodes2,
                     int k,
                     bool l2Distance,
                     Tensor<float, 2, true>& deviceOutDistances,
                     Tensor<int, 3, true>& deviceOutIndices,
                     GpuResources* resources,
                     cudaStream_t stream) {
    FAISS_ASSERT_MSG(l2Distance, "runHQThirdStage currently supports only L2 distance");

    auto& mem = resources->getMemoryManagerCurrentDevice();

    int numQueries = deviceQueries.getSize(0);
    int numItems = deviceIndices.getSize(2);

    DeviceTensor<float, 2, true> deviceAllDistances(mem, {numQueries, numItems}, stream);
    // TODO: support inner product
    runHQThirdStageL2Distances(deviceQueries, deviceIndices, deviceListCodes1, deviceListCodes2, deviceCodewordsIMI, deviceCodewords1, deviceCodewords2, imiSize, numCodes2, deviceAllDistances, true, stream);

    DeviceTensor<int, 2, true> deviceTmpIndices(mem, {numQueries, k}, stream);
    runBlockSelect(deviceAllDistances, deviceOutDistances, deviceTmpIndices, !l2Distance, k, stream);

    // TODO: should I fuse it with block select?
    runHQThirdStageConvertIndices(deviceIndices, deviceTmpIndices, deviceOutIndices);
}

} } // namespace
