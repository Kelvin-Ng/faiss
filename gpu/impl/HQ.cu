/**
 * Copyright (c) 2019-present, Husky Data Lab.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */


#include "HQ.cuh"
#include "L2DistanceWithVectorHQ.cuh"
#include "HQSecondStage.cuh"
#include "HQThirdStage.cuh"
#include "../utils/HostTensor.cuh"
#include "../utils/CopyUtils.cuh"

#include <thrust/transform_scan.h>
#include <thrust/execution_policy.h>

namespace faiss { namespace gpu {

// Note: this function assumes that all vectors contain only the corresponding subspace
__global__ void computeHQL2DistanceTableOneIMI(const Tensor<float, 2, true> deviceQueries,
                                               // (qid, coarseRank) -> coarseIdx
                                               const Tensor<int, 2, true> deviceIMIIndices,
                                               // qid -> upper_bound
                                               Tensor<int, 1, true> deviceIMIUpperBounds,
                                               // (coarseIdx, fineIdx, dim) -> val
                                               Tensor<float, 3, true> fineCentroids,
                                               // (qid, coarseRank, fineIdx) -> val
                                               Tensor<float, 3, true> outDistances) {
    int qid = blockIdx.x;
    int upper_bound = deviceIMIUpperBounds[qid];
    auto outDistancesView = outDistances.narrowOutermost(qid, 1).view<2>();
    runL2DistanceWithVectorHQ(fineCentroids,
                              deviceIMIIndices.narrowOutermost(qid, 1).view<1>().narrowOutermost(0, upper_bound),
                              deviceQueries.narrowOutermost(qid, 1).view<1>(),
                              outDistancesView,
                              true,
                              0);
}

void runComputeHQL2DistanceTable(const Tensor<float, 2, true>& deviceQueries,
                                 // (imiId, qid, coarseRank) -> coarseIdx
                                 const Tensor<int, 3, true>& deviceIMIIndices,
                                 // (imiId, qid) -> upper_bound
                                 const Tensor<int, 2, true>& deviceIMIUpperBounds,
                                 // (imiId, coarseIdx, fineIdx, dim) -> val
                                 const Tensor<float, 4, true>& deviceFineCentroids,
                                 // (imiId, qid, coarseRank, fineIdx) -> val
                                 Tensor<float, 4, true>& outDistances,
                                 GpuResources* resources) {
    auto stream = resources->getDefaultStreamCurrentDevice();
    auto streams = resources->getAlternateStreamsCurrentDevice();

    streamWait(streams, {stream});

    for (int imiId = 0; imiId < 2; ++imiId) {
        computeHQL2DistanceTableOneIMI<<<deviceQueries.getSize(0), 1, 0, streams[imiId]>>>(
                deviceQueries.narrow(1, imiId * deviceQueries.getSize(1) / 2, deviceQueries.getSize(1) / 2), // get the data of the corresponding subspace
                deviceIMIIndices.narrowOutermost(imiId, 1).view<2>(),
                deviceIMIUpperBounds.narrowOutermost(imiId, 1).view<1>(),
                deviceFineCentroids.narrowOutermost(imiId, 1).view<3>(),
                outDistances.narrowOutermost(imiId, 1).view<3>());
    }

    streamWait({stream}, streams);
}

void runInitializeHQLists(thrust::device_vector<const void*>& deviceListCodes1,
                          thrust::device_vector<const void*>& deviceListCodes2,
                          std::vector<const faiss::Index::idx_t*>& listIndices,
                          const thrust::device_vector<int>& deviceListLengths,
                          const int* listLengths,
                          const thrust::device_vector<unsigned char>& deviceListCodes1Data,
                          const thrust::device_vector<unsigned char>& deviceListCodes2Data,
                          const std::vector<faiss::Index::idx_t>& listIndicesData,
                          int numCodes2,
                          GpuResources* resources) {
    auto stream = resources->getDefaultStreamCurrentDevice();
    auto streams = resources->getAlternateStreamsCurrentDevice();

    streamWait(streams, {stream});

    thrust::transform_exclusive_scan(thrust::cuda::par.on(streams[0]),
                                     deviceListLengths.begin(),
                                     deviceListLengths.end(),
                                     (uintptr_t*)deviceListCodes1.data().get(),
                                     [] __device__ (int len) -> uintptr_t { return (uintptr_t)len * 2; },
                                     (uintptr_t)deviceListCodes1Data.data().get(),
                                     thrust::plus<uintptr_t>());

    thrust::transform_exclusive_scan(thrust::cuda::par.on(streams[1]),
                                     deviceListLengths.begin(),
                                     deviceListLengths.end(),
                                     (uintptr_t*)deviceListCodes2.data().get(),
                                     [numCodes2] __device__ (int len) -> uintptr_t { return (uintptr_t)len * numCodes2; },
                                     (uintptr_t)deviceListCodes2Data.data().get(),
                                     thrust::plus<uintptr_t>());

    thrust::transform_exclusive_scan(thrust::host,
                                     listLengths,
                                     listLengths + deviceListLengths.size(),
                                     (uintptr_t*)listIndices.data(),
                                     [] (int len) -> uintptr_t { return (uintptr_t)len * sizeof(faiss::Index::idx_t); },
                                     (uintptr_t)listIndicesData.data(),
                                     thrust::plus<uintptr_t>());

    streamWait({stream}, streams);
}

HQ::HQ(GpuResources* resources,
       DeviceTensor<float, 4, true> deviceFineCentroids,
       DeviceTensor<float, 4, true> deviceCodewords2,
       thrust::device_vector<unsigned char> deviceListCodes1Data,
       thrust::device_vector<unsigned char> deviceListCodes2Data,
       std::vector<faiss::Index::idx_t> listIndicesData,
       thrust::device_vector<int> deviceListLengths,
       const int* listLengths,
       SimpleIMI* simpleIMI,
       int numCodes2,
       bool l2Distance) : resources_(resources),
                          simpleIMI_(simpleIMI),
                          imiSize_(deviceFineCentroids.getSize(1)),
                          numCodes2_(numCodes2),
                          l2Distance_(l2Distance),
                          deviceFineCentroids_(std::move(deviceFineCentroids)),
                          deviceCodewords2_(std::move(deviceCodewords2)),
                          deviceListCodes1Data_(std::move(deviceListCodes1Data)),
                          deviceListCodes2Data_(std::move(deviceListCodes2Data)),
                          listIndicesData_(std::move(listIndicesData)),
                          deviceListLengths_(std::move(deviceListLengths)),
                          deviceListCodes1_(deviceListLengths_.size()),
                          deviceListCodes2_(deviceListLengths_.size()),
                          listIndices_(deviceListLengths_.size()) {
    runInitializeHQLists(deviceListCodes1_,
                         deviceListCodes2_,
                         listIndices_,
                         deviceListLengths_,
                         listLengths,
                         deviceListCodes1Data_,
                         deviceListCodes2Data_,
                         listIndicesData_,
                         numCodes2_,
                         resources_);
}

void HQ::query(const Tensor<float, 2, true>& deviceQueries, int imiNprobeSquareLen, int imiNprobeSideLen, int secondStageNProbe, int k, Tensor<float, 2, true>& deviceOutDistances, Tensor<faiss::Index::idx_t, 2, true>& outIndices) {
    auto stream = resources_->getDefaultStreamCurrentDevice();
    auto& mem = resources_->getMemoryManagerCurrentDevice();

    DeviceTensor<int, 3, true> deviceIMIOutIndices(mem, {2, deviceQueries.getSize(0), imiSize_}, stream);
    DeviceTensor<int, 2, true> deviceIMIOutUpperBounds(mem, {2, deviceQueries.getSize(0)}, stream);
    simpleIMI_->query(deviceQueries, imiNprobeSquareLen, imiNprobeSideLen, deviceIMIOutIndices, deviceIMIOutUpperBounds);

    int imiNprobe = imiNprobeSideLen * imiNprobeSquareLen - imiNprobeSquareLen * imiNprobeSquareLen;

    DeviceTensor<float, 4, true> deviceDistanceTable(mem, {2, deviceQueries.getSize(0), imiNprobeSideLen, deviceFineCentroids_.getSize(2)}, stream); // the third dimension is an over-estimation of the actual size
    // TODO: handle inner product
    runComputeHQL2DistanceTable(deviceQueries, deviceIMIOutIndices, deviceIMIOutUpperBounds, deviceFineCentroids_, deviceDistanceTable, resources_);

    DeviceTensor<int, 3, true> deviceSecondStageIndices(mem, {3, deviceQueries.getSize(0), secondStageNProbe}, stream);
    runHQSecondStage(deviceIMIOutIndices,
                     deviceIMIOutUpperBounds,
                     deviceDistanceTable,
                     deviceListCodes1_.data().get(),
                     deviceListLengths_.data().get(),
                     secondStageNProbe,
                     imiNprobe,
                     imiNprobeSquareLen,
                     imiSize_,
                     !l2Distance_,
                     deviceSecondStageIndices,
                     resources_,
                     stream);

    DeviceTensor<int, 3, true> deviceThirdStageIndices(mem, {3, deviceQueries.getSize(0), k}, stream);
    runHQThirdStage(deviceQueries,
                    deviceSecondStageIndices,
                    deviceListCodes1_.data().get(),
                    deviceListCodes2_.data().get(),
                    deviceFineCentroids_,
                    deviceCodewords2_,
                    imiSize_,
                    numCodes2_,
                    k,
                    l2Distance_,
                    deviceOutDistances,
                    deviceThirdStageIndices,
                    resources_,
                    stream);

    HostTensor<int, 3, true> thirdStageIndices({3, deviceQueries.getSize(0), k});
    fromDevice<int, 3>(deviceThirdStageIndices, thirdStageIndices.data(), stream);
    cudaDeviceSynchronize();
    
    for (int qid = 0; qid < deviceQueries.getSize(0); ++qid) {
        for (int rank = 0; rank < k; ++rank) {
            uint64_t listId = (uint64_t)thirdStageIndices[0][qid][rank] * imiSize_ + thirdStageIndices[1][qid][rank];
            int offset = thirdStageIndices[2][qid][rank];
            outIndices[qid][rank] = listIndices_[listId][offset];
        }
    }
}

} } // namespace
