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
    runL2DistanceWithVectorHQ(fineCentroids,
                              deviceIMIIndices.narrowOutermost(qid, 1).view<1>().narrowOutermost(0, upper_bound),
                              deviceQueries.narrowOutermost(qid, 1).view<1>(),
                              outDistances.narrowOutermost(qid, 1).view<2>(),
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

    for (int imiId = 0; imiId < 2; ++imiId) {
        computeHQL2DistanceTableOneIMI<<<deviceQueries.getSize(0), 1, 0, streams[imiId]>>>(
                deviceQueries.narrow(1, imiId * deviceQueries.getSize(1) / 2, deviceQueries.getSize(1) / 2), // get the data of the corresponding subspace
                deviceIMIIndices.narrowOutermost(imiId, 1).view<2>(),
                deviceIMIUpperBounds.narrowOutermost(imiId, 1).view<1>(),
                deviceFineCentroids.narrowOutermost(imiId, 1).view<3>(),
                outDistances.narrowOutermost(imiId, 1).view<3>());
    }

    streamWait(streams, {stream});
}

HQ::HQ(GpuResources* resources,
       DeviceTensor<float, 4, true> deviceFineCentroids,
       DeviceTensor<float, 3, true> deviceCodewordsIMI,
       DeviceTensor<float, 4, true> deviceCodewords1,
       DeviceTensor<float, 4, true> deviceCodewords2,
       thrust::device_vector<unsigned char> deviceListCodes1Data,
       thrust::device_vector<unsigned char> deviceListCodes2Data,
       SimpleIMI* simpleIMI,
       int numCodes2,
       bool l2Distance) : resources_(resources),
                          deviceFineCentroids_(std::move(deviceFineCentroids)),
                          deviceCodewordsIMI_(std::move(deviceCodewordsIMI)),
                          deviceCodewords1_(std::move(deviceCodewords1)),
                          deviceCodewords2_(std::move(deviceCodewords2)),
                          deviceListCodes1Data_(std::move(deviceListCodes1Data)),
                          deviceListCodes2Data_(std::move(deviceListCodes2Data)),
                          deviceListLengths_(std::move(deviceListLengths)),
                          deviceListCodes1_(deviceListLengths_.size()),
                          deviceListCodes2_(deviceListLengths_.size()),
                          simpleIMI_(simpleIMI),
                          numCodes2_(numCodes2),
                          l2Distance_(l2Distance) {
    auto stream = resources_->getDefaultStreamCurrentDevice();
    auto streams = resources->getAlternateStreamsCurrentDevice();

    thrust::transform_exclusive_scan(thrust::cuda::par.on(streams[0]), deviceListLengths_.begin(), deviceListLengths_.end(), deviceListCodes1_.begin(), [](int len) { return (unsigned long long)len * 4 }, deviceListCodes1Data_.data().get(), thrust::plus<unsigned long long>());
    thrust::transform_exclusive_scan(thrust::cuda::par.on(streams[1]), deviceListLengths_.begin(), deviceListLengths_.end(), deviceListCodes2_.begin(), [](int len) { return (unsigned long long)len * 4 }, deviceListCodes2Data_.data().get(), thrust::plus<unsigned long long>());

    streamWait(streams, {stream});
}

void HQ::query(const Tensor<float, 2, true>& deviceQueries, int imiNprobeSquareLen, int imiNprobeSideLen, int secondStageNProbe, int k, Tensor<float, 2, true>& deviceOutDistances, Tensor<int, 3, true>& deviceOutIndices) {
    auto stream = resources_->getDefaultStreamCurrentDevice();
    auto& mem = resources_->getMemoryManagerCurrentDevice();

    DeviceTensor<int, 3, true> deviceIMIOutIndices(mem, {2, deviceQueries.getSize(0), deviceFineCentroids_.getSize(1)}, stream);
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
                     deviceListCodes1_,
                     deviceListLengths_,
                     secondStageNProbe,
                     imiNprobe,
                     imiNprobeSquareLen,
                     deviceFineCentroids_.getSize(1),
                     !l2Distance_,
                     deviceSecondStageIndices,
                     resources_,
                     stream);

    runHQThirdStage(deviceQueries,
                    deviceSecondStageIndices,
                    deviceListCodes1_,
                    deviceListCodes2_,
                    deviceCodewordsIMI_,
                    deviceCodewords1_,
                    deviceCodewords2_,
                    deviceFineCentroids_.getSize(1),
                    numCodes2_,
                    k,
                    l2Distance_,
                    deviceOutDistances,
                    deviceOutIndices,
                    resources_,
                    stream);
}

} } // namespace
