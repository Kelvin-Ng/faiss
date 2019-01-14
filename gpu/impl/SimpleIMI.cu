/**
 * Copyright (c) 2019-present, Husky Data Lab.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "SimpleIMI.cuh"

#include "Distance.cuh"
#include "SimpleIMICut.cuh"
#include "../utils/DeviceTensor.cuh"

namespace faiss { namespace gpu {

void SimpleIMI::query(const Tensor<float, 2, true>& deviceQueries,
                      int nprobeSquareLen,
                      int nprobeSideLen,
                      // (imiId, qid, rank) -> index
                      Tensor<int, 3, true>& deviceOutIndices,
                      // (imiId, qid) -> upper_bound
                      Tensor<int, 2, true>& deviceOutUpperBounds) {
    auto stream = resources_->getDefaultStreamCurrentDevice();
    auto& mem = resources_->getMemoryManagerCurrentDevice();

    DeviceTensor<float, 3, true> deviceOutDistances(mem, {2, queries.getSize(0), deviceCentroids_.getSize(1)}, stream);

    // TODO: the two runDistance calls can run concurrently (i.e. in different streams)
    // TODO: Do not do k-selection
    for (int imiId = 0; imiId < 2; ++imiId) {
        Tensor<float, 2, true> deviceCentroidsView = deviceCentroids_.narrowOutermost(imiId, 1).view<2>();
        Tensor<float, 2, true> queriesView = queries.narrow(1, imiId * queries.getSize(1) / 2, queries.getSize(1) / 2); // handle when query dim is not even
        Tensor<float, 2, true> deviceOutDistancesView = deviceOutDistances.narrowOutermost(imiId, 1).view<2>();
        Tensor<int, 2, true> deviceOutIndicesView = deviceOutIndices.narrowOutermost(imiId, 1).view<2>();

        // TODO: check if it is really sorted
        runDistance(l2Distance_,
                    resources_,
                    deviceCentroidsView,
                    nullptr,
                    nullptr, // compute norms in temp memory. TODO: can store it because it does not consume much memory
                    queriesView,
                    deviceCentroids_.getSize(1),
                    deviceOutDistancesView,
                    deviceOutIndicesView,
                    false,
                    false);
    }

    runSimpleIMICut(deviceOutDistances, deviceOutUpperBounds, nprobeSquareLen, nprobeSideLen, stream);
}

} } // namespace
