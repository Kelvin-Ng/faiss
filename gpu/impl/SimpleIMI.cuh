/**
 * Copyright (c) 2019-present, Husky Data Lab.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */


#pragma once

#include "../GpuResources.h"
#include "../utils/Tensor.cuh"
#include "../utils/DeviceTensor.cuh"

namespace faiss { namespace gpu {

class SimpleIMI {
  public:
    SimpleIMI(GpuResources* resources,
              DeviceTensor<float, 3, true> deviceCentroids, // (imiId, centroidId, dim)
              bool l2Distance) : resources_(resources),
                                 deviceCentroids_(std::move(deviceCentroids)),
                                 l2Distance_(l2Distance) {}

    void query(const Tensor<float, 2, true>& deviceQueries,
               int nprobeSquareLen,
               int nprobeSideLen,
               // (imiId, qid, rank) -> index
               Tensor<int, 3, true>& deviceOutIndices,
               // (imiId, qid) -> upper_bound
               Tensor<int, 2, true>& deviceOutUpperBounds);

    virtual ~SimpleIMI() {}

  protected:
    GpuResources* resources_;
    DeviceTensor<float, 3, true> deviceCentroids_;
    bool l2Distance_;
};

} } // namespace
