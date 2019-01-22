/**
 * Copyright (c) 2019-present, Husky Data Lab.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */


#pragma once

#include "SimpleIMI.cuh"
#include "../GpuResources.h"
#include "../utils/Tensor.cuh"

namespace faiss { namespace gpu {

class HQ {
  public:
    HQ(GpuResources* resources,
       SimpleIMI* simpleIMI,
       bool l2Distance) : resources_(resources),
                          simpleIMI_(simpleIMI),
                          l2Distance_(l2Distance) {}

    void query(const Tensor<float, 2, true>& deviceQueries, int imiNprobeSquareLen, int imiNprobeSideLen, int secondStageNProbe, int k, Tensor<float, 2, true>& deviceOutDistances, Tensor<int, 3, true>& deviceOutIndices);

  protected:
    GpuResources* resources_;
    SimpleIMI* simpleIMI_;
    const bool l2Distance_;
    // (imiId, coarseIdx, fineIdx, dim) -> val
    const Tensor<float, 4, true> deviceFineCentroids_;
    const void** deviceListCodes1_;
    const void** deviceListCodes2_;
    const int* deviceListLengths_;
    const Tensor<float, 4, true> deviceCodewords1_;
    const Tensor<float, 4, true> deviceCodewords2_;
    int numCodes2_;
};


} } // namespace
