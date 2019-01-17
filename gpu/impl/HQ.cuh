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
       int bytesPerVector,
       IndicesOptions indicesOptions,
       MemorySpace space);

    void query();

  protected:
    GpuResources* resources_;
    SimpleIMI* simpleIMI_;
    const Tensor<float, 4, true>& fineCentroids_;
};


} } // namespace
