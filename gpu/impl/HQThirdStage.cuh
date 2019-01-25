/**
 * Copyright (c) 2019-present, Husky Data Lab.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */


#pragma once

#include "../utils/Tensor.cuh"
#include "../GpuResources.h"

namespace faiss { namespace gpu {

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
                     cudaStream_t stream);

} } // namespace

