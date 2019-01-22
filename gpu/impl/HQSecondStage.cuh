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

void runHQSecondStage(const Tensor<int, 3, true>& deviceIMIIndices,
                      const Tensor<int, 2, true>& deviceIMIUpperBounds,
                      const Tensor<float, 4, true>& deviceDistanceTable,
                      const void** deviceListCodes,
                      const int* deviceListLengths,
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
                      cudaStream_t stream);

} } // namespace
