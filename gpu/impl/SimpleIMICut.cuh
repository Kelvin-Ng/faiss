/**
 * Copyright (c) 2019-present, Husky Data Lab.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */


#pragma once

#include "../utils/Float16.cuh"
#include "../utils/Tensor.cuh"

namespace faiss { namespace gpu {

void runSimpleIMICut(// (imiId, qid, rank) -> index
                     const Tensor<float, 3, true>& imiDistances,
                     // (imiId, qid) -> upper_bound
                     Tensor<int, 2, true>& imiUpperBounds,
                     int size,
                     cudaStream_t stream);

#ifdef FAISS_USE_FLOAT16
void runSimpleIMICut(// (imiId, qid, rank) -> index
                     const Tensor<half, 3, true>& imiDistances,
                     // (imiId, qid) -> upper_bound
                     Tensor<int, 2, true>& imiUpperBounds,
                     int size,
                     cudaStream_t stream);
#endif

} } // namespace
