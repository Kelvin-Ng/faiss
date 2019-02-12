/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */


#pragma once

#include "../utils/Float16.cuh"
#include "../utils/Tensor.cuh"

namespace faiss { namespace gpu {

__device__
void runL2DistanceWithVectorHQ(const Tensor<float, 3, true>& input, // (coarseIdx, fineIdx, dim) -> val
                               const Tensor<int, 1, true>& inputIndices, // coarseRank -> coarseIdx
                               const Tensor<float, 1, true>& vec,
                               Tensor<float, 2, true>& output, // (coarseRank, fineIdx) -> val
                               bool normSquared,
                               cudaStream_t stream);

#ifdef FAISS_USE_FLOAT16
__device__
void runL2DistanceWithVectorHQ(const Tensor<half, 3, true>& input, // (coarseIdx, fineIdx, dim) -> val
                               const Tensor<int, 1, true>& inputIndices, // coarseRank -> coarseIdx
                               const Tensor<half, 1, true>& vec,
                               Tensor<float, 2, true>& output, // (coarseRank, fineIdx) -> val
                               bool normSquared,
                               cudaStream_t stream);
#endif

} } // namespace
