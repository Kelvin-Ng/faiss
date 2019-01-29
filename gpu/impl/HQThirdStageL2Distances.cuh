/**
 * Copyright (c) 2019-present, Husky Data Lab.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "../utils/Tensor.cuh"

namespace faiss { namespace gpu {

void runHQThirdStageL2Distances(const Tensor<float, 2, true>& queries,
                                const Tensor<int, 3, true>& indices,
                                const void** listCodes1,
                                const void** listCodes2,
                                const Tensor<float, 4, true>& codewords1,
                                const Tensor<float, 4, true>& codewords2,
                                int imiSize,
                                int numCodes2,
                                Tensor<float, 2, true>& distances,
                                bool normSquared,
                                cudaStream_t stream);

} } // namespace
