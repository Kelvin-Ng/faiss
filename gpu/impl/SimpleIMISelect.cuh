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

void runSimpleIMISelect(const Tensor<float, 3, true>& input,
                        Tensor<int, 2, true>& output,
                        int S;
	                int T;
                        cudaStream_t stream);

#ifdef FAISS_USE_FLOAT16
void runSimpleIMISelect(const Tensor<half, 3, true>& input,
               		Tensor<int, 2, true>& output,
               		int S;
	       		int T;
               		cudaStream_t stream);
#endif

} } // namespace
