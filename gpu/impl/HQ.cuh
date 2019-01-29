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
#include "../utils/DeviceTensor.cuh"
#include "../../Index.h"

#include <thrust/device_vector.h>

namespace faiss { namespace gpu {

class HQ {
  public:
    HQ(GpuResources* resources,
       DeviceTensor<float, 4, true> deviceFineCentroids,
       const Tensor<float, 3, true>& deviceCodewordsIMI,
       DeviceTensor<float, 4, true> deviceCodewords1,
       DeviceTensor<float, 4, true> deviceCodewords2,
       thrust::device_vector<unsigned char> deviceListCodes1Data,
       thrust::device_vector<unsigned char> deviceListCodes2Data,
       const faiss::Index::idx_t* listIndicesData,
       thrust::device_vector<int> deviceListLengths,
       const int* listLengths,
       SimpleIMI* simpleIMI,
       int numCodes2,
       bool l2Distance);

    void query(const Tensor<float, 2, true>& deviceQueries, int imiNprobeSquareLen, int imiNprobeSideLen, int secondStageNProbe, int k, Tensor<float, 2, true>& deviceOutDistances, Tensor<faiss::Index::idx_t, 2, true>& outIndices);

  protected:
    GpuResources* resources_;
    SimpleIMI* simpleIMI_;
    const bool l2Distance_;
    // (imiId, coarseIdx, fineIdx, dim) -> val
    DeviceTensor<float, 4, true> deviceFineCentroids_;
    thrust::device_vector<const void*> deviceListCodes1_;
    thrust::device_vector<const void*> deviceListCodes2_;
    std::vector<const faiss::Index::idx_t*> listIndices_;
    thrust::device_vector<unsigned char> deviceListCodes1Data_;
    thrust::device_vector<unsigned char> deviceListCodes2Data_;
    thrust::device_vector<int> deviceListLengths_;
    const Tensor<float, 3, true>& deviceCodewordsIMI_;
    DeviceTensor<float, 4, true> deviceCodewords1_;
    DeviceTensor<float, 4, true> deviceCodewords2_;
    int imiSize_;
    int numCodes2_;
};


} } // namespace
