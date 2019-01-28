/**
 * Copyright (c) 2019-present, Husky Data Lab.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */


#include "GpuIndexHQ.h"

#include "utils/DeviceUtils.h"
#include "utils/CopyUtils.cuh"

#include <memory>

namespace faiss { namespace gpu {

GpuIndexHQ::GpuIndexHQ(GpuResources* resources,
                       int dims,
                       faiss::MetricType metric,
                       int imiSize,
                       int numData,
                       int numCodes2,
                       int imiNprobeSquareLen,
                       int imiNprobeSideLen,
                       int secondStageNProbe,
                       const float* centroids,
                       const float* fineCentroids,
                       const float* codewords1,
                       const float* codewords2,
                       const unsigned char* listCodes1Data,
                       const unsigned char* listCodes2Data,
                       const faiss::Index::idx_t* listIndicesData,
                       const int* listLengths,
                       GpuIndexConfig config) :
    GpuIndex(resources, dims, metric, config),
    imiNprobeSquareLen_(imiNprobeSquareLen),
    imiNprobeSideLen_(imiNprobeSideLen),
    secondStageNProbe_(secondStageNProbe) {
  auto stream = resources_->getDefaultStream(device_);

  deviceCentroids_ = toDevice<float, 3>(resources_, device_, const_cast<float*>(centroids), stream, {2, imiSize, dims});
  auto deviceFineCentroids = toDevice<float, 4>(resources_, device_, const_cast<float*>(fineCentroids), stream, {2, imiSize, 256, dims});
  auto deviceCodewords1 = toDevice<float, 4>(resources_, device_, const_cast<float*>(codewords1), stream, {2, imiSize, 256, dims});
  auto deviceCodewords2 = toDevice<float, 4>(resources_, device_, const_cast<float*>(codewords2), stream, {numCodes2 / 2, 256, 256, dims});
  thrust::device_vector<unsigned char> deviceListCodes1Data(numData * 2);
  cudaMemcpy(deviceListCodes1Data.data().get(), listCodes1Data, numData * 2, cudaMemcpyHostToDevice);
  thrust::device_vector<unsigned char> deviceListCodes2Data(numData * numCodes2);
  cudaMemcpy(deviceListCodes2Data.data().get(), listCodes2Data, numData * numCodes2, cudaMemcpyHostToDevice);
  thrust::device_vector<int> deviceListLengths(imiSize * imiSize);
  cudaMemcpy(deviceListLengths.data().get(), listLengths, imiSize * imiSize, cudaMemcpyHostToDevice);

  // TODO: make_unique should be safer, but is not supported in C++11

  simpleIMI_.reset(new SimpleIMI(resources_,
                                 deviceCentroids_,
                                 metric == faiss::METRIC_L2));

  index_.reset(new HQ(resources_,
                      std::move(deviceFineCentroids),
                      deviceCentroids_,
                      std::move(deviceCodewords1),
                      std::move(deviceCodewords2),
                      std::move(deviceListCodes1Data),
                      std::move(deviceListCodes2Data),
                      listIndicesData,
                      std::move(deviceListLengths),
                      listLengths,
                      simpleIMI_.get(),
                      numCodes2,
                      metric == faiss::METRIC_L2));
}

void
GpuIndexHQ::searchImpl_(faiss::Index::idx_t n,
                        const float* x,
                        faiss::Index::idx_t k,
                        float* distances,
                        faiss::Index::idx_t* labels) const {
  // Device is already set in GpuIndex::search
  FAISS_ASSERT(index_);
  FAISS_ASSERT(n > 0);

  auto stream = resources_->getDefaultStream(device_);
  auto& mem = resources_->getMemoryManager(device_);

  // Make sure arguments are on the device we desire; use temporary
  // memory allocations to move it if necessary
  auto devX =
    toDevice<float, 2>(resources_,
                       device_,
                       const_cast<float*>(x),
                       stream,
                       {(int) n, this->d});

  DeviceTensor<float, 2, true> devDistances(mem, {(int)n, (int)k}, stream);
  Tensor<faiss::Index::idx_t, 2, true> labelsTensor(labels, {(int)n, (int)k});

  index_->query(devX, imiNprobeSquareLen_, imiNprobeSideLen_, secondStageNProbe_, k, devDistances, labelsTensor);

  // Copy back if necessary
  fromDevice<float, 2>(devDistances, distances, stream);
}

} } // namespace
