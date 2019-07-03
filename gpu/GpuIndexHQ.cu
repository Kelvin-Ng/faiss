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
                       unsigned long long numData,
                       int numCodes2,
                       int imiNprobeSquareLen,
                       int imiNprobeSideLen,
                       int secondStageNProbe,
                       const float* centroids,
                       const float* fineCentroids,
                       const float* codewords2,
                       const unsigned char* listCodes1Data,
                       const unsigned char* listCodes2Data,
                       const faiss::Index::idx_t* listIndicesData,
                       const int* listLengths,
                       const float* rotate,
                       GpuIndexConfig config) :
    GpuIndex(resources, dims, metric, config),
    imiNprobeSquareLen_(imiNprobeSquareLen),
    imiNprobeSideLen_(imiNprobeSideLen),
    secondStageNProbe_(secondStageNProbe) {
  auto stream = resources_->getDefaultStream(device_);

  auto deviceCentroids = toDevice<float, 3>(resources_, device_, const_cast<float*>(centroids), stream, {2, imiSize, dims / 2}); // TODO: Handle cases when dims is not divisible by 2
  auto deviceFineCentroids = toDevice<float, 4>(resources_, device_, const_cast<float*>(fineCentroids), stream, {2, imiSize, 256, dims / 2}); // TODO: Handle cases when dims is not divisible by 2
  auto deviceCodewords2 = toDevice<float, 4>(resources_, device_, const_cast<float*>(codewords2), stream, {numCodes2 / 2, 257, 256, dims});
  auto deviceRotate = toDevice<float, 2>(resources_, device_, const_cast<float*>(rotate), stream, {dims, dims});
  thrust::device_vector<unsigned char> deviceListCodes1Data(numData * 2);
  cudaMemcpy(deviceListCodes1Data.data().get(), listCodes1Data, numData * 2, cudaMemcpyHostToDevice);
  thrust::device_vector<unsigned char> deviceListCodes2Data(numData * numCodes2);
  cudaMemcpy(deviceListCodes2Data.data().get(), listCodes2Data, numData * numCodes2, cudaMemcpyHostToDevice);
  thrust::device_vector<int> deviceListLengths(imiSize * imiSize);
  cudaMemcpy(deviceListLengths.data().get(), listLengths, imiSize * imiSize * sizeof(int), cudaMemcpyHostToDevice);
  std::vector<faiss::Index::idx_t> listIndicesDataOwned(listIndicesData, listIndicesData + numData);

  // TODO: make_unique should be safer, but is not supported in C++11

  simpleIMI_.reset(new SimpleIMI(resources_,
                                 std::move(deviceCentroids),
                                 metric == faiss::METRIC_L2));

  index_.reset(new HQ(resources_,
                      std::move(deviceFineCentroids),
                      std::move(deviceCodewords2),
                      std::move(deviceListCodes1Data),
                      std::move(deviceListCodes2Data),
                      std::move(listIndicesDataOwned),
                      std::move(deviceListLengths),
                      listLengths,
                      std::move(deviceRotate),
                      simpleIMI_.get(),
                      numCodes2,
                      metric == faiss::METRIC_L2));
}

void
GpuIndexHQ::search(Index::idx_t n,
                   const float* x,
                   Index::idx_t k,
                   float* distances,
                   Index::idx_t* labels) const {
  FAISS_THROW_IF_NOT_MSG(this->is_trained, "Index not trained");

  // For now, only support <= max int results
  FAISS_THROW_IF_NOT_FMT(n <= (Index::idx_t) std::numeric_limits<int>::max(),
                         "GPU index only supports up to %d indices",
                         std::numeric_limits<int>::max());

  // Maximum k-selection supported is based on the CUDA SDK
  FAISS_THROW_IF_NOT_FMT(k <= (Index::idx_t) getMaxKSelection(),
                         "GPU index only supports k <= %d (requested %d)",
                         getMaxKSelection(),
                         (int) k); // select limitation

  if (n == 0 || k == 0) {
    // nothing to search
    return;
  }

  DeviceScope scope(device_);
  auto stream = resources_->getDefaultStream(device_);

  // We guarantee that the searchImpl_ will be called with device-resident
  // pointers.

  // The input vectors may be too large for the GPU, but we still
  // assume that the output distances and labels are not.
  // Go ahead and make space for output distances and labels on the
  // GPU.
  // If we reach a point where all inputs are too big, we can add
  // another level of tiling.
  auto outDistances =
    toDevice<float, 2>(resources_, device_, distances, stream,
                       {(int) n, (int) k});

  int labels_dev = getDeviceForAddress(labels);
  HostTensor<faiss::Index::idx_t, 2> outLabels;
  if (labels_dev == -1) {
    outLabels = HostTensor<faiss::Index::idx_t, 2>(labels, {(int)n, (int)k});
  } else {
    outLabels = HostTensor<faiss::Index::idx_t, 2>({(int)n, (int)k});
  }

  bool usePaged = false;

  if (getDeviceForAddress(x) == -1) {
    // It is possible that the user is querying for a vector set size
    // `x` that won't fit on the GPU.
    // In this case, we will have to handle paging of the data from CPU
    // -> GPU.
    // Currently, we don't handle the case where the output data won't
    // fit on the GPU (e.g., n * k is too large for the GPU memory).
    size_t dataSize = (size_t) n * this->d * sizeof(float);

    if (dataSize >= minPagedSize_) {
      searchFromCpuPaged_(n, x, k,
                          outDistances.data(),
                          outLabels.data());
      usePaged = true;
    }
  }

  if (!usePaged) {
    searchNonPaged_(n, x, k,
                    outDistances.data(),
                    outLabels.data());
  }

  // Copy back if necessary
  fromDevice<float, 2>(outDistances, distances, stream);
  if (labels_dev != -1) {
    DeviceTensor<faiss::Index::idx_t, 2> labelsTensor(labels, {(int)n, (int)k});
    labelsTensor.copyFrom(outLabels, stream);
  }
}

void
GpuIndexHQ::searchImpl_(int n,
                        const float* x,
                        int k,
                        float* distances,
                        faiss::Index::idx_t* labels) const {
  // Device is already set in GpuIndex::search
  FAISS_ASSERT(index_);
  FAISS_ASSERT(n > 0);

  // Data is already resident on the GPU
  Tensor<float, 2, true> queries(const_cast<float*>(x), {n, (int) this->d});
  Tensor<float, 2, true> deviceOutDistances(distances, {n, k});

  static_assert(sizeof(long) == sizeof(Index::idx_t), "size mismatch");
  Tensor<long, 2, true> outLabels(const_cast<long*>(labels), {n, k});

  index_->query(queries, imiNprobeSquareLen_, imiNprobeSideLen_, secondStageNProbe_, k, deviceOutDistances, outLabels);
}

} } // namespace
