/**
 * Copyright (c) 2019-present, Husky Data Lab.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */


#pragma once

#include "GpuIndex.h"
#include "GpuIndicesOptions.h"
#include "GpuResources.h"
#include "impl/HQ.cuh"
#include "impl/SimpleIMI.cuh"

#include <memory>

namespace faiss { namespace gpu {

class GpuIndexHQ : public GpuIndex {
  public:
    GpuIndexHQ(GpuResources* resources,
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
               GpuIndexConfig config = GpuIndexConfig());

  protected:
    virtual void addImpl_(int n,
                          const float* x,
                          const Index::idx_t* ids) override {
      FAISS_ASSERT_MSG(false, "Not implemented");
    }

    virtual bool addImplRequiresIDs_() const override {
      return false;
    }

    void reset() override {
      FAISS_ASSERT_MSG(false, "Not implemented");
    }

    // Called from GpuIndex for search
    void searchImpl_(
        int n,
        const float* x,
        int k,
        float* distances,
        faiss::Index::idx_t* labels) const override;

    void search(Index::idx_t n,
                const float* x,
                Index::idx_t k,
                float* distances,
                Index::idx_t* labels) const override;

    int imiNprobeSquareLen_;
    int imiNprobeSideLen_;
    int secondStageNProbe_;

    std::unique_ptr<SimpleIMI> simpleIMI_;
    std::unique_ptr<HQ> index_;
};
    
} } // namespace
