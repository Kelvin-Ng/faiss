#include "SimpleIMICut.cuh"
#include "SimpleIMISelect.cuh"


namespace faiss { namespace gpu {


void runSimpleIMIcut(const Tensor<float, 3, true>& imiDistances,
						   Tensor<int, 2, true>& imiUpperBounds,
						   int squareLen;
						   int totalLen;
						   cudaStream_t stream) {


		runSimpleIMISelect(imiDistances,
						   imiUpperBounds,
						   squareLen,
						   totalLen,
						   cudaStream_t stream);
	}

#ifdef FAISS_USE_FLOAT16
void runSimpleIMIcut(const Tensor<half, 3, true>& imiDistances,
					 Tensor<int, 2, true>& imiUpperBounds,
					 int squareLen;
					 int totalLen;
					 cudaStream_t stream) {

	runSimpleIMISelect(imiDistances,
					   imiUpperBuounds,
					   squareLen,
					   totalLen,
					   cudaStream_t stream);
}

#endif

} } //namespace
