# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD+Patents license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import sys

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def bvecs_read(fname):
    a = np.fromfile(fname, dtype='uint8')
    d = a[:4].view('uint8')[0]
    return a.reshape(-1, d + 4)[:, 4:].copy()

def sanitize(x):
    """ convert array to a c-contiguous float array """
    return np.ascontiguousarray(x.astype(np.float32))

def true_positives(topK, G, T):
    result = np.empty(shape=(len(G)))
    for i in range(len(G)):
        result[i] = len(np.intersect1d(G[i], topK[i][:T]))
    return result


loaded = np.load(sys.argv[1])
if sys.argv[5] == 'fvecs':
    xq = fvecs_read(sys.argv[2])
else:
    xq = sanitize(bvecs_read(sys.argv[2]))
gt = ivecs_read(sys.argv[3])
gtd = fvecs_read(sys.argv[4])

d = loaded['codewords2'].shape[3]  # dimension
nb = loaded['listCodes1Data'].shape[0]# database size
nq = len(xq)                       # nb of queries
imiSize = loaded['centroids'].shape[1]
numCodes2 = loaded['codewords2'].shape[0] * 2
np.random.seed(1234)             # make reproducible

import faiss                     # make faiss available

res = faiss.StandardGpuResources()  # use a single GPU

codewords2 = sanitize(loaded['codewords2'])
fineCentroids = sanitize(loaded['fineCentroids'])
print(fineCentroids.flags['C_CONTIGUOUS'])
print(fineCentroids[1, 1055, 77])

index = faiss.GpuIndexHQ(res, d, faiss.METRIC_L2, imiSize, nb, numCodes2, 100, 250, 1000, loaded['centroids'], loaded['fineCentroids'], codewords2, loaded['listCodes1Data'], loaded['listCodes2Data'], loaded['listIndicesData'].astype(np.int64), loaded['listLengths'].astype(np.int32), loaded['rotate'])

k = 4                          # we want to see 4 nearest neighbors
D, I = index.search(xq[:10], k)  # actual search

print(D)
print(gtd[:10, :k])
print(I)
print(gt[:10, :k])

print(true_positives(I, gt[:10, :k], k))
