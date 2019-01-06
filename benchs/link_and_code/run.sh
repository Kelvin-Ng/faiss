#!/bin/bash

python2 bench_link_and_code.py \
   --db sift1b_11m \
   --M0 6 \
   --indexkey OPQ32,HNSW32_PQ32 \
   --indexfile sift1b_11m_PQ32_L6.index \
   --beta_nsq 4  \
   --beta_centroids sift1b_11m_PQ32_L6_nsq4.npy \
   --neigh_recons_codes sift1b_11m_PQ32_L6_nsq4_codes.npy \
   --k_reorder 0,5 --efSearch 1,1024 \
   --maxtrain 10000000 \
   --beta_ntrain 10000000 \
   --reconstruct true \
   --residue_norms sift1b_11m_lc_36_256_residue_norms
