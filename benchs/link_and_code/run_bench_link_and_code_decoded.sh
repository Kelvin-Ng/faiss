dataset="$1"
method="$2"
num_codebooks="$3"

python2 bench_link_and_code_decoded.py \
   --db "$dataset" \
   --M0 6 \
   --indexkey HNSW128 \
   --indexfile ${dataset}_${method}${num_codebooks}_L6.index \
   --beta_nsq 4  \
   --beta_centroids ${dataset}_${method}${num_codebooks}_L6_nsq4.npy \
   --neigh_recons_codes ${dataset}_${method}${num_codebooks}_L6_nsq4_codes.npy \
   --k_reorder 0,5 --efSearch 1,1024 \
   --maxtrain 10000000 \
   --beta_ntrain 250000 \
   --reconstruct true \
   --residue_norms /data/kelvin/aq_research/product-quantization/${dataset}_lc_${method}_${num_codebooks}_256_residue_norms \
   --decoded /data/kelvin/aq_research/product-quantization/${dataset}_${method}_${num_codebooks}_256_decoded
