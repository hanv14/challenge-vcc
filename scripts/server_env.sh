# Source this before any run on the shared server:
#     source scripts/server_env.sh
#
# The server has 192 cores and shared GPUs. These limits keep the pipeline
# from taking more than it needs. The pipeline never overrides them: if a
# variable is already set, it is respected (CLAUDE.md §1.2).

# CPU courtesy — 192 cores, be a good neighbor
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMBA_NUM_THREADS=1

# GPU
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# polars / rayon (used by cell-eval2) otherwise start one thread per core.
# Keep these equal to resources.cpu_threads in your config.
export POLARS_MAX_THREADS=4
export RAYON_NUM_THREADS=4
