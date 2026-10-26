#!/usr/bin/env bash
#SBATCH --job-name=meno-idx
#SBATCH --partition=a100
#SBATCH --exclude=ngpu01,ngpu05,ngpu06
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=5-00:00:00
#SBATCH --output=/storage2/bi/meno-idx.%j.log
#SBATCH --error=/storage2/bi/meno-idx.%j.err

# The generative service (Meno LLM) for the Meno knowledge base indexing.
# The job is not pinned to a node: it starts on the first free node with the
# modern stack (CUDA 12.6 + actual_env). The nodes ngpu01, ngpu05 and ngpu06
# are excluded because they only have the legacy CUDA 11.7 + vllm_env stack,
# whose vLLM 0.7.3 engine died on guided decoding during indexing (2026-10-21).
# Submit from the head node: sbatch submit_meno_service.sh

. "/storage2/bi/miniconda/etc/profile.d/conda.sh"
conda activate /storage2/bi/miniconda/envs/actual_env
export HOME=/storage2/bi
export PIP_CACHE_DIR=/storage2/bi/.cache/pip
export HF_HOME=/storage2/bi/huggingface_cached
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_DIR="/usr/local/cuda-12.6"
[ -d "${CUDA_DIR}" ] || CUDA_DIR="/usr/local/cuda-11.7"
export PATH="${CUDA_DIR}/bin${PATH:+:${PATH}}"
export LD_LIBRARY_PATH="${CUDA_DIR}/lib64:/storage2/bi/miniconda/envs/actual_env/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MKL_THREADING_LAYER=GNU
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1

vllm serve /storage2/bi/models/meno-lite-0.2 \
  --served-model-name bond005/meno-lite-0.2 \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 16384 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.90 \
  --dtype bfloat16
