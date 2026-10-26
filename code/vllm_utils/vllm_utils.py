import gc

import torch
from vllm import LLMEngine
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment


def finalize_vllm():
    if hasattr(LLMEngine, 'shutdown'):
        LLMEngine.shutdown()
    destroy_model_parallel()
    destroy_distributed_environment()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()
