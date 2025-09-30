import torch
import torch.multiprocessing as mp

from src.trainer.train_chroma_moe_distill import train_chroma_moe_distill


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError("train_chroma_moe_distill requires at least one CUDA device")

    mp.spawn(
        train_chroma_moe_distill,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )
