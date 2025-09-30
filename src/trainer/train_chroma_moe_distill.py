"""Distillation script that trains timestep MoE experts from a dense Chroma teacher.

This entrypoint mirrors the standard ``train_chroma`` pipeline but swaps the loss for
an MSE between a frozen teacher (with the original dense image MLP) and the student
MoE blocks. Only the selected MoE parameters receive gradients.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import os
from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataloaders.dataloader import TextImageDataset
from src.general_utils import load_file_multipart, load_safetensors
from src.models.chroma.model import Chroma, chroma_params
from src.models.chroma.module.autoencoder import AutoEncoder, ae_params
from src.models.chroma.module.t5 import T5Config, T5EncoderModel, replace_keys
from src.trainer.train_chroma import (
    DataloaderConfig,
    ModelConfig,
    TrainingConfig,
    dump_dict_to_json,
    init_optimizer,
    load_config_from_json,
    optimizer_state_to,
    prepare_sot_pairings,
    setup_distributed,
    synchronize_gradients,
)
from transformers import T5Tokenizer


@dataclass
class DistillationConfig:
    """Additional configuration specific to teacher-student distillation."""

    teacher_path: str
    block_indices: list[int]


def _build_dense_img_mlp(hidden_size: int, mlp_ratio: float) -> nn.Sequential:
    mlp_hidden_dim = int(hidden_size * mlp_ratio)
    return nn.Sequential(
        nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
        nn.GELU(approximate="tanh"),
        nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
    )


def _materialize_model(distill: DistillationConfig) -> tuple[Chroma, Chroma]:
    """Create student/teacher pairs with appropriate image MLPs."""

    chroma_params._use_compiled = True
    with torch.device("meta"):
        student = Chroma(chroma_params)
    with torch.device("meta"):
        teacher = Chroma(chroma_params)

    mlp_ratio = chroma_params.mlp_ratio
    for block_idx in distill.block_indices:
        if block_idx < 0 or block_idx >= len(teacher.double_blocks):
            raise ValueError(f"Invalid block index {block_idx} for teacher replacement")
        teacher.double_blocks[block_idx].img_mlp = _build_dense_img_mlp(
            teacher.hidden_size, mlp_ratio
        )
        teacher.double_blocks[block_idx]._img_mlp_requires_timesteps = False

    teacher_state = load_safetensors(distill.teacher_path)
    teacher.load_state_dict(teacher_state, assign=True)

    # The student shares the same checkpoint but tolerates missing MoE tensors.
    student.load_state_dict(teacher_state, assign=True, strict=False)

    teacher.eval()
    teacher.requires_grad_(False)

    return student, teacher


def _load_vae_and_t5(model_config: ModelConfig):
    with torch.device("meta"):
        ae = AutoEncoder(ae_params)
    ae.load_state_dict(load_safetensors(model_config.vae_path), assign=True)
    ae.to(torch.bfloat16)

    t5_tokenizer = T5Tokenizer.from_pretrained(model_config.t5_tokenizer_path)
    t5_config = T5Config.from_json_file(model_config.t5_config_path)
    with torch.device("meta"):
        t5 = T5EncoderModel(t5_config)
    t5.load_state_dict(
        replace_keys(load_file_multipart(model_config.t5_path)), assign=True
    )
    t5.eval()
    t5.to(torch.bfloat16)

    return ae, t5, t5_tokenizer


def _loss_keywords_from_blocks(block_indices: Iterable[int]) -> list[str]:
    return [f"double_blocks.{idx}.img_mlp" for idx in block_indices]


def train_chroma_moe_distill(rank: int, world_size: int, debug: bool = False) -> None:
    if not debug:
        setup_distributed(rank, world_size)

    config = load_config_from_json("training_config_moe_distill.json")

    training_config = TrainingConfig(**config["training"])
    dataloader_config = DataloaderConfig(**config["dataloader"])
    model_config = ModelConfig(**config["model"])
    distill_config = DistillationConfig(**config["distillation"])

    os.makedirs(training_config.save_folder, exist_ok=True)
    dump_dict_to_json(config, f"{training_config.save_folder}/training_config_moe_distill.json")

    torch.manual_seed(training_config.master_seed)
    torch.cuda.manual_seed_all(training_config.master_seed)

    student, teacher = _materialize_model(distill_config)
    ae, t5, t5_tokenizer = _load_vae_and_t5(model_config)

    dataset = TextImageDataset(
        batch_size=dataloader_config.batch_size,
        jsonl_path=dataloader_config.jsonl_metadata_path,
        image_folder_path=dataloader_config.image_folder_path,
        base_res=dataloader_config.base_resolution,
        shuffle_tags=dataloader_config.shuffle_tags,
        tag_drop_percentage=dataloader_config.tag_drop_percentage,
        uncond_percentage=dataloader_config.uncond_percentage,
        resolution_step=dataloader_config.resolution_step,
        seed=training_config.master_seed,
        rank=rank,
        num_gpus=world_size,
        ratio_cutoff=dataloader_config.ratio_cutoff,
        thread_per_worker=dataloader_config.thread_per_worker,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,  # dataset handles global batch logic internally
        shuffle=False,
        num_workers=dataloader_config.num_workers,
        prefetch_factor=dataloader_config.prefetch_factor,
        pin_memory=True,
        collate_fn=dataset.dummy_collate_fn,
    )

    trained_layer_keywords = _loss_keywords_from_blocks(distill_config.block_indices)
    optimizer, scheduler, hooks, _ = init_optimizer(
        student, trained_layer_keywords, training_config.lr, training_config.weight_decay, training_config.warmup_steps
    )

    global_step = 0

    for data in dataloader:
        images, caption, _, loss_weighting = data[0]
        caption = [x if x is not None else "" for x in caption]
        loss_weighting = torch.tensor(loss_weighting, device=rank)

        student.to("cpu")
        teacher.to("cpu")
        ae.to(rank)
        t5.to(rank)

        acc_latents = []
        acc_embeddings = []
        acc_mask = []

        for mb_idx in tqdm(
            range(
                dataloader_config.batch_size
                // training_config.cache_minibatch
                // max(world_size, 1)
            ),
            desc=f"prepare latents rank {rank}",
            position=rank,
        ):
            start = mb_idx * training_config.cache_minibatch
            end = start + training_config.cache_minibatch
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                text_inputs = t5_tokenizer(
                    caption[start:end],
                    padding="max_length",
                    max_length=model_config.t5_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(rank)

                t5_embed = t5(text_inputs.input_ids, text_inputs.attention_mask).to("cpu", non_blocking=True)
                acc_embeddings.append(t5_embed)
                acc_mask.append(text_inputs.attention_mask.to("cpu", non_blocking=True))

                latents = ae.encode_for_train(
                    images[start:end].to(rank)
                ).to("cpu", non_blocking=True)
                acc_latents.append(latents)

        t5.to("cpu")
        ae.to("cpu")
        torch.cuda.empty_cache()

        if not debug:
            dist.barrier()

        student.to(rank)
        teacher.to(rank)

        acc_latents = torch.cat(acc_latents, dim=0)
        acc_embeddings = torch.cat(acc_embeddings, dim=0)
        acc_mask = torch.cat(acc_mask, dim=0)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            noisy_latents, _, input_timestep, image_pos_id, _ = prepare_sot_pairings(
                acc_latents.to(rank)
            )
            noisy_latents = noisy_latents.to(torch.bfloat16)
            input_timestep = input_timestep.to(torch.bfloat16)
            image_pos_id = image_pos_id.to(rank)

            text_ids = torch.zeros((noisy_latents.shape[0], 512, 3), device=rank)
            static_guidance = torch.zeros((acc_latents.shape[0],), device=rank)

        noisy_latents.requires_grad_(True)
        acc_embeddings.requires_grad_(True)

        mb = training_config.train_minibatch
        loss_log = []

        for micro_idx in tqdm(
            range(dataloader_config.batch_size // mb // max(world_size, 1)),
            desc=f"distill rank {rank}",
            position=rank,
        ):
            mb_start = micro_idx * mb
            mb_end = mb_start + mb

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                student_pred = student(
                    img=noisy_latents[mb_start:mb_end].to(rank, non_blocking=True),
                    img_ids=image_pos_id[mb_start:mb_end].to(rank, non_blocking=True),
                    txt=acc_embeddings[mb_start:mb_end].to(rank, non_blocking=True),
                    txt_ids=text_ids[mb_start:mb_end].to(rank, non_blocking=True),
                    txt_mask=acc_mask[mb_start:mb_end].to(rank, non_blocking=True),
                    timesteps=input_timestep[mb_start:mb_end].to(rank, non_blocking=True),
                    guidance=static_guidance[mb_start:mb_end].to(rank, non_blocking=True),
                )

                with torch.no_grad():
                    teacher_pred = teacher(
                        img=noisy_latents[mb_start:mb_end].to(rank, non_blocking=True),
                        img_ids=image_pos_id[mb_start:mb_end].to(rank, non_blocking=True),
                        txt=acc_embeddings[mb_start:mb_end].to(rank, non_blocking=True),
                        txt_ids=text_ids[mb_start:mb_end].to(rank, non_blocking=True),
                        txt_mask=acc_mask[mb_start:mb_end].to(rank, non_blocking=True),
                        timesteps=input_timestep[mb_start:mb_end].to(rank, non_blocking=True),
                        guidance=static_guidance[mb_start:mb_end].to(rank, non_blocking=True),
                    )

                per_sample = (
                    (student_pred.float() - teacher_pred.float()) ** 2
                ).mean(dim=(1, 2))
                per_sample = per_sample / (dataloader_config.batch_size // mb)
                weights = loss_weighting[mb_start:mb_end].to(per_sample.dtype)
                weights = weights / weights.sum()
                loss = (per_sample * weights).sum()

            loss.backward()
            loss_log.append(loss.detach().clone() * (dataloader_config.batch_size // mb))

        loss_log = sum(loss_log) / len(loss_log)

        del acc_embeddings, noisy_latents, acc_latents
        torch.cuda.empty_cache()

        for name, param in student.named_parameters():
            if not any(keyword in name for keyword in trained_layer_keywords):
                param.data = param.data.to("cpu", non_blocking=True)

        optimizer_state_to(optimizer, rank)
        if not debug:
            synchronize_gradients(student)

        scheduler.step()
        optimizer.step()
        optimizer.zero_grad()

        optimizer_state_to(optimizer, "cpu")
        torch.cuda.empty_cache()

        student.to("cpu")
        teacher.to("cpu")

        if (global_step + 1) % training_config.save_every == 0 and rank == 0:
            model_filename = os.path.join(
                training_config.save_folder,
                f"moe_distill_{global_step + 1}.pth",
            )
            torch.save({k: v.cpu() for k, v in student.state_dict().items()}, model_filename)

        if not debug:
            dist.barrier()

        global_step += 1

    if rank == 0:
        model_filename = os.path.join(
            training_config.save_folder,
            f"moe_distill_final_{global_step}.pth",
        )
        torch.save({k: v.cpu() for k, v in student.state_dict().items()}, model_filename)

    if not debug:
        dist.destroy_process_group()


def save_distill_config(path: str, **configs) -> None:
    json_data = {key: asdict(value) for key, value in configs.items()}
    with open(path, "w") as fh:
        json.dump(json_data, fh, indent=4)

