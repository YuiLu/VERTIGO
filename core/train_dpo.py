import math
import os
import shutil
import time
from functools import partial
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import tyro
from accelerate import Accelerator
from safetensors.torch import load_file
from tqdm.auto import tqdm

from core.dpo_provider import TrajectoryDpoDataset, dpo_collate_fn
from core.models import LMM
from core.options import AllConfigs
from core.utils import init_logger


def load_checkpoint_tolerant(model: torch.nn.Module, checkpoint_path: str, logger) -> None:
    if checkpoint_path.endswith(".safetensors"):
        ckpt = load_file(checkpoint_path, device="cpu")
    else:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        ckpt = ckpt["state_dict"]

    state_dict = model.state_dict()
    loaded = 0
    skipped = 0
    for key, value in ckpt.items():
        clean_key = key.replace("module.", "", 1)
        if clean_key in state_dict and state_dict[clean_key].shape == value.shape:
            state_dict[clean_key].copy_(value)
            loaded += 1
        else:
            skipped += 1
    logger.info(f"loaded checkpoint={checkpoint_path} params={loaded} skipped={skipped}")


def concat_chosen_rejected(batch: Dict[str, Dict[str, torch.Tensor]]) -> Tuple[Dict[str, torch.Tensor], int]:
    chosen = batch["chosen"]
    rejected = batch["rejected"]
    concat = {}
    for key in ("tokens", "labels", "masks", "num_tokens", "rgb", "depth"):
        concat[key] = torch.cat([chosen[key], rejected[key]], dim=0)
    concat["text"] = chosen["text"] + rejected["text"]
    return concat, chosen["tokens"].shape[0]


def sequence_logps(model: LMM, data: Dict[str, torch.Tensor], reduction: str) -> torch.Tensor:
    logits = model(data, compute_loss=False)["logits"]

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = data["labels"][:, 1:].contiguous()
    loss_mask = shift_labels.ne(-100)
    safe_labels = shift_labels.masked_fill(~loss_mask, 0)
    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    token_logps = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps * loss_mask.float()
    seq_logps = token_logps.sum(dim=-1)

    if reduction == "mean":
        denom = loss_mask.float().sum(dim=-1).clamp_min(1.0)
        seq_logps = seq_logps / denom
    elif reduction != "sum":
        raise ValueError(f"Unknown dpo_logprob_reduction: {reduction}")
    return seq_logps


def dpo_loss(
    policy_logps: Tuple[torch.Tensor, torch.Tensor],
    reference_logps: Tuple[torch.Tensor, torch.Tensor],
    beta: float,
    sft_weight: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    policy_chosen, policy_rejected = policy_logps
    ref_chosen, ref_rejected = reference_logps

    policy_logratios = policy_chosen - policy_rejected
    ref_logratios = ref_chosen - ref_rejected
    preference_logits = policy_logratios - ref_logratios
    losses = -F.logsigmoid(float(beta) * preference_logits)

    loss = losses.mean()
    sft_nll = -policy_chosen.mean()
    if sft_weight > 0:
        loss = loss + float(sft_weight) * sft_nll

    chosen_rewards = float(beta) * (policy_chosen - ref_chosen).detach()
    rejected_rewards = float(beta) * (policy_rejected - ref_rejected).detach()
    metrics = {
        "loss_dpo": losses.mean().detach(),
        "loss_sft": sft_nll.detach(),
        "reward_chosen": chosen_rewards.mean(),
        "reward_rejected": rejected_rewards.mean(),
        "reward_margin": (chosen_rewards - rejected_rewards).mean(),
        "preference_accuracy": (chosen_rewards > rejected_rewards).float().mean(),
        "policy_chosen_logp": policy_chosen.detach().mean(),
        "policy_rejected_logp": policy_rejected.detach().mean(),
    }
    return loss, metrics


def main():
    opt = tyro.cli(AllConfigs)
    if opt.resume is None:
        raise ValueError("DPO post-training requires --resume to initialize the policy checkpoint.")
    reference_path = opt.dpo_reference or opt.resume

    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
    )

    save_root = os.path.join(opt.workspace, opt.exp_name)
    os.makedirs(save_root, exist_ok=True)
    logger = init_logger(os.path.join(save_root, "dpo_log.txt"))
    accelerator.print(opt)

    policy = LMM(opt)
    reference = LMM(opt)
    load_checkpoint_tolerant(policy, opt.resume, logger)
    load_checkpoint_tolerant(reference, reference_path, logger)
    reference.eval()
    reference.requires_grad_(False)

    num_p = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logger.info(f"DPO trainable param num: {num_p / 1024 / 1024:.6f} M")

    dataset = TrajectoryDpoDataset(opt)
    logger.info(f"DPO preference pair count: {len(dataset)}")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(dpo_collate_fn, opt=opt),
    )

    optimizer = torch.optim.AdamW(policy.parameters(), lr=opt.lr, weight_decay=0.01, betas=(0.9, 0.95))
    total_steps = max(1, opt.num_epochs * len(dataloader) // opt.gradient_accumulation_steps)

    def lr_lambda(current_step, num_cycles=0.5, min_ratio=0.1):
        progress = current_step / total_steps
        if opt.warmup_ratio > 0 and progress < opt.warmup_ratio:
            return progress / opt.warmup_ratio
        progress = (progress - opt.warmup_ratio) / max(1e-8, 1 - opt.warmup_ratio)
        return max(min_ratio, min_ratio + (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    policy, reference, optimizer, dataloader, scheduler = accelerator.prepare(
        policy, reference, optimizer, dataloader, scheduler
    )

    wandb = None
    if opt.use_wandb and accelerator.is_main_process:
        import wandb as wandb_lib

        wandb_project = os.environ.get("WANDB_PROJECT", "vertigo")
        wandb_name = os.environ.get("WANDB_NAME", opt.exp_name)
        wandb = wandb_lib.init(project=wandb_project, name=wandb_name, config=vars(opt))

    best_loss = float("inf")
    global_step = 0
    for epoch in range(opt.start_epoch, opt.num_epochs):
        policy.train()
        epoch_loss = 0.0
        epoch_metrics = {}
        start_time = time.time()

        data_iter = enumerate(dataloader)
        pbar = None
        if accelerator.is_main_process:
            pbar = tqdm(data_iter, total=len(dataloader), desc=f"dpo e{epoch:03d}", dynamic_ncols=True)
            data_iter = pbar

        for step, batch in data_iter:
            with accelerator.accumulate(policy):
                concat_batch, batch_size = concat_chosen_rejected(batch)
                policy_logps_all = sequence_logps(policy, concat_batch, reduction=opt.dpo_logprob_reduction)
                policy_chosen, policy_rejected = policy_logps_all[:batch_size], policy_logps_all[batch_size:]

                with torch.no_grad():
                    ref_logps_all = sequence_logps(reference, concat_batch, reduction=opt.dpo_logprob_reduction)
                    ref_chosen, ref_rejected = ref_logps_all[:batch_size], ref_logps_all[batch_size:]

                loss, metrics = dpo_loss(
                    (policy_chosen, policy_rejected),
                    (ref_chosen, ref_rejected),
                    beta=opt.dpo_beta,
                    sft_weight=opt.dpo_sft_weight,
                )

                optimizer.zero_grad()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(policy.parameters(), opt.gradient_clip)
                optimizer.step()
                scheduler.step()

            gathered_loss = accelerator.gather_for_metrics(loss.detach()).mean()
            epoch_loss += float(gathered_loss.item())
            for key, value in metrics.items():
                gathered_value = accelerator.gather_for_metrics(value.detach()).mean()
                epoch_metrics[key] = epoch_metrics.get(key, 0.0) + float(gathered_value.item())

            if accelerator.is_main_process and opt.use_wandb:
                log_dict = {f"dpo/{key}": float(value.detach().float().mean().item()) for key, value in metrics.items()}
                log_dict["dpo/loss"] = float(loss.detach().float().mean().item())
                log_dict["dpo/lr"] = float(scheduler.get_last_lr()[0])
                wandb.log(log_dict, step=global_step)
            global_step += 1

        if pbar is not None:
            pbar.close()

        steps = max(1, len(dataloader))
        epoch_loss /= steps
        epoch_metrics = {key: value / steps for key, value in epoch_metrics.items()}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (time.time() - start_time) / 60.0

        if accelerator.is_main_process:
            metric_text = " ".join([f"{key}={value:.4f}" for key, value in sorted(epoch_metrics.items())])
            logger.info(f"dpo epoch done: epoch={epoch} loss={epoch_loss:.6f} time_min={elapsed:.2f} {metric_text}")
            if opt.use_wandb:
                wandb.log(
                    {
                        "dpo/loss_epoch": epoch_loss,
                        "dpo/epoch_time_min": elapsed,
                        **{f"dpo/{key}_epoch": value for key, value in epoch_metrics.items()},
                    },
                    step=global_step,
                )

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                accelerator.save_model(policy, save_root)
                shutil.copy(os.path.join(save_root, "model.safetensors"), os.path.join(save_root, "best.safetensors"))

        if epoch % opt.save_epoch == 0 or epoch == opt.num_epochs - 1:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                save_dir = os.path.join(save_root, f"dpo_ep{epoch:04d}")
                accelerator.save_model(policy, save_dir)

    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
