"""Deterministic PyTorch training engine: schedulers, early stopping, clipping, AMP, checkpoints, resume, NaN recovery."""

from __future__ import annotations

import os
import random
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..config import TrainConfig
from ..errors import CancelledError, ModelError, TrainingError
from .base import ProgressFn, TrainingHistory


def set_determinism(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise ModelError("device='cuda' requested but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _atomic_save(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _build_opt(params, cfg: TrainConfig, lr: float | None = None) -> torch.optim.Optimizer:
    lr = lr or cfg.lr
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=cfg.weight_decay)
    return torch.optim.SGD(params, lr=lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)


def _build_sched(opt: torch.optim.Optimizer, cfg: TrainConfig):
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    if cfg.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, cfg.epochs // 3), gamma=0.5)
    if cfg.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(1, cfg.patience // 3))
    return None


@torch.no_grad()
def predict_logits(
    module: nn.Module, X: torch.Tensor, device: torch.device, batch_size: int = 512
) -> torch.Tensor:
    module.eval()
    outs = [module(X[i : i + batch_size].to(device)).float().cpu() for i in range(0, len(X), batch_size)]
    return torch.cat(outs) if outs else torch.zeros((0, 0))


def fit_module(
    module: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None,
    y_val: np.ndarray | None,
    cfg: TrainConfig,
    seed: int,
    n_classes: int,
    *,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
    progress: ProgressFn | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> TrainingHistory:
    set_determinism(seed)
    device = resolve_device(cfg.device)
    module.to(device)
    hist = TrainingHistory(device=device.type, n_train=len(y_train))
    t0 = time.perf_counter()
    Xt = torch.as_tensor(np.ascontiguousarray(X_train, dtype=np.float32))
    yt = torch.as_tensor(np.asarray(y_train), dtype=torch.long)
    has_val = X_val is not None and y_val is not None and len(y_val) > 0
    if has_val:
        Xv = torch.as_tensor(np.ascontiguousarray(X_val, dtype=np.float32))
        yv = torch.as_tensor(np.asarray(y_val), dtype=torch.long)
    weight = None
    if cfg.class_weighting == "balanced":
        counts = torch.bincount(yt, minlength=n_classes).float().clamp(min=1)
        weight = (counts.sum() / (n_classes * counts)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    opt = _build_opt(module.parameters(), cfg)
    sched = _build_sched(opt, cfg)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best_val, bad, best_state = 0, float("inf"), 0, None
    last_ckpt = checkpoint_dir / "last.pt" if checkpoint_dir and cfg.checkpoint else None
    if resume and last_ckpt and last_ckpt.is_file():
        ck = torch.load(
            last_ckpt, map_location=device, weights_only=True
        )  # tensors/primitives only: no code execution
        module.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if sched is not None and ck.get("sched"):
            sched.load_state_dict(ck["sched"])
        start_epoch, best_val, bad = ck["epoch"], ck["best_val"], ck["bad"]
        best_state = ck.get("best_model")
        hist.epochs = ck["history"]
        hist.best_epoch = ck.get("best_epoch")
        hist.resumed_from_epoch = start_epoch
    good_state = {k: v.detach().clone() for k, v in module.state_dict().items()}
    retries = 0
    n = len(yt)

    def evaluate() -> tuple[float, float]:
        lg = predict_logits(module, Xv, device)
        return float(
            nn.functional.cross_entropy(lg, yv, weight=weight.cpu() if weight is not None else None)
        ), float((lg.argmax(1) == yv).float().mean())

    epoch = start_epoch
    while epoch < cfg.epochs:
        if should_cancel and should_cancel():
            raise CancelledError("training cancelled")
        te = time.perf_counter()
        torch.manual_seed(seed * 100003 + epoch)
        gen = torch.Generator().manual_seed(seed * 100003 + epoch)
        perm = torch.randperm(n, generator=gen)
        module.train()
        tot, correct, nan_hit = 0.0, 0, False
        for i in range(0, n, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            if len(idx) < 2:  # BatchNorm cannot train on a single sample
                continue
            xb, yb = Xt[idx].to(device), yt[idx].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=use_amp):
                out = module(xb)
                loss = loss_fn(out.float(), yb)
            if not torch.isfinite(loss):
                nan_hit = True
                break
            scaler.scale(loss).backward()
            if cfg.grad_clip:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(module.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            tot += float(loss.detach()) * len(idx)
            correct += int((out.argmax(1) == yb).sum())
        if nan_hit or not np.isfinite(tot):
            if retries >= cfg.max_nan_retries:
                raise TrainingError(
                    f"non-finite loss at epoch {epoch + 1}; recovery attempts exhausted",
                    details={"epoch": epoch + 1},
                )
            retries += 1
            hist.nan_recoveries += 1
            module.load_state_dict(
                good_state
            )  # restore last good weights, lower the learning rate, retry the epoch
            new_lr = opt.param_groups[0]["lr"] * 0.5
            opt = _build_opt(module.parameters(), cfg, new_lr)
            sched = _build_sched(opt, cfg)
            hist.notes.append(f"epoch {epoch + 1}: non-finite loss; restored weights, lr -> {new_lr:.3g}")
            continue
        rec = {
            "epoch": float(epoch + 1),
            "train_loss": tot / n,
            "train_acc": correct / n,
            "lr": opt.param_groups[0]["lr"],
        }
        if has_val:
            rec["val_loss"], rec["val_acc"] = evaluate()
            monitor = rec["val_loss"]
        else:
            monitor = rec["train_loss"]
        rec["seconds"] = time.perf_counter() - te
        hist.epochs.append(rec)
        good_state = {k: v.detach().clone() for k, v in module.state_dict().items()}
        if sched is not None:
            sched.step(monitor) if cfg.scheduler == "plateau" else sched.step()
        if monitor < best_val - 1e-4:
            best_val, bad, hist.best_epoch = monitor, 0, epoch + 1
            best_state = {k: v.detach().clone().cpu() for k, v in module.state_dict().items()}
        else:
            bad += 1
        epoch += 1
        if last_ckpt:
            _atomic_save(
                {
                    "model": module.state_dict(),
                    "opt": opt.state_dict(),
                    "sched": sched.state_dict() if sched else None,
                    "epoch": epoch,
                    "best_val": best_val,
                    "bad": bad,
                    "history": hist.epochs,
                    "best_epoch": hist.best_epoch,
                    "best_model": best_state,
                },
                last_ckpt,
            )
        if progress:
            progress(epoch, cfg.epochs, rec)
        if has_val and bad >= cfg.patience:
            hist.stopped_early = True
            break
    if best_state is not None:
        module.load_state_dict(best_state)
    hist.seconds = time.perf_counter() - t0
    return hist
