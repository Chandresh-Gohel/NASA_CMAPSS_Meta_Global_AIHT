"""Training CLI: data prep, the CNN baseline/pretraining, and the DKL head.

    python -m ehagent.train prepare                   # build per-subset data
    python -m ehagent.train baseline --mode multitask # CNN point model (also the DKL pretraining)
    python -m ehagent.train dkl --mode multitask      # deep kernel learning GP head (primary)
    python -m ehagent.train dkl --mode single         # one DKL per subset (ablation)

The ``dkl`` stage reuses a trained CNN when one is already saved.
"""
from __future__ import annotations

import argparse
import copy
import math
from datetime import datetime, timezone

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import Config, ensure_dir, get_device, load_config, set_seed
from .data import load_subset, prepare_subset, save_subset, training_windows
from .metrics import rmse
from .models.cnn import CNNRegressor, FeatureExtractor, extractor_kwargs_from_cfg
from .predictor import load_module, model_path


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def arrays_for(cfg: Config, datasets: dict, subsets: list[str], split: str):
    """Stack windows of several subsets. Targets scale to RUL / cap; task = subset index."""
    cap = float(cfg.data.rul_cap)
    xs, ys, ts = [], [], []
    for i, s in enumerate(subsets):
        X, y, _ = training_windows(datasets[s], split, int(cfg.data.window), int(cfg.data.min_history), cap)
        if len(y) == 0:
            continue
        xs.append(X)
        ys.append((y / cap).astype(np.float32))
        ts.append(np.full(len(y), i, dtype=np.int64))
    if not xs:
        raise ValueError(f"No windows for split '{split}' in {subsets}")
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.float32), np.concatenate(ts)


def make_loader(arrays, batch_size: int, shuffle: bool, cfg: Config, device: torch.device) -> DataLoader:
    X, y, t = arrays
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(t))
    return DataLoader(ds, batch_size=int(batch_size), shuffle=shuffle, num_workers=int(cfg.train.num_workers),
                      pin_memory=(device.type == "cuda"))


def save_checkpoint(path, family: str, meta: dict, module: nn.Module) -> None:
    ensure_dir(path.parent)
    torch.save({"family": family, "meta": meta, "state": module.state_dict()}, path)
    log(f"  saved {family} -> {path}")


@torch.no_grad()
def eval_point(model: nn.Module, loader: DataLoader, device: torch.device, cap: float) -> float:
    model.eval()
    preds, ys = [], []
    for xb, yb, tb in loader:
        preds.append(model(xb.to(device), tb.to(device)).float().cpu())
        ys.append(yb)
    return rmse(torch.cat(ys).numpy() * cap, torch.cat(preds).numpy() * cap)


def train_cnn(cfg: Config, tr, va, n_tasks: int, use_task_embedding: bool, device: torch.device):
    cap = float(cfg.data.rul_cap)
    ek = extractor_kwargs_from_cfg(cfg, tr[0].shape[2], n_tasks, use_task_embedding)
    model = CNNRegressor(FeatureExtractor(**ek), n_tasks).to(device)
    epochs = int(cfg.train.pretrain_epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.train.pretrain_lr), weight_decay=float(cfg.train.weight_decay))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    tl = make_loader(tr, cfg.train.batch_size, True, cfg, device)
    vl = make_loader(va, 2048, False, cfg, device)
    best, best_state, bad, history = math.inf, None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for xb, yb, tb in tl:
            xb, yb, tb = xb.to(device), yb.to(device), tb.to(device)
            loss = nn.functional.mse_loss(model(xb, tb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip))
            opt.step()
            total += float(loss.detach()) * len(yb)
            n += len(yb)
        sched.step()
        val = eval_point(model, vl, device, cap)
        history.append({"epoch": epoch, "train_mse_scaled": total / max(n, 1), "val_rmse": val})
        log(f"  [cnn] epoch {epoch:3d}  train_mse={total / max(n, 1):.4f}  val_rmse={val:.2f}")
        if val < best - 1e-3:
            best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= int(cfg.train.patience):
                log("  early stop")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, ek, {"val_rmse": best, "history": history}


def training_groups(mode: str, subsets: list[str], multitask_mode: str):
    """Yield (tag, subsets in this model, multitask mode) for the requested mode."""
    if mode == "multitask":
        yield "multitask", list(subsets), multitask_mode
    else:
        for s in subsets:
            yield s, [s], "none"


@torch.no_grad()
def embed(extractor, X: np.ndarray, t: np.ndarray, device: torch.device, batch_size: int = 2048) -> np.ndarray:
    extractor.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i : i + batch_size]).to(device)
        tb = torch.from_numpy(t[i : i + batch_size]).to(device)
        out.append(extractor(xb, tb if extractor.use_task_embedding else None).float().cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def eval_dkl(model, loader: DataLoader, device: torch.device, cap: float) -> dict:
    import gpytorch

    from .metrics import gaussian_nll

    model.eval()
    mus, vs, ys = [], [], []
    with gpytorch.settings.cholesky_jitter(1e-4):
        for xb, yb, tb in loader:
            f = model.latent(xb.to(device), tb.to(device))
            noise = model.likelihood.noise.reshape(-1)[0]
            mus.append(f.mean.float().cpu())
            vs.append((f.variance + noise).float().cpu())
            ys.append(yb)
    mu = torch.cat(mus).numpy() * cap
    sd = np.sqrt(np.maximum(torch.cat(vs).numpy(), 1e-12)) * cap
    y = torch.cat(ys).numpy() * cap
    return {"rmse": rmse(y, mu), "nll": gaussian_nll(y, mu, sd)}


def train_dkl(cfg: Config, cnn_model, tr, va, n_tasks: int, mode: str, device: torch.device):
    import gpytorch

    from .models.dkl import DKLRegressor, LMCMultiTaskGPHead, SingleTaskGPHead, init_gp_hyperparameters, init_inducing_points

    cap = float(cfg.data.rul_cap)
    seed = int(cfg.project.seed)
    extractor = cnn_model.extractor
    rng = np.random.default_rng(seed)
    n_init = min(int(cfg.train.inducing_init_samples), len(tr[0]))
    idx = rng.choice(len(tr[0]), size=n_init, replace=False)
    emb = embed(extractor, tr[0][idx], tr[2][idx], device)

    num_inducing = int(cfg.model.inducing_points)
    if mode == "lmc":
        num_latents = int(cfg.model.num_latents)
        inducing = init_inducing_points(emb, num_inducing, seed, num_latents=num_latents)
        gp = LMCMultiTaskGPHead(inducing, num_tasks=n_tasks)
    else:
        num_latents = 0
        inducing = init_inducing_points(emb, num_inducing, seed)
        gp = SingleTaskGPHead(inducing)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    init_gp_hyperparameters(gp, likelihood, emb, float(tr[1].mean()))
    model = DKLRegressor(extractor, gp, likelihood, mode).to(device)

    mll = gpytorch.mlls.VariationalELBO(likelihood, gp, num_data=len(tr[0]))
    opt = torch.optim.Adam([
        {"params": model.extractor.parameters(), "lr": float(cfg.train.dkl_lr_features), "weight_decay": float(cfg.train.weight_decay)},
        {"params": list(gp.parameters()) + list(likelihood.parameters()), "lr": float(cfg.train.dkl_lr_gp)},
    ])
    epochs = int(cfg.train.dkl_epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    tl = make_loader(tr, cfg.train.dkl_batch_size, True, cfg, device)
    vl = make_loader(va, 2048, False, cfg, device)
    metric = str(cfg.train.early_stop_metric)
    best, best_state, best_val, bad, history = math.inf, None, None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for xb, yb, tb in tl:
            xb, yb, tb = xb.to(device), yb.to(device), tb.to(device)
            with gpytorch.settings.cholesky_jitter(1e-4):
                out = model.latent(xb, tb)
                loss = -mll(out, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip))
            opt.step()
            total += float(loss.detach()) * len(yb)
            n += len(yb)
        sched.step()
        val = eval_dkl(model, vl, device, cap)
        history.append({"epoch": epoch, "train_neg_elbo": total / max(n, 1), **val})
        log(f"  [dkl] epoch {epoch:3d}  -elbo={total / max(n, 1):.4f}  val_rmse={val['rmse']:.2f}  val_nll={val['nll']:.3f}")
        score = val["rmse"] if metric == "rmse" else val["nll"]
        if score < best - 1e-4:
            best, best_state, best_val, bad = score, copy.deepcopy(model.state_dict()), val, 0
        else:
            bad += 1
            if bad >= int(cfg.train.patience):
                log("  early stop")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"val": best_val, "history": history, "num_inducing": int(inducing.shape[-2]), "num_latents": num_latents}


def stage_prepare(cfg: Config, subsets: list[str]) -> None:
    for s in subsets:
        log(f"Preparing {s} ...")
        sd = prepare_subset(cfg, s)
        path = save_subset(cfg, sd)
        counts = " / ".join(f"{len(sd.splits[k])} {k}" for k in ("train", "val", "calib", "sim", "test"))
        log(f"  {s}: {counts} engines -> {path}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train Engine Health Agent models")
    ap.add_argument("stage", choices=["prepare", "baseline", "dkl"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--mode", choices=["multitask", "single"], default=None)
    ap.add_argument("--subsets", default=None, help="comma-separated subset list (default: all in config)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(int(cfg.project.seed))
    subsets = args.subsets.split(",") if args.subsets else list(cfg.data.subsets)
    mode = args.mode or ("multitask" if cfg.features.multitask_model else "single")
    device = get_device(cfg)
    log(f"device={device}  mode={mode}  subsets={subsets}")

    if args.stage == "prepare":
        stage_prepare(cfg, subsets)
        log("done")
        return

    datasets = {s: load_subset(cfg, s) for s in subsets}
    for tag, group, mt_mode in training_groups(mode, subsets, str(cfg.model.multitask_mode)):
        n_tasks = len(group)
        use_emb = mt_mode == "embedding"
        log(f"== model '{tag}' on {group} (multitask_mode={mt_mode})")
        tr = arrays_for(cfg, datasets, group, "train")
        va = arrays_for(cfg, datasets, group, "val")
        log(f"  windows: train={len(tr[1])} val={len(va[1])}")
        base_meta = {
            "subsets": group, "num_tasks": n_tasks, "multitask_mode": mt_mode, "window": int(cfg.data.window),
            "rul_cap": float(cfg.data.rul_cap), "sensors": list(cfg.data.sensors),
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        cnn_file = model_path(cfg, "cnn", tag)
        if args.stage == "dkl" and cnn_file.exists():
            log(f"  loading pretrained CNN {cnn_file}")
            cnn, cnn_meta, _ = load_module(cnn_file, device, train_mode=True)
            ek = cnn_meta["extractor"]
        else:
            cnn, ek, info = train_cnn(cfg, tr, va, n_tasks, use_emb, device)
            save_checkpoint(model_path(cfg, "cnn", tag), "cnn", {**base_meta, "extractor": ek, "val": {"rmse": info["val_rmse"]}}, cnn)
            if args.stage == "baseline":
                continue
        if args.stage == "dkl":
            dkl_mode = "lmc" if mt_mode == "lmc" else "none"
            dkl, dinfo = train_dkl(cfg, cnn, tr, va, n_tasks if dkl_mode == "lmc" else 1, dkl_mode, device)
            save_checkpoint(model_path(cfg, "dkl", tag), "dkl", {**base_meta, "extractor": ek, "num_inducing": dinfo["num_inducing"], "num_latents": dinfo["num_latents"], "val": dinfo["val"]}, dkl)
    log("done")


if __name__ == "__main__":
    main()
