"""Sensor drivers: Integrated Gradients against a healthy-engine reference window.

Attributions come out in cycles and roughly sum to
``RUL(this engine) - RUL(average healthy engine)``: "T50: -18.2 cycles" means the
T50 deviation from healthy drags the estimate down by about 18 cycles.
A deletion test backs the explanation: resetting the top sensors to healthy
values should move the prediction much more than resetting random ones.
"""
from __future__ import annotations

import numpy as np
import torch

from .constants import sensor_description, sensor_name


def integrated_gradients(predictor, subset: str, x: np.ndarray, baseline: np.ndarray, steps: int = 32):
    """(attributions (L, F) in cycles, f(x), f(baseline))."""
    device = predictor.device
    x_t = torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)
    b_t = torch.as_tensor(np.asarray(baseline, dtype=np.float32), device=device)
    alphas = (torch.arange(steps, device=device, dtype=torch.float32) + 0.5) / steps   # midpoint rule
    path = (b_t.unsqueeze(0) + alphas.view(-1, 1, 1) * (x_t - b_t).unsqueeze(0)).detach()
    path.requires_grad_(True)
    with torch.enable_grad():
        out = predictor.mean_torch(subset, path)
        grads = torch.autograd.grad(out.sum(), path)[0]
    attr = ((x_t - b_t) * grads.mean(dim=0)).detach().cpu().numpy()
    with torch.no_grad():
        both = torch.stack([x_t, b_t])
        f = predictor.mean_torch(subset, both).detach().cpu().numpy()
    return attr, float(f[0]), float(f[1])


def sensor_contributions(attr: np.ndarray, sensors: list[str], top_k: int | None = None) -> list[dict]:
    per_sensor = attr.sum(axis=0)
    order = np.argsort(-np.abs(per_sensor))
    rows = [{
        "sensor": sensors[j],
        "name": sensor_name(sensors[j]),
        "description": sensor_description(sensors[j]),
        "contribution_cycles": round(float(per_sensor[j]), 1),
        "effect": "lowers the RUL estimate" if per_sensor[j] < 0 else "raises the RUL estimate",
    } for j in order]
    return rows[:top_k] if top_k else rows


def deletion_test(predictor, subset: str, x: np.ndarray, baseline: np.ndarray, per_sensor: np.ndarray,
                  top_k: int = 3, trials: int = 20, seed: int = 0) -> dict:
    n_feat = x.shape[1]
    top_k = int(min(top_k, n_feat))
    order = np.argsort(-np.abs(per_sensor))
    rng = np.random.default_rng(seed)

    def reset(cols) -> np.ndarray:
        xx = x.copy()
        xx[:, cols] = baseline[:, cols]
        return xx

    batch = [x, reset(order[:top_k])] + [reset(rng.choice(n_feat, size=top_k, replace=False)) for _ in range(trials)]
    preds = predictor.predict(subset, np.stack(batch).astype(np.float32))["mu"]
    top_change = abs(float(preds[1] - preds[0]))
    rand_change = float(np.mean(np.abs(preds[2:] - preds[0]))) if trials else 0.0
    return {
        "top_k": top_k,
        "top_k_change_cycles": round(top_change, 1),
        "random_k_change_cycles": round(rand_change, 1),
        "faithfulness_ratio": round(top_change / max(rand_change, 1e-6), 2),
    }
