"""Armijo init probe: a learning-rate safety cap for Adam and AdamW.

Reference implementation of the direction-matched init probe from
"Directional Curvature from Armijo Backtracking" (Ashmitha R and Joerg Frochte,
TMLR 2026). Before the first optimiser step, one Armijo backtracking line search
is run along Adam's first-step direction d = -g / (|g| + eps) on a single
mini-batch. The accepted step alpha measures the directional curvature at
initialisation, and the learning rate is capped at kappa * alpha (kappa = 2).

The probe leaves no trace: parameters, non-parameter buffers (e.g. BatchNorm
running statistics) and the torch RNG state are restored afterwards, so when the
cap does not bind, training is bit-identical to training without the probe.

Typical use::

    opt = torch.optim.AdamW(model.parameters(), lr=0.1)
    X, Y = next(iter(train_loader))
    cap_learning_rate(opt, model, lambda: criterion(model(X), Y))
    # ... train as usual

Optionally, ProbationWindow re-runs the probe during the first 50 steps and can
only lower the cap further (needed where curvature rises within the first few
steps, e.g. transformer language-model pretraining).
"""

import contextlib
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import torch

__all__ = ["ProbeResult", "armijo_probe", "cap_learning_rate", "ProbationWindow"]

PROBATION_STEPS = (1, 2, 3, 5, 8, 12, 20, 35, 50)


@dataclass
class ProbeResult:
    alpha: float       # accepted (or, if saturated, last untested) step size
    accepted: bool     # False: no candidate satisfied the Armijo condition
    backtracks: int    # number of halvings performed
    loss: float        # probe-batch loss at the current parameters


@contextlib.contextmanager
def _state_guard(model: torch.nn.Module):
    """Restore non-parameter buffers and the RNG state (CPU and CUDA) on exit."""
    bufs = [(b, b.detach().clone()) for b in model.buffers()]
    devs = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devs):
            yield
    finally:
        with torch.no_grad():
            for b, b0 in bufs:
                b.copy_(b0)


def armijo_probe(model: torch.nn.Module, loss_fn: Callable[[], torch.Tensor],
                 direction: str = "adam", c: float = 1e-4, beta: float = 0.5,
                 alpha_max: float = 1.0, max_backtracks: int = 30,
                 eps: float = 1e-8) -> ProbeResult:
    """Largest alpha in {alpha_max * beta^k} with L(theta + alpha d) <= L + c alpha g^T d.

    loss_fn: evaluates the loss of `model` on one fixed mini-batch at the
        current parameters (forward pass only, returns a scalar tensor).
    direction: "adam" for Adam's first-step direction -g / (|g| + eps)
        (recommended, use with kappa = 2), "grad" for the raw gradient -g
        (kappa is then architecture-specific).
    max_backtracks: if no candidate is accepted after this many halvings,
        the next untested candidate alpha_max * beta^max_backtracks is returned
        with accepted=False.
    """
    if direction not in ("adam", "grad"):
        raise ValueError("direction must be 'adam' or 'grad'")
    params = [p for p in model.parameters() if p.requires_grad]
    with _state_guard(model):
        with torch.enable_grad():
            loss0 = loss_fn()
            grads = torch.autograd.grad(loss0, params, allow_unused=True)
        grads = [torch.zeros_like(p) if g is None else g for p, g in zip(params, grads)]
        L0 = float(loss0.detach())
        if direction == "adam":
            d = [-g / (g.abs() + eps) for g in grads]
        else:
            d = [-g for g in grads]
        gTd = float(sum((g * di).sum() for g, di in zip(grads, d)))
        x0 = [p.detach().clone() for p in params]

        def trial_loss(a):
            with torch.no_grad():
                for p, x, di in zip(params, x0, d):
                    p.copy_(x + a * di)
                return float(loss_fn().detach())

        alpha, backtracks, accepted = alpha_max, 0, False
        try:
            for _ in range(max_backtracks):
                if trial_loss(alpha) <= L0 + c * alpha * gTd:
                    accepted = True
                    break
                alpha *= beta
                backtracks += 1
        finally:
            with torch.no_grad():
                for p, x in zip(params, x0):
                    p.copy_(x)
    return ProbeResult(alpha=alpha, accepted=accepted, backtracks=backtracks, loss=L0)


def cap_learning_rate(optimizer: torch.optim.Optimizer, model: torch.nn.Module,
                      loss_fn: Callable[[], torch.Tensor], kappa: float = 2.0,
                      **probe_kwargs) -> ProbeResult:
    """Probe once and lower every param group's lr to kappa * alpha if it is larger.

    Call before the first optimiser step. Returns the ProbeResult; the
    resulting learning rate is in optimizer.param_groups[i]["lr"].
    """
    res = armijo_probe(model, loss_fn, **probe_kwargs)
    cap = kappa * res.alpha
    for group in optimizer.param_groups:
        if group["lr"] > cap:
            group["lr"] = cap
    return res


class ProbationWindow:
    """Re-probe at fixed early steps and tighten the cap monotonically.

    After optimiser step t (counting from 1), call `window.after_step(t)`.
    At t in `steps` the probe is re-run (at depth `max_backtracks`, default 8)
    and the learning rate becomes min(lr, kappa * min alpha seen so far).
    After the last step of the window nothing happens any more.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, model: torch.nn.Module,
                 loss_fn: Callable[[], torch.Tensor], kappa: float = 2.0,
                 steps: Iterable[int] = PROBATION_STEPS, max_backtracks: int = 8,
                 init_alpha: Optional[float] = None, **probe_kwargs):
        self.optimizer, self.model, self.loss_fn = optimizer, model, loss_fn
        self.kappa = kappa
        self.steps = set(steps)
        self.probe_kwargs = dict(probe_kwargs, max_backtracks=max_backtracks)
        self.alpha_min = float("inf") if init_alpha is None else init_alpha

    def after_step(self, step: int) -> Optional[ProbeResult]:
        if step not in self.steps:
            return None
        res = armijo_probe(self.model, self.loss_fn, **self.probe_kwargs)
        self.alpha_min = min(self.alpha_min, res.alpha)
        cap = self.kappa * self.alpha_min
        for group in self.optimizer.param_groups:
            if group["lr"] > cap:
                group["lr"] = cap
        return res
