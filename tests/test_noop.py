"""The probe leaves no trace: with a non-binding cap, training is bit-identical.

    python -m pytest tests
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from armijo_probe import armijo_probe, cap_learning_rate  # noqa: E402


def make_model():
    # BatchNorm (buffers) and Dropout (RNG) are what the probe must not disturb.
    return nn.Sequential(nn.Linear(20, 64), nn.BatchNorm1d(64), nn.ReLU(),
                         nn.Dropout(0.2), nn.Linear(64, 5))


def state(model):
    return [t.detach().clone() for t in list(model.parameters()) + list(model.buffers())]


def train(use_probe, lr, steps=20):
    torch.manual_seed(0)
    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    data = torch.Generator().manual_seed(1)
    for t in range(steps):
        X = torch.randn(32, 20, generator=data)
        Y = torch.randint(0, 5, (32,), generator=data)
        if use_probe and t == 0:
            cap_learning_rate(opt, model, lambda: crit(model(X), Y), kappa=2.0)
        opt.zero_grad()
        crit(model(X), Y).backward()
        opt.step()
    return state(model), opt.param_groups[0]["lr"]


def test_noop_when_cap_does_not_bind():
    s0, lr0 = train(False, 1e-4)
    s1, lr1 = train(True, 1e-4)
    assert lr0 == lr1 == 1e-4
    assert all(torch.equal(a, b) for a, b in zip(s0, s1))


def test_probe_restores_parameters_buffers_and_rng():
    torch.manual_seed(0)
    model = make_model()
    X, Y = torch.randn(32, 20), torch.randint(0, 5, (32,))
    before = state(model)
    rng = torch.get_rng_state()
    armijo_probe(model, lambda: nn.functional.cross_entropy(model(X), Y))
    assert all(torch.equal(a, b) for a, b in zip(before, state(model)))
    assert torch.equal(rng, torch.get_rng_state())
    assert all(p.grad is None for p in model.parameters())


def test_cap_binds_for_large_lr():
    torch.manual_seed(0)
    model = make_model()
    X, Y = torch.randn(32, 20), torch.randint(0, 5, (32,))
    opt = torch.optim.AdamW(model.parameters(), lr=10.0)
    res = cap_learning_rate(opt, model, lambda: nn.functional.cross_entropy(model(X), Y))
    assert opt.param_groups[0]["lr"] == 2.0 * res.alpha < 10.0
