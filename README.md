# Armijo init probe for Adam

A learning-rate safety cap for Adam and AdamW from a single Armijo line search
at initialisation. Reference implementation for

> Ashmitha R and Jörg Frochte. *Directional Curvature from Armijo Backtracking:
> A Low-Cost Sharpness Probe and a Calibration-Free Learning-Rate Safeguard for
> Adam.* Transactions on Machine Learning Research, 2026.
> [OpenReview](https://openreview.net/forum?id=foLGwG3ne4) ·
> [arXiv:2607.03998](https://arxiv.org/abs/2607.03998) ·
> [Video](https://youtu.be/15dP4rbk3QE)

## Idea

Before the first optimiser step, run one Armijo backtracking line search on one
mini-batch along Adam's first-step direction `d = -g / (|g| + eps)`. The accepted
step `alpha` measures the directional curvature at initialisation. If the
learning rate you chose is larger than `kappa * alpha` (with `kappa = 2`), it is
lowered to that value; otherwise nothing changes. The probe costs one backward
pass and a few forward passes, and it restores parameters, buffers and the RNG
state, so a non-binding cap leaves training bit-identical.

## Usage

`armijo_probe.py` is a single file with no dependency beyond PyTorch; copy it
into your project.

```python
from armijo_probe import cap_learning_rate

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for step, (X, Y) in enumerate(train_loader):
    if step == 0:  # once, on the first mini-batch, before the first update
        cap_learning_rate(optimizer, model, lambda: criterion(model(X), Y), kappa=2.0)
    optimizer.zero_grad()
    criterion(model(X), Y).backward()
    optimizer.step()
```

`loss_fn` only has to return the loss on a fixed batch at the current
parameters, so autocast, custom losses or language models work the same way.

Where curvature rises within the first few steps (in the paper: GPT-2
pretraining), add the probationary window, which re-probes at steps
1, 2, 3, 5, 8, 12, 20, 35, 50 and can only lower the learning rate:

```python
from armijo_probe import ProbationWindow

res = cap_learning_rate(optimizer, model, loss_fn)
window = ProbationWindow(optimizer, model, loss_fn, init_alpha=res.alpha)
# after each optimizer.step():  window.after_step(step_number)  (counting from 1)
```

`armijo_probe(...)` alone returns the measured `alpha` without changing anything.
With `direction="grad"` it probes along the raw gradient; `kappa` is then
architecture-specific (see the paper).

## Example

```
pip install torch torchvision
python examples/fashion_mnist.py            # Adam at lr = 1.0, without and with probe
python examples/fashion_mnist.py --lr 1e-3  # cap does not bind: identical runs
```

Without the probe, Adam at `lr = 1.0` diverges in the first epoch; with the
probe the learning rate is capped at about 0.002 and the model reaches about
89 % test accuracy. At `lr = 1e-3` both runs print identical numbers.

Tests: `python -m pytest tests`

## Code and results of the paper

`paper/tmlr10362_paper_code.zip` contains the complete experiment code and
the frozen per-run results behind all tables and figures, as provided to the
reviewers (author names restored, otherwise unchanged). Datasets are not
included; the code downloads them.

## Citation

```bibtex
@article{r2026directional,
  title   = {Directional Curvature from Armijo Backtracking: A Low-Cost Sharpness
             Probe and a Calibration-Free Learning-Rate Safeguard for Adam},
  author  = {R, Ashmitha and Frochte, J{\"o}rg},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  url     = {https://openreview.net/forum?id=foLGwG3ne4}
}
```

## License

MIT, see `LICENSE`.
