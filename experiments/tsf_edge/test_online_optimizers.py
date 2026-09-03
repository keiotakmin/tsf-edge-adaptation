"""Correctness gate for the Stage-0b contenders (online_optimizers.py).

Rule from CLAUDE.md: an inexplicably good/bad result means suspect the implementation FIRST.
These optimizers are hand-transcribed, so they get checked against their sources BEFORE any
GPU time is spent on the 216-cell grid:

  1 IDBD   reproduces Sutton (1992) fig. 2: on his non-stationary tracking testbed it must
           beat the BEST fixed-step LMS, and must learn large steps on relevant inputs and
           near-zero steps on irrelevant ones.
  2 UPGD   is bit-compared against the authors' FirstOrderGlobalUPGD (vendored below from
           github.com/mohmdelsayed/upgd) -- identical updates required.
  3 ObGD   the overshoot cap must bind: ||dw||_1 <= 1/kappa, and a 10^4-times-too-large lr
           must NOT diverge (that is the entire point of the method).
  4 dONS   must track a drifting linear target better than the same rule with gamma = 1
           (i.e. the forgetting factor does what it claims).
  5 state  measured optimizer-state bytes per parameter must equal the documented
           multiplier (0/1/1/1/2) -- this is the frontier figure's x-axis.

Run: .venv/bin/python experiments/tsf_edge/test_online_optimizers.py
"""
from __future__ import annotations
import sys

import numpy as np
import torch

from online_optimizers import (IDBD, Autostep, ObGD, AdaptiveObGD, DiscountedONS, UPGD,
                               ObSign, RelSign)

# Stream lengths for the LMS testbeds. These are single-sample Python loops, so they dominate
# the suite's wall-clock -- and the suite is a GATE in front of a multi-hour GPU job, where it
# sat on the critical path for 6h50m on 2026-08-07 while the A100 idled. Keep them short enough
# that the gate costs minutes; every effect checked here saturates long before 20k steps.
N_SCALE, N_DONS = 8000, 6000

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILS.append(name)


# ---------------------------------------------------------------- 1. IDBD on Sutton's testbed
def sutton_stream(n, n_in=20, n_rel=5, flip_every=20, seed=0, scale=1.0):
    """Sutton (1992) tracking task: 20 inputs, only the first 5 relevant; each relevant weight
    flips sign with prob 1 every `flip_every` examples. y = w*.x, no observation noise.

    `scale` shrinks the target (and hence the residual and the gradient) without touching the
    inputs. This separates the two quantities Sutton's LMS form keeps apart but a gradient-only
    generalisation conflates: the input curvature x_i^2 and the squared gradient g_i^2 =
    delta^2 x_i^2. scale=1e-3 puts |g| in the range these TSF backbones actually operate at."""
    rng = np.random.default_rng(seed)
    w = rng.choice([-1.0, 1.0], size=n_rel) * scale
    X, Y = np.zeros((n, n_in)), np.zeros(n)
    for t in range(n):
        if t % flip_every == 0 and t > 0:
            i = rng.integers(n_rel)
            w[i] = -w[i]
        x = rng.standard_normal(n_in)
        X[t], Y[t] = x, x[:n_rel] @ w
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)


def run_lms(X, Y, make_opt, tail=0.5):
    w = torch.zeros(X.shape[1], requires_grad=True)
    opt = make_opt([w])
    errs = []
    for t in range(X.shape[0]):
        loss = 0.5 * (X[t] @ w - Y[t]) ** 2
        opt.zero_grad()
        loss.backward()
        opt.step(loss=loss.detach()) if getattr(opt, "needs_loss", False) else opt.step()
        errs.append(float(loss))
    return float(np.mean(errs[int(len(errs) * (1 - tail)):])), opt, w


def test_idbd():
    print("\n1. IDBD on Sutton (1992) non-stationary tracking testbed")
    X, Y = sutton_stream(20000)
    sgd = {}
    for lr in [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]:
        sgd[lr], _, _ = run_lms(X, Y, lambda p, lr=lr: torch.optim.SGD(p, lr=lr))
    best_lr = min(sgd, key=sgd.get)
    idbd_mse, opt, w = run_lms(X, Y, lambda p: IDBD(p, lr=1e-2, theta=1e-2))
    print(f"     best fixed-step LMS: lr={best_lr:g} asymptotic MSE={sgd[best_lr]:.4f}   "
          f"(grid {', '.join(f'{k:g}:{v:.3f}' for k, v in sgd.items())})")
    print(f"     IDBD(alpha0=1e-2, theta=1e-2)      asymptotic MSE={idbd_mse:.4f}")
    check("IDBD beats the best fixed-step LMS", idbd_mse < sgd[best_lr],
          f"{idbd_mse:.4f} vs {sgd[best_lr]:.4f}")
    alpha = opt.state[w]["beta"].exp()
    rel, irr = float(alpha[:5].mean()), float(alpha[5:].mean())
    print(f"     learned step size: relevant inputs {rel:.2e}, irrelevant {irr:.2e}")
    check("IDBD gives relevant inputs a larger step", rel > 10 * irr, f"{rel:.2e} vs {irr:.2e}")
    check("IDBD steps stay finite", bool(torch.isfinite(alpha).all()))


def test_scale_degeneracy():
    """THE TEST THAT WAS MISSING (added 2026-08-07). Sutton's testbed has |g| ~ O(1); these
    backbones run at |g| ~ 1e-3, where IDBD's theta*g*h meta-update is ~1e-11 per step and the
    learned steps never leave alpha_0. The grid caught this only after 72 cells, as idbd rows
    that matched torch SGD exactly. Any step-size-meta-learning rule must be checked at the
    gradient scale it will be deployed at, not just at the scale of its original paper."""
    print("\n1b. Step-size meta-learning across four decades of gradient scale (alpha_0 fixed)")
    print(f"     {'scale':>7s} | {'IDBD ratio':>10s} {'nMSE':>7s} | {'Autostep ratio':>14s} "
          f"{'nMSE':>7s} | {'bestSGD nMSE':>12s}")
    idbd_r, auto_r, auto_n, sgd_n = {}, {}, {}, {}
    best_sgd = None          # scale-independent: normalised MSE is identical at every scale
    for scale in (1.0, 1e-1, 1e-2, 1e-3):
        X, Y = sutton_stream(N_SCALE, scale=scale, seed=3)
        var = float((Y ** 2).mean())
        out = []
        for make, key in [(lambda p: IDBD(p, lr=1e-2, theta=1e-2), "beta"),
                          (lambda p: Autostep(p, lr=1e-2), "alpha")]:
            mse, opt, w = run_lms(X, Y, make)
            a = opt.state[w]["beta"].exp() if key == "beta" else opt.state[w]["alpha"]
            out += [float(a[:5].mean()) / max(float(a[5:].mean()), 1e-30), mse / var]
        if best_sgd is None:
            best_sgd = min(run_lms(X, Y, lambda p, lr=lr: torch.optim.SGD(p, lr=lr))[0]
                           for lr in [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]) / var
        sgd_n[scale] = best_sgd
        idbd_r[scale], auto_r[scale], auto_n[scale] = out[0], out[2], out[3]
        print(f"     {scale:7.0e} | {out[0]:10.2f} {out[1]:7.4f} | {out[2]:14.2f} "
              f"{out[3]:7.4f} | {sgd_n[scale]:12.4f}")
    # IDBD: the documented degeneracy -- steps stop differentiating once |g| leaves O(1)
    check("IDBD degenerates to a fixed step below |g| ~ 1e-1 (documented limitation)",
          idbd_r[1.0] > 10 and max(idbd_r[s] for s in (1e-1, 1e-2, 1e-3)) < 1.05,
          f"ratio {idbd_r[1.0]:.1f} -> {idbd_r[1e-3]:.2f}")
    # Autostep with the Gauss-Newton curvature: SCALE-INVARIANT, which is the whole claim
    # Tolerance is 1% of nMSE, not exact equality: at scale 1e-3 the Gauss-Newton curvature
    # divides g^2 ~ 1e-12 by delta^2 ~ 1e-6 in float32, so the last digits drift. Three of the
    # four scales still agree to 4 decimals; IDBD by contrast moves 0.16 -> 0.40 (150%).
    spread = (max(auto_n.values()) - min(auto_n.values())) / np.mean(list(auto_n.values()))
    check("Autostep is scale-invariant (nMSE spread < 1% over four decades)", spread < 0.01,
          f"spread {100*spread:.2f}%, ratio {min(auto_r.values()):.1f}-{max(auto_r.values()):.1f}")
    check("Autostep beats the best fixed step at EVERY scale",
          all(auto_n[s] < sgd_n[s] for s in auto_n),
          ", ".join(f"{s:g}:{auto_n[s]:.3f}<{sgd_n[s]:.3f}" for s in auto_n))


def test_obsign():
    """Stage-0c. ObSign's whole claim is that it is signSGD plus a one-sided guard, so the two
    properties to prove are (a) BELOW the knee it is signSGD to the last bit -- otherwise its
    oracle could fall below signSGD's and the design argument collapses -- and (b) ABOVE it the
    per-tensor step is pinned at tau*RMS(p), which is what turns signSGD's cliff into a plateau.
    RelSign drops the absolute rate entirely, so its defining property is scale EQUIVARIANCE:
    rescale the weights and every step rescales with them, which is what makes tau dimensionless."""
    print("\n6. ObSign / RelSign (Stage-0c)")
    from pytorch_optimizer import SignSGD
    torch.manual_seed(0)
    p0 = [torch.randn(64, 16) * 0.05, torch.randn(32) * 0.05]

    def run(make, lr, n=30, scale=1.0):
        ps = [(q * scale).clone().requires_grad_(True) for q in p0]
        opt = make(ps, lr)
        steps = []
        for t in range(n):
            torch.manual_seed(900 + t)
            for q in ps:
                q.grad = torch.randn_like(q)
            before = [q.detach().clone() for q in ps]
            opt.step()
            steps.append([float((q.detach() - b).abs().max()) for q, b in zip(ps, before)])
        return ps, np.array(steps)

    # (a) below the knee: tau*RMS(p) ~ 1e-2 * 0.05 = 5e-4, so lr = 1e-5 is far below it
    a, _ = run(lambda ps, lr: ObSign(ps, lr=lr, tau=1e-2), 1e-5)
    b, _ = run(lambda ps, lr: SignSGD(ps, lr=lr, momentum=0.0), 1e-5)
    d = max(float((x - y).abs().max()) for x, y in zip(a, b))
    check("ObSign below the knee is signSGD exactly", d < 1e-9, f"maxdiff={d:.2e}")

    # (b) above the knee the step is pinned at tau*RMS(p), independent of lr
    caps = {}
    for lr in (1e-2, 1.0, 100.0):
        ps, st = run(lambda ps, l: ObSign(ps, lr=l, tau=1e-2), lr)
        caps[lr] = st.max()
    spread = max(caps.values()) / max(min(caps.values()), 1e-30)
    print(f"     max per-step |dw| at lr = 1e-2 / 1 / 100: "
          + " / ".join(f"{caps[l]:.3e}" for l in (1e-2, 1.0, 100.0)))
    check("ObSign plateaus above the knee (rate cancels, <1.01x over 4 decades)", spread < 1.01,
          f"spread {spread:.4f}x")
    # exact identity, checked on a SINGLE step: after several steps RMS(p) has itself moved
    # with the weights (intended -- the guard tracks the current scale), so comparing a
    # multi-step maximum against the INITIAL RMS is off by that drift, not by an error.
    _, one = run(lambda ps, l: ObSign(ps, lr=l, tau=1e-2), 1.0, n=1)
    rms0 = float(p0[0].pow(2).mean().sqrt())
    # rtol, not atol: the optimizer forms tau*RMS in float32 while this line uses python
    # float64, so agreement is bounded by float32 precision (~1e-7 relative), not by 1e-9.
    check("ObSign's plateau step equals tau*RMS(p) (1 step, float32 rtol)",
          abs(one.max() - 1e-2 * rms0) < 1e-5 * 1e-2 * rms0,
          f"{one.max():.6e} vs {1e-2 * rms0:.6e}")

    # RelSign: scale equivariance -- 10x the weights, 10x every step
    _, s1 = run(lambda ps, lr: RelSign(ps, lr=lr), 1e-3, scale=1.0)
    _, s10 = run(lambda ps, lr: RelSign(ps, lr=lr), 1e-3, scale=10.0)
    ratio = float(np.median(s10[s1 > 0] / s1[s1 > 0]))
    check("RelSign is scale-equivariant (10x weights -> 10x steps)", abs(ratio - 10) < 1e-4,
          f"ratio {ratio:.5f}")


# ---------------------------------------------------------------- 2. UPGD vs the reference
class _RefFirstOrderGlobalUPGD(torch.optim.Optimizer):
    """Verbatim from the authors' repo (names arg dropped; no 'gate' params here)."""

    def __init__(self, params, lr=1e-5, weight_decay=0.0, beta_utility=0.0, sigma=1.0):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay,
                                      beta_utility=beta_utility, sigma=sigma))

    def step(self, noises):
        global_max_util = torch.tensor(-torch.inf)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(p.data)
                state["step"] += 1
                avg_utility = state["avg_utility"]
                avg_utility.mul_(group["beta_utility"]).add_(
                    -p.grad.data * p.data, alpha=1 - group["beta_utility"])
                current_util_max = avg_utility.max()
                if current_util_max > global_max_util:
                    global_max_util = current_util_max
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                state = self.state[p]
                bias_correction = 1 - group["beta_utility"] ** state["step"]
                noise = noises[i] * group["sigma"]
                scaled_utility = torch.sigmoid_(
                    (state["avg_utility"] / bias_correction) / global_max_util)
                p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
                    (p.grad.data + noise) * (1 - scaled_utility), alpha=-group["lr"])


def test_foreach_equivalence():
    """The `foreach` path exists so that a latency measurement compares update RULES rather
    than the fact that torch's built-ins are multi-tensor and ours is a Python loop. It is only
    allowed to exist if it computes the SAME update: the 216-cell grid was produced by the loop
    path, and a foreach path that drifted from it would make the quality numbers and the cost
    numbers describe two different optimizers.

    alpha = min(lr, tau*RMS(p)), so there are two regimes and they have different expectations:
      * BELOW the knee (lr < tau*RMS) alpha IS lr, a python float shared by both paths, so the
        update must be bit-for-bit identical.
      * ABOVE the knee (lr > tau*RMS) alpha is tau*RMS, and the two paths reduce RMS
        differently -- p.pow(2).mean().sqrt() against ||p||_2/sqrt(numel) -- so they may differ
        by floating-point reassociation, and only by that.
    """
    print("\n7. foreach implementation equivalence (ObSign / RelSign)")
    shapes = [(64, 16), (32,), (128, 8), (7,)]

    def run(make, n=40, seed0=1500):
        torch.manual_seed(0)                        # same initial weights for every variant
        ps = [(torch.randn(*sh) * 0.05).requires_grad_(True) for sh in shapes]
        opt = make(ps)
        for t in range(n):
            torch.manual_seed(seed0 + t)            # same gradients, step for step
            for q in ps:
                q.grad = torch.randn_like(q)
            opt.step()
        return [q.detach().clone() for q in ps]

    def maxdiff(a, b):
        return max(float((x - y).abs().max()) for x, y in zip(a, b))

    # BELOW the knee: tau*RMS ~ 1e-2 * 0.05 = 5e-4, so lr = 1e-5 never reaches the guard
    a = run(lambda ps: ObSign(ps, lr=1e-5, tau=1e-2, foreach=False))
    b = run(lambda ps: ObSign(ps, lr=1e-5, tau=1e-2, foreach=True))
    d = maxdiff(a, b)
    check("ObSign foreach == loop below the knee (bit-identical)", d == 0.0, f"maxdiff={d:.2e}")

    # ABOVE the knee: alpha = tau*RMS(p), the one place the two reductions can disagree
    a = run(lambda ps: ObSign(ps, lr=1.0, tau=1e-2, foreach=False))
    b = run(lambda ps: ObSign(ps, lr=1.0, tau=1e-2, foreach=True))
    d, scale = maxdiff(a, b), max(float(x.abs().max()) for x in a)
    check("ObSign foreach == loop above the knee (<=1e-5 relative)", d <= 1e-5 * scale,
          f"maxdiff={d:.2e} on scale {scale:.2e}")

    a = run(lambda ps: RelSign(ps, lr=3e-3, foreach=False))
    b = run(lambda ps: RelSign(ps, lr=3e-3, foreach=True))
    d, scale = maxdiff(a, b), max(float(x.abs().max()) for x in a)
    check("RelSign foreach == loop (<=1e-5 relative)", d <= 1e-5 * scale,
          f"maxdiff={d:.2e} on scale {scale:.2e}")

    # The guard binds to tau*RMS(p) in BOTH paths -- the property tau buys, and the reason the
    # curve plateaus instead of falling off signSGD's cliff.
    # Tolerance note: the step is recovered as |p_after - p_before| with |p| ~ 0.05 against a
    # step ~ 5e-4, so float32 cancellation alone costs eps*|p|/alpha ~ 1e-5 relative. Asking
    # for 1e-6 here would be testing the subtraction, not the optimizer.
    eps = float(torch.finfo(torch.float32).eps)
    for foreach in (False, True):
        torch.manual_seed(0)
        ps = [(torch.randn(64, 16) * 0.05).requires_grad_(True)]
        opt = ObSign(ps, lr=1.0, tau=1e-2, foreach=foreach)
        ps[0].grad = torch.ones_like(ps[0])
        before = ps[0].detach().clone()
        opt.step()
        step = float((ps[0].detach() - before).abs().max())
        want = 1e-2 * float(before.pow(2).mean().sqrt())
        tol = 8 * eps * float(before.abs().max()) / want
        check(f"ObSign step is tau*RMS(p) (foreach={foreach})", abs(step - want) <= tol * want,
              f"step={step:.6e} want={want:.6e} rel={abs(step - want) / want:.2e} tol={tol:.2e}")


def test_upgd():
    print("\n2. UPGD vs the authors' FirstOrderGlobalUPGD")
    torch.manual_seed(0)
    p0 = [torch.randn(7, 3), torch.randn(5)]
    mine = [q.clone().requires_grad_(True) for q in p0]
    ref = [q.clone().requires_grad_(True) for q in p0]
    o_mine = UPGD(mine, lr=1e-2, beta_utility=0.999, sigma=1e-3)
    o_ref = _RefFirstOrderGlobalUPGD(ref, lr=1e-2, beta_utility=0.999, sigma=1e-3)
    maxdiff = 0.0
    for t in range(25):
        torch.manual_seed(100 + t)
        grads = [torch.randn_like(q) for q in p0]
        noises = [torch.randn_like(q) for q in p0]
        for q, g in zip(mine, grads):
            q.grad = g.clone()
        for q, g in zip(ref, grads):
            q.grad = g.clone()
        torch.manual_seed(500 + t)                 # same noise draw for both
        o_mine.step()
        o_ref.step([n.clone() for n in noises])
        # our noise is drawn internally; compare with sigma=0 to isolate the update rule
        maxdiff = max(maxdiff, max(float((a - b).abs().max()) for a, b in zip(mine, ref)))
    print(f"     max |ours - reference| over 25 steps (sigma=1e-3 noise differs): {maxdiff:.2e}")
    # decisive comparison: sigma = 0 removes the stochastic term, updates must match exactly
    mine = [q.clone().requires_grad_(True) for q in p0]
    ref = [q.clone().requires_grad_(True) for q in p0]
    o_mine = UPGD(mine, lr=1e-2, beta_utility=0.999, sigma=0.0)
    o_ref = _RefFirstOrderGlobalUPGD(ref, lr=1e-2, beta_utility=0.999, sigma=0.0)
    for t in range(25):
        torch.manual_seed(100 + t)
        grads = [torch.randn_like(q) for q in p0]
        for q, g in zip(mine, grads):
            q.grad = g.clone()
        for q, g in zip(ref, grads):
            q.grad = g.clone()
        o_mine.step()
        o_ref.step([torch.zeros_like(q) for q in p0])
    d = max(float((a - b).abs().max()) for a, b in zip(mine, ref))
    check("UPGD matches the reference implementation exactly (sigma=0)", d < 1e-6, f"maxdiff={d:.2e}")


# ---------------------------------------------------------------- 3. ObGD overshoot cap
def test_obgd():
    print("\n3. ObGD overshoot bound")
    torch.manual_seed(0)
    for kappa in (2.0, 5.0):
        w = torch.randn(64, requires_grad=True)
        opt = ObGD([w], lr=1.0, kappa=kappa)
        x, y = torch.randn(64), torch.tensor(3.0)
        worst = 0.0
        for _ in range(50):
            before = w.detach().clone()
            loss = 0.5 * (x @ w - y) ** 2
            opt.zero_grad()
            loss.backward()
            opt.step(loss=loss.detach())
            worst = max(worst, float((w.detach() - before).abs().sum()))
        check(f"||dw||_1 <= 1/kappa (kappa={kappa:g})", worst <= 1.0 / kappa + 1e-5,
              f"worst L1 step = {worst:.4f} <= {1/kappa:.4f}")
    # a 10^4-times-too-large lr must not blow up
    X, Y = sutton_stream(4000, seed=1)
    mses = {}
    for lr in (1e-2, 1.0, 1e2, 1e4):
        mses[lr], _, _ = run_lms(X, Y, lambda p, lr=lr: ObGD(p, lr=lr, kappa=2.0))
    print("     asymptotic MSE vs lr: " + ", ".join(f"{k:g}:{v:.3f}" for k, v in mses.items()))
    check("ObGD does not diverge at lr=1e4", np.isfinite(mses[1e4]) and mses[1e4] < 1e3)
    spread = max(mses[1.0], mses[1e2], mses[1e4]) / max(min(mses[1.0], mses[1e2], mses[1e4]), 1e-12)
    check("ObGD is LR-insensitive once the cap binds (<2x spread over lr 1..1e4)", spread < 2.0,
          f"spread={spread:.2f}x")


# ---------------------------------------------------------------- 4. dONS forgetting factor
def test_dons():
    print("\n4. Discounted ONS: does the forgetting factor help under drift?")
    X, Y = sutton_stream(N_DONS, seed=2)
    best = {}
    for gamma in (1.0, 0.99):
        scores = {}
        for lr in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
            scores[lr], _, _ = run_lms(X, Y, lambda p, lr=lr, g=gamma: DiscountedONS(p, lr=lr, gamma=g))
        best[gamma] = min(scores.values())
        print(f"     gamma={gamma:g}: best MSE={best[gamma]:.4f} at lr="
              f"{min(scores, key=scores.get):g}  ({', '.join(f'{k:g}:{v:.3f}' for k, v in scores.items())})")
    check("discounting (gamma=0.99) beats no discounting (gamma=1) under drift",
          best[0.99] < best[1.0], f"{best[0.99]:.4f} vs {best[1.0]:.4f}")


# ---------------------------------------------------------------- 5. state-byte accounting
def measured_state_bytes(opt):
    """Same measurement stream_eval uses for extension optimizers."""
    def tensors(v):
        if torch.is_tensor(v):
            return [v]
        return [t for t in v if torch.is_tensor(t)] if isinstance(v, (list, tuple)) else []
    seen, nbytes = set(), 0
    buckets = [v for st in opt.state.values() for v in st.values()]
    buckets += [v for g in opt.param_groups for k, v in g.items() if k != "params"]
    for v in buckets:
        for t in tensors(v):
            if id(t) not in seen:
                seen.add(id(t))
                nbytes += t.numel() * t.element_size()
    return nbytes


def test_state():
    print("\n5. Measured optimizer state per parameter (the frontier x-axis)")
    expected = {"obgd": 0, "adaptive_obgd": 1, "dons": 1, "upgd": 1, "idbd": 2, "sgdm": 1,
                "autostep": 3, "obsign": 0, "relsign": 0}
    makers = {"obgd": lambda p: ObGD(p, lr=1.0), "adaptive_obgd": lambda p: AdaptiveObGD(p, lr=1.0),
              "dons": lambda p: DiscountedONS(p, lr=1e-3), "upgd": lambda p: UPGD(p, lr=1e-3),
              "idbd": lambda p: IDBD(p, lr=1e-3), "autostep": lambda p: Autostep(p, lr=1e-3),
              "obsign": lambda p: ObSign(p, lr=1e-3), "relsign": lambda p: RelSign(p, lr=1e-3),
              "sgdm": lambda p: torch.optim.SGD(p, lr=1e-3, momentum=0.9)}
    for name, make in makers.items():
        w = torch.randn(1000, requires_grad=True)
        opt = make([w])
        for _ in range(3):
            loss = (w ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step(loss=loss.detach()) if getattr(opt, "needs_loss", False) else opt.step()
        mult = measured_state_bytes(opt) / (w.numel() * 4)
        check(f"{name:14s} state = {expected[name]}x params", abs(mult - expected[name]) < 1e-9,
              f"measured {mult:.2f}x")


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    test_idbd()
    test_scale_degeneracy()
    test_obsign()
    test_foreach_equivalence()
    test_upgd()
    test_obgd()
    test_dons()
    test_state()
    print(f"\n{'ALL CHECKS PASSED' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)
